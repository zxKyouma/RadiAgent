from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .brain_mri_retrieval import BrainMriRetrievalBackend, RetrievalSlab, build_slab_rgb_preview
from .types import StructuredTarget, StudyContext


@dataclass(slots=True)
class MedSiglipBackendConfig:
    model_id: str = 'google/medsiglip-448'
    batch_size: int = 8
    device: str = 'cuda'
    query_template: str = 'this is a brain MRI of {query}'
    mixed_precision: bool = True


class MedSiglipRetrievalBackend(BrainMriRetrievalBackend):
    """MedSigLIP-backed slab retrieval for brain MRI windows."""

    backend_name = 'medsiglip'

    def __init__(self, config: Optional[MedSiglipBackendConfig] = None):
        self.config = config or MedSiglipBackendConfig()
        requested_device = self.config.device
        if requested_device == 'cuda' and not torch.cuda.is_available():
            requested_device = 'cpu'
        self.device = torch.device(requested_device)
        self.model = None
        self.processor = None

    def score_slabs(
        self,
        context: StudyContext,
        target: StructuredTarget,
        slabs: Sequence[RetrievalSlab],
    ) -> List[float]:
        if not slabs:
            return []
        self._ensure_loaded()
        query = self._build_query(target)

        with torch.no_grad():
            text_inputs = self.processor(text=[query], padding='max_length', return_tensors='pt')
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_features = self._get_text_features(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            all_scores: List[float] = []
            for start in range(0, len(slabs), self.config.batch_size):
                batch = slabs[start:start + self.config.batch_size]
                images = [Image.fromarray(build_slab_rgb_preview(context.image_volume, slab), mode='RGB') for slab in batch]
                image_inputs = self.processor(images=images, return_tensors='pt')
                image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
                image_features = self._get_image_features(image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                if image_features.dtype != text_features.dtype:
                    image_features = image_features.to(text_features.dtype)
                scores = (image_features @ text_features.T).squeeze(-1)
                all_scores.extend(float(score) for score in scores.detach().cpu())
        return all_scores

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        self.model = AutoModel.from_pretrained(self.config.model_id).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)

    def _get_text_features(self, text_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(self.model, 'get_text_features'):
            return self.model.get_text_features(**text_inputs)
        outputs = self.model(**text_inputs)
        if hasattr(outputs, 'text_embeds') and outputs.text_embeds is not None:
            return outputs.text_embeds
        raise RuntimeError('MedSigLIP backend could not extract text features')

    def _get_image_features(self, image_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.config.mixed_precision and self.device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                return self._forward_image_features(image_inputs)
        return self._forward_image_features(image_inputs)

    def _forward_image_features(self, image_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(self.model, 'get_image_features'):
            return self.model.get_image_features(**image_inputs)
        outputs = self.model(**image_inputs)
        if hasattr(outputs, 'image_embeds') and outputs.image_embeds is not None:
            return outputs.image_embeds
        raise RuntimeError('MedSigLIP backend could not extract image features')

    def _build_query(self, target: StructuredTarget) -> str:
        parts: List[str] = []
        if target.laterality and target.laterality != 'unknown':
            parts.append(target.laterality)
        if target.sub_anatomy:
            parts.append(target.sub_anatomy)
        if target.finding:
            parts.append(target.finding)
        elif target.anatomy:
            parts.append(target.anatomy)
        query = ' '.join(str(part).strip() for part in parts if str(part).strip())
        if not query:
            query = 'intracranial mass'
        return self.config.query_template.format(query=query)
