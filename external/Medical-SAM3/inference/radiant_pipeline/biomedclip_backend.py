from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from open_clip import create_model_from_pretrained, get_tokenizer

from .brain_mri_retrieval import BrainMriRetrievalBackend, RetrievalSlab, build_slab_rgb_preview
from .types import StructuredTarget, StudyContext


@dataclass(slots=True)
class BiomedClipBackendConfig:
    model_id: str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    context_length: int = 256
    batch_size: int = 16
    device: str = 'cuda'
    query_template: str = 'this is a brain MRI of {query}'
    mixed_precision: bool = True


class BiomedClipRetrievalBackend(BrainMriRetrievalBackend):
    """BiomedCLIP-backed slab retrieval for brain MRI windows."""

    backend_name = 'biomedclip'

    def __init__(self, config: Optional[BiomedClipBackendConfig] = None):
        self.config = config or BiomedClipBackendConfig()
        requested_device = self.config.device
        if requested_device == 'cuda' and not torch.cuda.is_available():
            requested_device = 'cpu'
        self.device = torch.device(requested_device)
        self.model = None
        self.preprocess = None
        self.tokenizer = None

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
            text_tokens = self.tokenizer([query], context_length=self.config.context_length).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            all_scores: List[float] = []
            for start in range(0, len(slabs), self.config.batch_size):
                batch = slabs[start:start + self.config.batch_size]
                images = [self.preprocess(_to_pil(build_slab_rgb_preview(context.image_volume, slab))) for slab in batch]
                image_tensor = torch.stack(images).to(self.device)
                if self.config.mixed_precision and self.device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        image_features = self.model.encode_image(image_tensor)
                else:
                    image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                if image_features.dtype != text_features.dtype:
                    image_features = image_features.to(text_features.dtype)
                scores = (image_features @ text_features.T).squeeze(-1)
                all_scores.extend(float(score) for score in scores.detach().cpu())
        return all_scores

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        model, preprocess = create_model_from_pretrained(self.config.model_id)
        tokenizer = get_tokenizer(self.config.model_id)
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocess = preprocess
        self.tokenizer = tokenizer

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


def _to_pil(rgb: np.ndarray):
    from PIL import Image

    return Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode='RGB')
