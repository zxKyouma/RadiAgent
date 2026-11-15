# RadiAgent

**Report-guided lesion segmentation for patient-facing radiology explanations.**

RadiAgent reads a radiology report, works out which finding it describes, locates and
segments that finding on the scan, and assembles a package a clinician can review before
it is shown to a patient. The current scope is a single dominant intracranial mass on
brain MRI (meningioma).

Status: research prototype. Not a medical device and not for clinical use.

## How it works

```
report text
   -> structured target   (finding, anatomy, laterality, size)
   -> support check       (single lesion: proceed; multiple, diffuse or post-operative: block)
   -> localisation        (shortlist slices that match the target)
   -> box proposals       (coarse per-slice boxes)
   -> segmentation        (Medical SAM3 refinement)
   -> QA + clinician review
   -> patient summary
```

Cases the system does not support are blocked with a reason instead of being segmented
anyway, and every result waits in a review queue for a clinician to approve or reject.

## What is in the repository

| Path | Contents |
|---|---|
| `apps/api` | Fastify API: create cases, run processing, fetch assets, record reviews |
| `apps/ui` | React + Vite review queue for clinicians |
| `apps/worker` | Queue worker, dataset import and study preprocessing commands |
| `packages/core` | Case store, manifests and the processing pipeline |
| `packages/shared` | Shared schemas and types |
| `data/manifests` | Dataset, case and split manifests (BraTS meningioma subset, demo case) |
| `external/Medical-SAM3/inference/radiant_pipeline` | Python research pipeline: finding extraction, localisation, box proposal, SAM3 refinement, candidate scoring, with BiomedCLIP and MedSigLIP backends |
| `external/Medical-SAM3/inference/run_brain_mri_*.py` | Experiments: localiser evaluation, reranking, VLM pointing, end-to-end evaluation |

The TypeScript app currently runs a demo pipeline: report parsing is rule-based and the
segmentation step returns demo output. The model-backed pipeline lives in the Python
research code and is not yet wired into the app.

## Quick start

```bash
npm install
npm run dev:api      # API on :3001
npm run dev:worker   # queue worker
npm run dev:ui       # review UI
```

Python research pipeline:

```bash
conda env create -f environment.yml
conda activate radiant
```

Other commands:

```bash
npm run typecheck
npm run import:dataset -- <args>
npm run preprocess:study -- <args>
```

## Data

No imaging data are included. The manifests reference a five-case subset of the public
BraTS meningioma release. BraTS has no paired clinical reports, so the report text used
in development is synthetic. Never commit real patient data to `data/assets`.

## Licence

MIT for the code in this repository (see `LICENSE`). The vendored SAM3 code under
`external/Medical-SAM3/sam3` keeps its own licence (see the `LICENSE` file there).
