# ComfyUI Workflow for Graphic Novel — AI Image Generation

Modular ComfyUI generation pipelines for **panel-to-panel continuity** across
sequential graphic novels.

A 200-page novel needs identity, environment and camera angle handled as three
separate problems. Text prompts alone drift: the same character's face, costume
and proportions shift between panels, and the room changes shape when the
camera moves. This pipeline decouples them.

---

## Architecture

### Identity — two tiers, chosen by cost not preference

| Tier | Cast | Method | Why |
|---|---|---|---|
| **1** | 4–6 leads | Dedicated **FLUX.1 [dev] LoRAs**, rank 32, trained on multi-angle turnaround sheets | Locks facial bone structure, costume silhouette and aged variants across extreme angles. Worth the training cost only for characters on many pages. |
| **2** | 15–20 secondary | **FLUX-PuLID + IP-Adapter**, zero-shot | Drop in a reference portrait on the fly. No weight training, so a character added mid-book costs minutes, not hours. |

### Scene continuity

**ControlNet Union** (OpenPose + Depth) plus latent regional prompting. Pose
and depth fix where bodies and architecture sit, so a room keeps its geometry
as the camera moves through a scene and characters stay put relative to it.

### Post-processing

Every panel: **FaceDetailer / SAM** pass to remove facial warping at small
panel scale, then a **tile upscaler** calibrated to print resolution
(300 DPI at final trim size).

---

## Deliverables

1. **Annotated ComfyUI workflow** (`.json`) with bypass switches for solo,
   multi-character and action panels
2. **Dataset configs and trained LoRA weights** for the lead cast
3. **Cloud deployment template** (RunPod / Vast.ai) plus a recorded walkthrough

---

## Five phases

| # | Phase | Output | Needs a GPU? |
|---|---|---|---|
| 1 | **Dataset preparation** | 20–30 curated turnarounds per lead, tagged, face-cropped, validated | No |
| 2 | **LoRA training & validation** | Rank-32 FLUX LoRAs, cross-epoch checkpoints benchmarked against overfitting | Yes — 24 GB+ |
| 3 | **Modular node architecture** | The master ComfyUI graph | Yes, to run |
| 4 | **Stress-testing** | Multi-panel sequences across angles, casts and lighting | Yes |
| 5 | **Packaging & hand-off** | Colour-coded graph, dependency manifest, RunPod template, video | No |

---

## Repository layout

```
01_dataset/     Curation, validation, tagging, face crops
02_training/    LoRA configs (AI-Toolkit / Kohya) and benchmark prompts
03_workflows/   ComfyUI graphs (.json), annotated
04_deploy/      RunPod / Vast.ai template, dependency manifest
docs/           RUNBOOK, client-facing hand-off notes
```

---

## Hardware reality

This is the constraint that shapes the schedule.

| Task | Minimum VRAM | Practical |
|---|---|---|
| FLUX.1 [dev] inference | 16 GB (quantised: 12 GB) | 24 GB |
| FLUX LoRA training, rank 32 | 24 GB | 40–48 GB |
| ControlNet + PuLID + detailer stack | 24 GB | 48 GB |

**The local RTX 3050 (4 GB) cannot run any of it.** It is useful only for
Phase 1 and Phase 5, which are CPU work. Phases 2–4 need rented GPUs:

| GPU | ~Cost/hr | Use |
|---|---|---|
| RTX 4090 (24 GB) | $0.34–0.44 | Inference, workflow development |
| A100 40 GB | $1.20–1.60 | LoRA training |
| H100 80 GB | $2.50–3.50 | Fast training if the schedule is tight |

Rough budget for one lead character's LoRA: 2–4 hours on an A100, about
**$4–6**. Five leads: **$20–30**, plus iteration.

---

## Status

Phase 1 is being built first because it is the only phase that can be fully
tested without a GPU, and because dataset quality determines everything
downstream — a LoRA trained on inconsistent references cannot be rescued by
better prompts later.

Workflow JSONs and training configs written before GPU access are **drafts
until executed**. They will be marked as such rather than presented as
verified.
