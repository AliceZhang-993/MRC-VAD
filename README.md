# MRC-VAD

Official PyTorch implementation of the paper:
**"Event-Attribute-Based Multi­Representation Complementarity for Video Anomaly Detection"** (Under Review).

---

### News
- **[September 2026]** Repository created. The code, pretrained checkpoints, and dataset preparation instructions are now released.

---

Video anomaly detection aims to identify events that deviate from normal patterns
in surveillance videos. Anomalous events are typically rare, diverse, and
difficult to enumerate, so most methods are trained only on normal data. Since
such events involve complex variations in appearance, motion and temporal
patterns, a single representation can hardly capture all of them.

MRC-VAD takes the **event** as the organizing unit of analysis. An anomalous
event can differ from normal events in several aspects: *what* it is, *how* it
moves, and *how* it evolves. We organize these aspects as three **event
attributes** — **Semantic Identity**, **Object Dynamics** and **Temporal
Organization** — which define the functional roles of the corresponding
representations in characterizing an event. MRC-VAD instantiates each attribute
as a representation that provides a complementary view of the event, and models
the three independently within a density-estimation framework, combining their
anomaly scores after a unified normalization step.

| branch         | attribute                                | representation                          | dim  | extractor                        |
|----------------|------------------------------------------|-----------------------------------------|------|----------------------------------|
| **Semantic**   | What it is (Semantic Identity)           | frame-centric semantic representation   | 1152 | `feature_extraction/hiera/`      |
| **Motion**     | How it moves (Object Dynamics)           | object-centric motion representation    | 8*   | `feature_extraction/motion/`     |
| **Continuity** | How it evolves (Temporal Organization)   | sequence-centric continuity representation | 1536 | `feature_extraction/continuity/` |

\* `ped2` uses 1 bin; `avenue` / `shanghaitech` use 8.

Frame-level AUC on UCSD Ped2 / CUHK Avenue / ShanghaiTech is reported in the
paper.

## Repository layout

```
README.md
LICENSE
docs/
  dataset_organization.md    # dataset root layout, ground truth, feature loading
  feature_hiera.md           # Semantic branch
  feature_motion.md          # Motion branch
  feature_continuity.md      # Continuity branch
  train_inference.md         # training / inference / evaluate_only commands
feature_extraction/
  count_feature.py
  hiera/       extract_hiera.py + requirements
  motion/      flows.py bboxes.py extract_motion.py object_detector.py video_dataset.py
               flownet_networks/ + install_custom_layers.sh
  continuity/  main_epoch.py config/ models/ data/ exps/exp3/ utils/
               checkpoints/    # trained Continuity models (download)
mrcvad/        main.py + models.py dataset.py dataloader.py
               checkpoints/    # pretrained density networks (download)
```

## Pretrained weights

The pretrained weights are hosted as two archives on
[Baidu Netdisk](https://pan.baidu.com/s/1fzEBnvG_VbGSbT46qPQlPQ)
(extraction code: `p3xb`). Both use repository-relative paths, so unzip them at
the repository root and the files land where the code expects them:

```bash
unzip mrcvad-checkpoints.zip     -d /path/to/MRC-VAD
unzip continuity-checkpoints.zip -d /path/to/MRC-VAD
```

| archive                    | contents                                                                        |
|----------------------------|---------------------------------------------------------------------------------|
| `mrcvad-checkpoints.zip` (9 files, ~2.0 GB) | the trained log-density network of each branch, per dataset: `mrcvad/checkpoints/{ped2,avenue,shanghaitech}/{semantic,motion,continuity}.pth` |
| `continuity-checkpoints.zip` (3 files, ~1.4 GB) | the trained Continuity model of each dataset: `feature_extraction/continuity/checkpoints/exp3/{ped2,avenue,shanghaitech}.pth` |

Three further weights are publicly available from their original sources — please
download them and place each at the path listed below:

| weight | target path | source |
|--------|-------------|--------|
| I3D initialisation (`rgb_imagenet.pt`) | `feature_extraction/continuity/checkpoints/rgb_imagenet.pt` | [piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d) (`models/`) |
| FlowNet2 (`FlowNet2_checkpoint.pth.tar`) | `feature_extraction/motion/checkpoints/FlowNet2_checkpoint.pth.tar` | [NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch) |
| Mask R-CNN R50-FPN (`model_final_f10217.pkl`) | `feature_extraction/motion/checkpoints/model_final_f10217.pkl` | [Detectron2 model zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) |

The FlowNet2 and Mask R-CNN weights are only needed by the Motion extractor
(`docs/feature_motion.md`).

## Quick start

1. Prepare a dataset root `DATA_ROOT` with raw frames and ground truth
   (`docs/dataset_organization.md`).
2. Extract the three branches, in order: Motion (`flows.py` -> `bboxes.py` ->
   `extract_motion.py`), Semantic (`extract_hiera.py`), and Continuity
   (`main_epoch.py --mode test` -> `combine_backbone_head.py`). Each branch has
   its own docs page and `requirements.txt`.
3. Download the pretrained weights and place them as shown above, then run:
   ```bash
   pip install -r mrcvad/requirements.txt
   # evaluate
   python mrcvad/main.py --evaluate_only \
       --dataset_name avenue --data_root DATA_ROOT \
       --motion_root MOTION_ROOT --gpus 0 --sigma_len 7
   # train from scratch instead
   python mrcvad/main.py \
       --dataset_name avenue --data_root DATA_ROOT \
       --motion_root MOTION_ROOT --gpus 0 --sigma_len 7
   ```
   See `docs/train_inference.md` for the per-dataset flags and for overriding a
   branch's checkpoint.

## Acknowledgements

We thank the authors of the following works, which this project builds on:

- **MULDE** — J. Micorek, H. Possegger, D. Narnhofer, H. Bischof, M. Kozinski,
  "MULDE: Multiscale Log-Density Estimation via Denoising Score Matching for
  Video Anomaly Detection", CVPR 2024.
  https://github.com/jakubmicorek/MULDE-Multiscale-Log-Density-Estimation-via-Denoising-Score-Matching-for-Video-Anomaly-Detection
- **AttriVAD** — T. Reiss, D. Hoshen, "An Attribute-based Method for Video
  Anomaly Detection", TMLR 2025.
  https://github.com/talreiss/Accurate-Interpretable-VAD
- **hf2vad** — Z. Liu et al., "HF$^2$-VAD: A Hierarchical Framework for
  Video Anomaly Detection", ECCV 2022.
  https://github.com/LiUzHiAn/hf2vad
- **FlowNet2** — E. Ilg, N. Mayer, T. Saikia, M. Keuper, A. Dosovitskiy,
  T. Brox, "FlowNet 2.0: Evolution of Optical Flow Estimation with Deep
  Networks", CVPR 2017; original implementation by NVIDIA.
  https://github.com/NVIDIA/flownet2-pytorch
- **Hiera** — facebookresearch/hiera (Hiera-Large backbone); the K400-finetuned
  weights are downloaded by torch.hub. https://github.com/facebookresearch/hiera
- **I3D** — "Quo Vadis, Action Recognition? A New Model and the Kinetics
  Dataset", CVPR 2017. https://github.com/piergiaj/pytorch-i3d
- **Detectron2** — Mask R-CNN detector for the object-level branch.
  https://github.com/facebookresearch/detectron2

## License

This repository is released for **research / non-commercial use** — see
`LICENSE`. Third-party components retain their own licences.
