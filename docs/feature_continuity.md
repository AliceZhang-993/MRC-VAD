# Continuity branch — sequence-centric continuity representation

Produces the **Continuity** representation (1536-dim), reflecting the Temporal
Organization attribute: a self-supervised temporal head on an I3D backbone,
learned from continuity-based proxy tasks.

## Dependencies & weights

```bash
pip install -r feature_extraction/continuity/requirements.txt
```

Weights go under `feature_extraction/continuity/checkpoints/`: the trained model
at `exp3/<dataset>.pth` (in `continuity-checkpoints.zip`) and the I3D
initialisation at `rgb_imagenet.pt` (see the README for both).

Edit `config/experiments.yaml`:
- `continuity_learning.pretrained_path` → your `rgb_imagenet.pt` path
  (default points at `checkpoints/rgb_imagenet.pt`);
- `data.data_root` → your dataset root; `data.dataset_name` → ped2/avenue/shanghaitech;
- `hardware.*.batch_size` → lower if you run out of memory.

## Train

```bash
cd feature_extraction/continuity
python main_epoch.py --mode train --exp exp3 --gpus 0
```

## Extract features

```bash
python main_epoch.py --mode test --exp exp3 --gpus 0
```

`test.infer_split: both` runs both splits. Writes under
`DATA_ROOT/<ds>/<split>/`:

- `continuity/` — backbone features (1024-d)
- `cjhead_base_512d/` — Continuity-head base features (512-d)

Each file is `<video>/<frame_idx>.pt` containing
`{"feature", "metadata": {"video_seq", "frame_idx"}}`.

## Build the combined 1536-d feature

```bash
python data/combine_backbone_head.py --root DATA_ROOT --dataset avenue
```

This concatenates the two feature dirs into
`DATA_ROOT/<ds>/<split>/continuity_cjhead_base_512d/`.
