# Dataset organization

MRC-VAD consumes three feature representations. Everything below refers to a
**dataset root** `DATA_ROOT` holding the raw frames, the ground-truth labels and
the extracted feature directories.

Supported benchmarks: **ped2** (UCSD Ped2), **avenue** (CUHK Avenue),
**shanghaitech** (ShanghaiTech Campus).

## Directory layout

```
DATA_ROOT/
├── ped2/
│   ├── ped2.mat                     # ground truth: MATLAB array 'gt' (1-indexed start/end per video)
│   ├── training/frames/<video>/*.jpg
│   ├── testing/frames/<video>/*.jpg
├── avenue/
│   ├── avenue.mat                   # same format as ped2.mat
│   ├── training/frames/<video>/*.jpg
│   ├── testing/frames/<video>/*.jpg
├── shanghaitech/
│   ├── training/frames/<video>/*.jpg
│   ├── testing/frames/<video>/*.jpg
│   └── testing/test_frame_mask/<video>.npy     # frame-level binary masks
```

Raw frames are stored per video as `frames/<video>/<frame>.jpg`; every extractor
starts from these directories.

## Extracted features

```
DATA_ROOT/<ds>/{training,testing}/<FEATURE_DIR>/<video>/<frame_index>.pt
```

Each `.pt` is a dict:
```python
{
  "feature": torch.Tensor,
  "metadata": {"video_seq": str, "frame_idx": int},
}
```

| representation | FEATURE_DIR                   | dim  | extractor                        |
|----------------|-------------------------------|------|----------------------------------|
| Semantic       | `features_RGB_L_offi_gpu`     | 1152 | `feature_extraction/hiera/`      |
| Continuity     | `continuity_cjhead_base_512d` | 1536 | `feature_extraction/continuity/` |

The Motion representation is object-level and is stored as one NumPy file per
split; see `feature_motion.md`.
