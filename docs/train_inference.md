# MRC-VAD: training & inference

```bash
pip install -r mrcvad/requirements.txt
```

Run from `mrcvad/` (or with `mrcvad/` on `PYTHONPATH`).

## Evaluate

With the pretrained weights placed under `mrcvad/checkpoints/<dataset>/`
(see the README), no checkpoint path is needed on the command line:

```bash
python main.py --evaluate_only \
    --dataset_name avenue \
    --data_root DATA_ROOT \
    --motion_root MOTION_ROOT \
    --gpus 0 \
    --sigma_len 7
```

To use your own checkpoints instead, override a branch with `branch:path`:

```bash
python main.py --evaluate_only --dataset_name avenue \
    --data_root DATA_ROOT --motion_root MOTION_ROOT --gpus 0 --sigma_len 7 \
    --features semantic:/path/to/semantic.pth \
               motion:/path/to/motion.pth \
               continuity:/path/to/continuity.pth
```

- `--features` selects which branches to use and optionally overrides their
  checkpoints. Default: all three, with the default checkpoint paths.
- feature folders are read from
  `DATA_ROOT/<ds>/{training,testing}/<feature>/` (see
  `dataset_organization.md`). The Motion branch is read from
  `<MOTION_ROOT>/<ds>/{train,test}/motion.npy` instead. 

## Train from scratch

```bash
python main.py --dataset_name avenue \
    --data_root DATA_ROOT --motion_root MOTION_ROOT --gpus 0 --sigma_len 7
```

