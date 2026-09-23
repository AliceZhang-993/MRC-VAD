# Semantic branch — frame-centric semantic representation

Produces the **Semantic** representation: a frame-level RGB embedding from
Hiera-Large (1152-dim), reflecting the Semantic Identity attribute.

## Dependencies

```bash
pip install -r feature_extraction/hiera/requirements.txt
# the hiera package itself (torch.hub downloads hiera_large_16x224.pth on first run):
pip install 'git+https://github.com/facebookresearch/hiera.git'
#   or: export PYTHONPATH=/path/to/facebookresearch/hiera
```

## Usage

```bash
python feature_extraction/hiera/extract_hiera.py \
    --root DATA_ROOT --dataset avenue --split training --gpu 0
python feature_extraction/hiera/extract_hiera.py \
    --root DATA_ROOT --dataset avenue --split testing  --gpu 0
```

Input: `DATA_ROOT/<ds>/<split>/frames/<video>/*.jpg`.
Output: one `.pt` per input frame, same index, under
`DATA_ROOT/<ds>/<split>/features_RGB_L_offi_gpu/<video>/`.
Each `.pt` holds `{"feature": Tensor[1152], "metadata": {"video_seq", "frame_idx"}}`.
