# Motion branch — object-centric motion representation

Produces the **Motion** representation: an object-level optical-flow direction
histogram per detected object per frame (the Object Dynamics attribute). Three
stages, run in order. The scripts live in `feature_extraction/motion/`.

## Dependencies

```bash
pip install -r feature_extraction/motion/requirements.txt
# FlowNet2 checkpoint + custom CUDA extensions (needs nvcc):
mkdir -p feature_extraction/motion/checkpoints
wget -O feature_extraction/motion/checkpoints/FlowNet2_checkpoint.pth.tar \
  https://github.com/NVIDIA/flownet2-pytorch/raw/master/checkpoints/FlowNet2_checkpoint.pth.tar
./feature_extraction/motion/install_custom_layers.sh
# Detectron2 (bboxes step) + Mask R-CNN weights:
pip install 'git+https://github.com/facebookresearch/detectron2.git'
wget -O feature_extraction/motion/checkpoints/model_final_f10217.pkl \
  https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```

All commands below run from `feature_extraction/motion/`. `--root` points at the
dataset root (containing `DATA_ROOT/<ds>/{training,testing}/frames`).

## Step 1 — optical flow

```bash
python flows.py --root DATA_ROOT --dataset avenue --train
python flows.py --root DATA_ROOT --dataset avenue
python flows.py --root DATA_ROOT --dataset shanghaitech --train
```

Writes `DATA_ROOT/<ds>/{training,testing}/flows/<video>/<frame>.jpg.npy`
(`[H,W,2]` float32; the filename keeps the `.jpg`).

## Step 2 — object + foreground bounding boxes

```bash
python bboxes.py --root DATA_ROOT --dataset avenue --train
python bboxes.py --root DATA_ROOT --dataset avenue
```

Per-dataset thresholds are in `bboxes.py::DATASET_CFGS`. Writes
`DATA_ROOT/<ds>/<ds>_bboxes_{train,test}.npy` (+ `_classes` for shanghaitech),
one entry per frame.

## Step 3 — motion direction histogram

```bash
# bins per dataset: ped2 -> 1 , avenue -> 8 , shanghaitech -> 8
python extract_motion.py --root DATA_ROOT --out_root <MOTION_ROOT> \
    --dataset avenue --bins 8
python extract_motion.py --root DATA_ROOT --out_root <MOTION_ROOT> \
    --dataset ped2 --bins 1
```

Writes `<MOTION_ROOT>/<ds>/{train,test}/motion_bins<bins>.npy`.

## Rename for the main package

`mrcvad/main.py` loads the Motion representation from a plain filename
`<MOTION_ROOT>/<ds>/{train,test}/motion.npy`:

```bash
mv <MOTION_ROOT>/avenue/train/motion_bins8.npy <MOTION_ROOT>/avenue/train/motion.npy
mv <MOTION_ROOT>/avenue/test/motion_bins8.npy  <MOTION_ROOT>/avenue/test/motion.npy
```

Then run with `--motion_root <MOTION_ROOT>`.
