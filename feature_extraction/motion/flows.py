"""Estimate optical flow for the motion branch with FlowNet2 (Ilg et al., CVPR 2017).

Writes one flow file per frame to
``<root>/<dataset>/{training,testing}/flows/<video>/<frame>.jpg.npy``
(``[H,W,2]`` float32, resized back to the native frame size).
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from video_dataset import VideoDataset
from flownet_networks.flownet2_models import FlowNet2

# FlowNet2 runs at a fixed input resolution per dataset; the predicted flow is
# resized back to the native frame size afterwards.
FLOWNET_INPUT_WIDTH = {"ped2": 512 * 2, "avenue": 512 * 2, "shanghaitech": 1024}
FLOWNET_INPUT_HEIGHT = {"ped2": 384 * 2, "avenue": 384 * 2, "shanghaitech": 640}


def _count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def extracting_flows(dataset_name, root, train, weights_path, device):
    if train:
        of_save_dir = os.path.join(root, dataset_name, "training", "flows")
    else:
        of_save_dir = os.path.join(root, dataset_name, "testing", "flows")

    dataset = VideoDataset(dataset_name=dataset_name, root=root, train=train, sequence_length=1,
                           bboxes_extractions=True)

    WIDTH, HEIGHT = FLOWNET_INPUT_WIDTH[dataset_name], FLOWNET_INPUT_HEIGHT[dataset_name]

    flownet2 = FlowNet2()
    pretrained_dict = torch.load(weights_path)['state_dict']
    model_dict = flownet2.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    flownet2.load_state_dict(model_dict)
    flownet2.to(device)
    flownet2.eval()

    total_params, trainable_params = _count_params(flownet2)
    print(f'[FlowNet2] params: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M')

    dataset_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    for idx, (batch, _) in tqdm(enumerate(dataset_loader), total=len(dataset)):
        cur_img_addr = dataset.frame_addresses[idx]
        cur_img_name = cur_img_addr.split('/')[-1]

        video_of_path = os.path.join(of_save_dir, cur_img_addr.split('/')[-2])
        if os.path.exists(video_of_path) is False:
            os.makedirs(video_of_path, exist_ok=True)

        cur_imgs = np.transpose(batch[0].numpy(), [0, 2, 3, 1])
        old_size = (cur_imgs.shape[2], cur_imgs.shape[1])

        im1 = cv2.resize(cur_imgs[0], (WIDTH, HEIGHT))
        im2 = cv2.resize(cur_imgs[1], (WIDTH, HEIGHT))

        ims = np.array([im1, im2]).astype(np.float32)
        ims = torch.from_numpy(ims).unsqueeze(0).to(device)
        ims = ims.permute(0, 4, 1, 2, 3).contiguous()

        pred_flow = flownet2(ims).cpu().data
        pred_flow = pred_flow[0].numpy().transpose((1, 2, 0))
        new_inputs = cv2.resize(pred_flow, old_size)

        np.save(os.path.join(video_of_path, cur_img_name + '.npy'), new_inputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ped2", help='dataset name')
    parser.add_argument("--train", action='store_true', default=False, help='train or test data')
    parser.add_argument("--root", type=str, default='../../../../dataset',
                        help='root dir that contains <dataset>/{training,testing}/frames')
    parser.add_argument("--gpu", type=int, default=0, help='cuda device index')
    parser.add_argument("--weights_path", type=str, default=None,
                        help='FlowNet2 checkpoint (default: ./checkpoints/FlowNet2_checkpoint.pth.tar)')

    args = parser.parse_args()
    root = os.path.abspath(args.root)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    weights_path = args.weights_path
    if weights_path is None:
        weights_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoints", "FlowNet2_checkpoint.pth.tar")

    with torch.no_grad():
        extracting_flows(dataset_name=args.dataset, root=root, train=args.train,
                         weights_path=weights_path, device=device)
