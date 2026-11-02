import torch
import numpy as np
import argparse
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tqdm import tqdm
from scipy.ndimage import uniform_filter
from video_dataset import VideoDatasetWithFlows

def extract_motion(flow, magnitude, orientation, orientations=8, motion_threshold=0.):
    orientation *= (180 / np.pi)

    cy, cx = flow.shape[:2]

    orientation_histogram = np.zeros(orientations)
    subsample = np.index_exp[cy // 2:cy:cy, cx // 2:cx:cx]
    for i in range(orientations):

        temp_ori = np.where(orientation < 360 / orientations * (i + 1),
                            orientation, -1)

        temp_ori = np.where(orientation >= 360 / orientations * i,
                            temp_ori, -1)

        cond2 = (temp_ori > -1) * (magnitude >= motion_threshold)
        temp_mag = np.where(cond2, magnitude, 0)
        temp_filt = uniform_filter(temp_mag, size=(cy, cx))

        orientation_histogram[i] = temp_filt[subsample][0, 0]

    return orientation_histogram


def extract(args, root, out_root):
    all_bboxes_train = np.load(os.path.join(root, args.dataset, '%s_bboxes_train.npy' % args.dataset),
                               allow_pickle=True)
    all_bboxes_test = np.load(os.path.join(root, args.dataset, '%s_bboxes_test.npy' % args.dataset),
                              allow_pickle=True)

    if args.dataset == 'shanghaitech':
        all_bboxes_train_classes = np.load(os.path.join(root, args.dataset, '%s_bboxes_train_classes.npy' % args.dataset),
                                   allow_pickle=True)
        all_bboxes_test_classes = np.load(os.path.join(root, args.dataset, '%s_bboxes_test_classes.npy' % args.dataset),
                                  allow_pickle=True)

    train_dataset = VideoDatasetWithFlows(dataset_name=args.dataset, root=root,
                                          train=True, sequence_length=0, all_bboxes=all_bboxes_train, normalize=True)
    test_dataset = VideoDatasetWithFlows(dataset_name=args.dataset, root=root,
                                         train=False, sequence_length=0, all_bboxes=all_bboxes_test, normalize=True)

    bins_list = args.bins if args.bins else [8]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for bins in bins_list:
            print(f'Extracting motion features with bins = {bins}')

            train_motion = []
            for idx in tqdm(range(len(train_dataset)), total=len(train_dataset),
                            desc=f"Train bins={bins}"):
                batch, batch_flows, _ = train_dataset.__getitem__(idx)
                batch = batch[:, 0].to(device)
                batch_flows = batch_flows[:, 0].numpy()
                train_sample_motions = []

                frame_bbox = train_dataset.all_bboxes[idx]

                if len(frame_bbox) > 0 and args.dataset == 'shanghaitech':
                    frame_classes = all_bboxes_train_classes[idx]
                    length_y = (frame_bbox[:, 3] - frame_bbox[:, 1])
                    non_person_indices = np.where(frame_classes != 0)[0]
                    length_y[non_person_indices] = 1
                else:
                    length_y = np.ones(1)

                for i in range(batch_flows.shape[0]):
                    img_flow = np.transpose(batch_flows[i], [1, 2, 0])
                    # convert from cartesian to polar
                    _, ang = cv2.cartToPolar(img_flow[..., 0], img_flow[..., 1])
                    mag = np.sqrt(img_flow[..., 0] ** 2) + np.sqrt(img_flow[..., 1] ** 2)  # L1 Magnitudes
                    mag = mag / length_y[i] if args.dataset == 'shanghaitech' else mag  # normalization
                    motion_cur = extract_motion(img_flow, mag, ang, orientations=bins)
                    train_sample_motions.append(motion_cur[None])

                train_sample_motions = np.concatenate(train_sample_motions, axis=0)
                train_motion.append(train_sample_motions)

            train_motion = np.array(train_motion, dtype=object)
            train_out = os.path.join(out_root, args.dataset, 'train')
            os.makedirs(train_out, exist_ok=True)
            np.save(os.path.join(train_out, f'motion_bins{bins}.npy'), train_motion)

            test_motion = []
            for idx in tqdm(range(len(test_dataset)), total=len(test_dataset),
                            desc=f"Test bins={bins}"):
                batch, batch_flows, _ = test_dataset.__getitem__(idx)
                batch = batch[:, 0].to(device)
                batch_flows = batch_flows[:, 0].numpy()
                test_sample_motions = []

                frame_bbox = test_dataset.all_bboxes[idx]

                if len(frame_bbox) > 0 and args.dataset == 'shanghaitech':
                    frame_classes = all_bboxes_test_classes[idx]
                    length_y = (frame_bbox[:, 3] - frame_bbox[:, 1])
                    non_person_indices = np.where(frame_classes != 0)[0]
                    length_y[non_person_indices] = 1
                else:
                    length_y = np.ones(1)

                for i in range(batch_flows.shape[0]):
                    img_flow = np.transpose(batch_flows[i], [1, 2, 0])
                    _, ang = cv2.cartToPolar(img_flow[..., 0], img_flow[..., 1])
                    mag = np.sqrt(img_flow[..., 0] ** 2) + np.sqrt(img_flow[..., 1] ** 2)
                    mag = mag / length_y[i] if args.dataset == 'shanghaitech' else mag
                    motion_cur = extract_motion(img_flow, mag, ang, orientations=bins)
                    test_sample_motions.append(motion_cur[None])

                test_sample_motions = np.concatenate(test_sample_motions, axis=0)
                test_motion.append(test_sample_motions)

            test_motion = np.array(test_motion, dtype=object)
            test_out = os.path.join(out_root, args.dataset, 'test')
            os.makedirs(test_out, exist_ok=True)
            np.save(os.path.join(test_out, f'motion_bins{bins}.npy'), test_motion)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ped2", help='dataset name')
    parser.add_argument("--root", type=str, default='../../../../dataset',
                        help='root dir that contains <dataset>/{training,testing}/frames and '
                             '<dataset>/<ds>_bboxes_{train,test}.npy')
    parser.add_argument("--out_root", type=str, default='extracted_features',
                        help='dir where <out_root>/<dataset>/{train,test}/motion_bins*.npy '
                             'will be written')
    parser.add_argument(
        "--bins",
        type=int,
        nargs='*',
        default=[8],
        help='numbers of orientation bins for motion, e.g. 4 8 12 16'
    )
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    out_root = os.path.abspath(args.out_root)
    extract(args, root, out_root)