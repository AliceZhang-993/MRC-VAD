"""Loading of the frame-level feature streams.

Each stream is stored as one .pt per frame under
<data_root>/<dataset>/{training,testing}/<features_name>/. Loading reads every
frame file and standardizes the test data with the training statistics. The
object-level motion branch is loaded directly in main.py.
"""

import time

import torch

from dataloader import build_dataloader


def get_dataset(data_root=None, dataset_name="shanghaitech",
                features_name="features", batchsize=2048):
    start_time = time.time()

    data_config = {
        "dataset_name": dataset_name,
        "data_root": data_root,
        "features_name": features_name,
        "batch_size": batchsize,
        "val_batch_size": batchsize,
    }

    train_loader, train_mean, train_std = build_dataloader(data_config, mode="train")
    data_train = torch.cat([batch["feature"] for batch in train_loader], dim=0)

    test_config = data_config.copy()
    test_config.update({"mean": train_mean, "std": train_std})
    test_loader, _, _ = build_dataloader(test_config, mode="test")
    data_test = torch.cat([batch["feature"] for batch in test_loader], dim=0)
    labels_test = torch.cat([batch["label"] for batch in test_loader], dim=0)

    id_to_type = {0: "normal", 1: "anomaly"}
    final_stats = {"mean": train_mean, "std": train_std}

    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
    print(f"Train data shape: {data_train.shape}, Test data shape: {data_test.shape}")

    return (
        data_train,
        torch.zeros(len(data_train), dtype=torch.uint8),
        data_test,
        labels_test,
        id_to_type,
        final_stats,
    )
