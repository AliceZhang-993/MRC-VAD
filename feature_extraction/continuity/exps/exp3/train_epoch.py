import os
import time

import torch

from data.dataloader import ContinuityDataset
from models.continuity_model import ContinuityModel
from models.losses import BCEAverageLoss, MultiClassCEAverageLoss, TripletContrastiveLoss
from utils.compatibility import pt_compat
from utils.distributed import dist_manager
from utils.logger import TrainingLogger


def build_dataloader(config, mode="train"):
    split = "training" if mode == "train" else "testing"
    video_dir = config["data_root"] + config["dataset_name"] + f"/{split}/frames/"
    assert os.path.exists(video_dir), f"dataset path not exist: {video_dir}"
    dataset = ContinuityDataset(video_dir, config)
    sampler = dist_manager.get_sampler(dataset, shuffle=(mode == "train"))
    shuffle = (mode == "train") if sampler is None else False
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=(mode == "train")
    )


class TrainingEngine:
    def __init__(self, config, checkpoint_path=None):
        self.config = config
        self.use_amp = self.config["train"]["mixed_precision"] == "fp16"
        self.val_interval = self.config["train"]["val_interval"]
        self.logger = TrainingLogger(self.config, "train")
        self.step_count = 0
        self.start_epoch = 0

        self.model = ContinuityModel(self.config)
        self.model = pt_compat.move_data_to_device(self.model, dist_manager.device)
        self.raw_model = self.model

        self.raw_model.configure_optimizers()

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        self.train_loader = build_dataloader(self.config["data"], mode="train")

        cl_cfg = self.config.get("continuity_learning", {})
        self.loss_weights = cl_cfg.get("loss_weights", {"cj": 1.0, "dl": 1.0, "mfe": 1.0})
        self.loss_cj = BCEAverageLoss()
        self.loss_dl = MultiClassCEAverageLoss()
        self.loss_mfe = TripletContrastiveLoss(
            omega=cl_cfg.get("contrastive", {}).get("omega", 0.5),
            temperature=cl_cfg.get("contrastive", {}).get("tau", 0.1),
        )

    def _load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
        checkpoint = pt_compat.load_checkpoint(checkpoint_path, model=self.model)
        self.start_epoch = checkpoint["epoch"] + 1
        self.step_count = checkpoint["step"]
        if 'learning_rate' in checkpoint:
            self.raw_model.learning_rate = checkpoint['learning_rate']
        if 'optimizer_state' in checkpoint:
            try:
                self.raw_model.optimizer.load_state_dict(checkpoint['optimizer_state'])
            except ValueError as e:
                print(f"[Warning] optimizer state not match: {str(e)}")
        if dist_manager.is_master:
            print(f"Resuming training from checkpoint, epoch {checkpoint['epoch']}, "
                  f"step {self.step_count}")

    def training_loop(self):
        try:
            epoch_start_time = time.time()
            optimizer = self.raw_model.configure_optimizers()
            if dist_manager.is_master:
                print(f"current_lr:{optimizer.param_groups[0]['lr']}")

            dataset_name = self.config["data"]["dataset_name"]
            for epoch in range(self.start_epoch, self.config["train"]["max_epochs"]):
                self.raw_model.apply_unfreeze_policy(epoch, dataset_name)
                self.model.train()

                total_batches = len(self.train_loader)
                print_interval = max(1, total_batches // 10)

                total_epochs = self.config["train"]["max_epochs"] - self.start_epoch
                elapsed_epochs = epoch - self.start_epoch
                avg_epoch_time = (time.time() - epoch_start_time) / (elapsed_epochs + 1e-7)
                remaining_epochs = total_epochs - elapsed_epochs
                remaining_time = avg_epoch_time * remaining_epochs
                if dist_manager.is_master:
                    print(f"Epoch {epoch} | Avg Time per Epoch: {avg_epoch_time:.2f}s | "
                          f"Remaining Epochs: {remaining_epochs} | "
                          f"ETA: {remaining_time // 3600:.0f}h {remaining_time % 3600 // 60:.0f}m")

                for batch_idx, batch in enumerate(self.train_loader, 1):
                    batch_start_time = time.time()
                    loss_dict = self.train_step(optimizer, batch)
                    if dist_manager.is_master:
                        self.logger.log_train(loss_dict["total_loss"], self.step_count)
                        for loss_name, loss_val in loss_dict.items():
                            if loss_name != "total_loss":
                                self.logger.log_scalar(f"Train/{loss_name}", loss_val, self.step_count)

                    batch_time = time.time() - batch_start_time
                    avg_batch_time = (time.time() - epoch_start_time) / batch_idx
                    remaining_batches = len(self.train_loader) - batch_idx
                    remaining_time_b = avg_batch_time * remaining_batches
                    if dist_manager.is_master and (batch_idx % print_interval == 0):
                        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}][train] Epoch {epoch} | "
                              f"Batch {batch_idx}/{total_batches} | Batch Time: {batch_time:.2f}s | "
                              f"ETA: {remaining_time_b // 60:.0f}m {remaining_time_b % 60:.0f}s | "
                              f"Step {self.step_count} | {loss_str}")

        except KeyboardInterrupt:
            print("-----------train KeyboardInterrupted by user")
            del self.model
            torch.cuda.empty_cache()
            self.logger.close()
            exit(0)

    def train_step(self, optimizer, batch):
        grad_accum_steps = self.config["train"].get("grad_accum_steps", 1)
        scaler = pt_compat.amp.GradScaler(enabled=self.use_amp)

        (
            vc, vd, im, vco,
            vc_mask, vd_mask, im_mask, vco_mask,
            cj_label_pos, cj_label_neg, dl_target,
            seqs, starts, missing_idxs
        ) = batch

        vc = pt_compat.move_data_to_device(vc, dist_manager.device)
        vd = pt_compat.move_data_to_device(vd, dist_manager.device)
        im = pt_compat.move_data_to_device(im, dist_manager.device)
        vco = pt_compat.move_data_to_device(vco, dist_manager.device)
        vc_mask = pt_compat.move_data_to_device(vc_mask, dist_manager.device)
        vd_mask = pt_compat.move_data_to_device(vd_mask, dist_manager.device)
        im_mask = pt_compat.move_data_to_device(im_mask, dist_manager.device)
        vco_mask = pt_compat.move_data_to_device(vco_mask, dist_manager.device)
        dl_target = pt_compat.move_data_to_device(dl_target, dist_manager.device)

        with pt_compat.autocast(enabled=self.use_amp):
            feat_vc = self.raw_model.forward_backbone(vc)
            feat_vd = self.raw_model.forward_backbone(vd)
            feat_im = self.raw_model.forward_backbone(im)
            feat_vco = self.raw_model.forward_backbone(vco)

            outputs = self.raw_model.forward_heads(
                feat_vc, feat_vd, feat_im, feat_vco,
                vc_mask, vd_mask, im_mask, vco_mask,
            )

            total_loss = 0.0
            loss_dict = {}

            if "cj_logits_pos" in outputs and "cj_logits_neg" in outputs:
                b_pos = outputs["cj_logits_pos"].size(0)
                b_neg = outputs["cj_logits_neg"].size(0)
                target_pos = torch.ones(b_pos, dtype=torch.long,
                                        device=outputs["cj_logits_pos"].device)
                target_neg = torch.zeros(b_neg, dtype=torch.long,
                                         device=outputs["cj_logits_neg"].device)
                cj_loss = 0.5 * (self.loss_cj(outputs["cj_logits_pos"], target_pos)
                                 + self.loss_cj(outputs["cj_logits_neg"], target_neg))
                loss_dict["cj_loss"] = float(cj_loss.detach().item())
                total_loss = total_loss + self.loss_weights.get("cj", 1.0) * cj_loss

            if "dl_logits" in outputs:
                dl_targets = dl_target.view(-1).to(outputs["dl_logits"].device)
                dl_loss = self.loss_dl(outputs["dl_logits"], dl_targets)
                loss_dict["dl_loss"] = float(dl_loss.detach().item())
                total_loss = total_loss + self.loss_weights.get("dl", 1.0) * dl_loss

            if {"mfe_vd", "mfe_im", "mfe_vc"}.issubset(outputs.keys()):
                mfe_loss = self.loss_mfe(
                    outputs["mfe_vd"], outputs["mfe_im"], outputs["mfe_vc"],
                    outputs["mfe_vc"], outputs.get("mfe_vco", outputs["mfe_vc"])
                )
                loss_dict["mfe_loss"] = float(mfe_loss.detach().item())
                total_loss = total_loss + self.loss_weights.get("mfe", 1.0) * mfe_loss

            loss_dict["total_loss"] = float(total_loss.detach().item())

        if self.use_amp:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        if (self.step_count + 1) % max(1, grad_accum_steps) == 0:
            if self.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        self.step_count += 1
        return loss_dict
