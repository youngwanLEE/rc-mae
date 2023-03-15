# --------------------------------------------------------------------------------
# Exploring the Role of Mean Teachers in Self-supervised Masked Auto-Encoders (ICLR'23)
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------------------
# Modified from MAE (https://github.com/facebookresearch/mae)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# --------------------------------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------------------------------

"""Components for pre-training script"""

import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    teacher_without_ddp: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    momentum_schedule,
    log_writer=None,
    args=None,
):

    """training step for each epoch"""

    student.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", misc.SmoothedValue(
            window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # ywlee
            s_loss, s_pred, s_mask, ids_shuffle = student(
                samples, mask_ratio=args.mask_ratio
            )
            with torch.no_grad():
                teacher.eval()
                t_pred = teacher(
                    samples,
                    mask_ratio=args.mask_ratio,
                    ids_shuffle=ids_shuffle)
            # On the same masked inputs
            # Including the masked / unmasked patches
            # pred's shape : [N, L, p*p*3]
            rec_cons_loss = (s_pred - t_pred.detach()) ** 2
            rec_cons_loss = rec_cons_loss.mean(dim=-1)  # [N, L], mean loss per patch
            rec_cons_loss = (rec_cons_loss * s_mask).sum() / s_mask.sum()  # mean loss on removed patches
            loss = s_loss + args.gamma * rec_cons_loss

        loss_value = loss.item()
        s_loss_value = s_loss.item()
        rec_cons_loss_value = rec_cons_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=student.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # EMA update for the teacher
        with torch.no_grad():
            ms = momentum_schedule[data_iter_step]  # momentum parameter
            for param_q, param_k in zip(
                student.module.parameters(), teacher_without_ddp.parameters()
            ):
                param_k.data.mul_(ms).add_((1 - ms) * param_q.detach().data)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(s_loss=s_loss_value)
        metric_logger.update(rec_cons_loss=rec_cons_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        s_loss_value_reduce = misc.all_reduce_mean(s_loss_value)
        rec_cons_loss_value_reduce = misc.all_reduce_mean(rec_cons_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("s_loss", s_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar(
                "rec_cons_loss", rec_cons_loss_value_reduce, epoch_1000x
            )
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
