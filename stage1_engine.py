# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import numpy as np

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader_image, data_loader_video, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger_image = misc.MetricLogger(delimiter="  ")
    metric_logger_image.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header_image = 'Epoch (image): [{}]'.format(epoch)

    metric_logger_video = misc.MetricLogger(delimiter="  ")
    metric_logger_video.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header_video = 'Epoch (video): [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    len_image = len(data_loader_image)
    len_video = len(data_loader_video)
    print(len_image, len_video)

    data_choice = torch.ones(len_image + len_video)
    dataset_index = torch.randperm(len_image + len_video)
    image_index = [index for index, value in enumerate(dataset_index) if value < len_image]
    data_choice[image_index] = 0
    data_choice = data_choice.int()

    iter_dataloader_image = metric_logger_image.log_every(data_loader_image, print_freq, header_image)
    iter_dataloader_video = metric_logger_video.log_every(data_loader_video, print_freq, header_video)

    for data_iter_step, data_type in enumerate(data_choice):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / (len_image + len_video) + epoch, args)
        
        if data_type == 0:
            samples, sensors = next(iter_dataloader_image)
        else:
            samples, sensors = next(iter_dataloader_video)

        samples = samples.to(device, non_blocking=True)
        sensors = sensors.to(device, non_blocking=True).int()

        if args.sensor_token_for_all:
            now_epoch_point = data_iter_step / (len_image + len_video) + epoch
            sensor_p = args.beta_start + (args.beta_end - args.beta_start) * (now_epoch_point / (args.epochs * 1.0))

            bernoulli_mask = torch.bernoulli(torch.full(sensors.shape, sensor_p)).to(device, non_blocking=True)
            sensors = sensors * (1 - bernoulli_mask) - bernoulli_mask
            sensors = sensors.int()

            if data_iter_step % (print_freq*2) == 0:
                print('sensor_p:',sensor_p)
            # print(sensor_p, sensors)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, sensor_type = sensors, data_type = data_type)
            #loss = torch.ones(1).to(device, non_blocking=True)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        if data_type == 0:
            metric_logger_image.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger_image.update(lr=lr)
        
        else:
            metric_logger_video.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger_video.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / (len_image + len_video) + epoch) * 1000)
            if data_type == 0:
                log_writer.add_scalar('train_loss_image', loss_value_reduce, epoch_1000x)
            else:
                log_writer.add_scalar('train_loss_video', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_full', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger_image.synchronize_between_processes()
    metric_logger_video.synchronize_between_processes()
    print("Averaged stats (image):", metric_logger_image)
    print("Averaged stats (video):", metric_logger_video)
    return {k: meter.global_avg for k, meter in metric_logger_image.meters.items()}, {k: meter.global_avg for k, meter in metric_logger_video.meters.items()}
    