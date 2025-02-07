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
import time


def train_one_epoch(model: torch.nn.Module,
                    data_loaders_image, data_loaders_video,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger_image = misc.MetricLogger(delimiter="  ")
    metric_logger_image.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header_image = 'Epoch (image): [{}]'.format(epoch)

    if args.use_video:
        metric_logger_video = misc.MetricLogger(delimiter="  ")
        metric_logger_video.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header_video = 'Epoch (video): [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    len_image = [len(loader) for loader in data_loaders_image]
    len_video = [len(loader) for loader in data_loaders_video]
    sum_len_image = sum(len_image)
    if args.use_video:
        sum_len_video = sum(len_video)
    else:
        sum_len_video = 0
    # print(len_image, len_video)

    # exit(0)
    step_cross = 0

    # data_choice = torch.zeros(sum_len_image + sum_len_video)
    if args.use_video:
        dataset_index = torch.randperm(sum_len_image + sum_len_video)

        min_max_values = []
        min_value = 0
        max_value = 0
        for index in range(len(len_image)):
            max_value += len_image[index]
            min_max_values.append((min_value, max_value))
            min_value = max_value

        for index in range(len(len_video)):
            max_value += len_video[index]
            min_max_values.append((min_value, max_value))
            min_value = max_value
        
        for i in range(len(min_max_values)):
            start, end = min_max_values[i]
            mask = (dataset_index >= start) & (dataset_index < end)
            dataset_index[mask] = i

        data_choice = dataset_index.int()
    
    else:
        data_choice = torch.zeros(sum_len_image).int()

    iter_dataloader_image = metric_logger_image.log_every(data_loaders_image[0], print_freq, header_image)
    if args.use_video:
        iter_dataloader_video = metric_logger_video.log_every(data_loaders_video[0], print_freq, header_video)
    # print(iter_dataloader_image, data_loaders_image[0], data_loaders_image[1])
    # exit(0)

    iter_cross_image = iter(data_loaders_image[1])
    iter_cross_video = iter(data_loaders_video[1])

    for data_iter_step, dataset_id in enumerate(data_choice):
        #print('start', time.time())

        # we use a per iteration (instead of per epoch) lr scheduler
        #dataset_id = (data_iter_step % 2) * 2 + 1
        do_cross=False

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / (sum_len_image + sum_len_video) + epoch, args)


        if args.use_video:
            if dataset_id == 0:
                touch, image, text, mask, sensors, vision_flag, text_flag = next(iter_dataloader_image)
                data_type = 0

            elif dataset_id == len(data_loaders_image):
                touch, image, text, mask, sensors, vision_flag, text_flag = next(iter_dataloader_video)
                data_type = 1

            elif dataset_id < len(data_loaders_image):
                touch, sensors, positive, pos_sensors, negative, neg_sensors = next(iter_cross_image)
                data_type = 0
                do_cross = True

            elif dataset_id > len(data_loaders_image):
                touch, sensors, positive, pos_sensors, negative, neg_sensors = next(iter_cross_video)
                data_type = 1
                do_cross = True
            
        else:
            touch, image, text, mask, sensors, vision_flag, text_flag = next(iter_dataloader_image)
            data_type = 0
            step_image += 1


        # elif dataset_id < len(data_loaders_image):
        #     touch, image, text, mask, sensors = next(iter(data_loaders_image[dataset_id]))
        #     data_type = 0
        # else:
        #     touch, image, text, mask, sensors = next(iter(data_loaders_video[dataset_id-len(data_loaders_image)]))
        #     data_type = 1

        # print(touch.shape, image.shape, text.shape, mask.shape, sensors.shape, dataset_id)
        #torch.cuda.synchronize()
        if do_cross:
            touch = touch.to(device, non_blocking=True)
            sensors = sensors.to(device, non_blocking=True).int()
            positive = positive.to(device, non_blocking=True)
            pos_sensors = pos_sensors.to(device, non_blocking=True).int()
            negative = negative.to(device, non_blocking=True)
            neg_sensors = neg_sensors.to(device, non_blocking=True).int()

        else:
            touch = touch.to(device, non_blocking=True)
            image = image.to(device, non_blocking=True)
            text = text.to(device, non_blocking=True).int()
            mask = mask.to(device, non_blocking=True).int()
            sensors = sensors.to(device, non_blocking=True).int()
            vision_flag = vision_flag.to(device, non_blocking=True).flatten().int()
            text_flag = text_flag.to(device, non_blocking=True).flatten().int()

        #torch.cuda.synchronize()

        # print(touch.shape, image.shape, text.shape, mask.shape, sensors.shape)

        if args.sensor_token_for_all:
            now_epoch_point = data_iter_step / (sum_len_image + sum_len_video) + epoch
            sensor_p = args.beta_start + (args.beta_end - args.beta_start) * (now_epoch_point / (args.epochs * 1.0))

            bernoulli_mask = torch.bernoulli(torch.full(sensors.shape, sensor_p)).to(device, non_blocking=True)
            new_sensors = sensors * (1 - bernoulli_mask) - bernoulli_mask
            new_sensors = new_sensors.int()

            if do_cross:
                pos_sensors = pos_sensors * (1 - bernoulli_mask) - bernoulli_mask
                pos_sensors = pos_sensors.int()

                neg_sensors = neg_sensors * (1 - bernoulli_mask) - bernoulli_mask
                neg_sensors = neg_sensors.int()

            if data_iter_step % (print_freq*20) == 0:
                print('sensor_p:',sensor_p)
            # print(sensor_p, sensors)
        else:
            new_sensors = sensors

        # print(touch.shape, image.shape if image is not None else None, text.shape if text is not None else None, mask.shape if mask is not None else None, sensors.shape, dataset_id)
        #print('load gpu', time.time())

        with torch.cuda.amp.autocast():
            # with torch.autograd.profiler.profile() as prof:
            if do_cross:
                matching_loss = model(touch_input = touch, sensor_type = new_sensors, data_type = data_type, positive_sample = positive, negative_sample = negative, pos_sensors=pos_sensors, neg_sensors=neg_sensors)
                loss = matching_loss
                if step_cross % 10 == 0:
                    print('Matching Loss', loss.item())

                step_cross += 1

            else:
                aligh_loss, mae_loss = model(text, mask, image, touch, sensor_type = new_sensors, data_type = data_type, target_sensor_type = sensors, vision_flag = vision_flag, text_flag = text_flag)
                #loss = torch.ones(1).to(device, non_blocking=True)
                if aligh_loss is not None:
                    if args.no_mae:
                        loss = aligh_loss
                    else:
                        loss = aligh_loss + mae_loss
                else:
                    loss = mae_loss
                if data_iter_step % (print_freq) == 0:
                    print('Align Loss:',aligh_loss.item() if aligh_loss is not None else None, 'Mae Loss:',mae_loss.item())

        # print(prof)
        #torch.cuda.synchronize()
        #print('forward', time.time())
        loss_value = loss.item()

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        #print('backward', time.time())
        torch.cuda.synchronize()
        #print('synchronize', time.time())
        if data_type == 0:
            metric_logger_image.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger_image.update(lr=lr)
        
        elif data_type == 1:
            metric_logger_video.update(loss=loss_value)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger_video.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / (sum_len_image + sum_len_video) + epoch) * 1000)
            if data_type == 0 and do_cross == False:
                log_writer.add_scalar('train_loss_image', loss_value_reduce, epoch_1000x)
            elif data_type == 1 and do_cross == False:
                log_writer.add_scalar('train_loss_video', loss_value_reduce, epoch_1000x)
            elif data_type == 0 and do_cross == True:
                log_writer.add_scalar('train_loss_image_cross', loss_value_reduce, epoch_1000x)
            elif data_type == 1 and do_cross == True:
                log_writer.add_scalar('train_loss_video_cross', loss_value_reduce, epoch_1000x)   

            log_writer.add_scalar('train_loss_full', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger_image.synchronize_between_processes()
    if args.use_video:
        metric_logger_video.synchronize_between_processes()
    print("Averaged stats (image):", metric_logger_image)
    if args.use_video:
        print("Averaged stats (video):", metric_logger_video)

    if args.use_video:
        return {k: meter.global_avg for k, meter in metric_logger_image.meters.items()}, {k: meter.global_avg for k, meter in metric_logger_video.meters.items()}
    
    else:
        return {k: meter.global_avg for k, meter in metric_logger_image.meters.items()}, 0
    