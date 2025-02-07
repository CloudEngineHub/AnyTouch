import os
#os.environ['CUDA_VISIBLE_DEVICES']= '0, 1, 2, 3'
import torch
from transformers import AutoConfig
from transformers.models.vit.configuration_vit import ViTConfig
from config import parse_args
import random
import numpy as np
import torch.nn as nn
import sys
from dataloader.stage2_dataset import PretrainDataset_integrate, PretrainDataset_video_integrate
from dataloader.cross_dataset import PretrainDataset_cross, PretrainDataset_cross_video
from model.process_clip import convert_model_to_lora, print_trainable_parameters
from model.multi_model import CLIPModel
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm
import timm.optim.optim_factory as optim_factory
from stage2_engine import train_one_epoch

import argparse
import datetime
import json
import time
from pathlib import Path
import copy
import psutil

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
torch.cuda.device_count.cache_clear()

def load_from_mae_video(mae_ckpt, clip_ckpt, model):
    new_ckpt = {}
    for key,item in clip_ckpt.items():
        if "vision_model" in key and 'position_ids' not in key:
            new_ckpt[key] = item
        
        if "visual_projection" in key:
            new_ckpt[key] = item

        if ("text" in key or "logit" in key) and 'position_ids' not in key:
            new_ckpt[key] = item
    
    for key,item in mae_ckpt.items():
        new_ckpt['touch_mae_model.'+key] = item

    for k,v in model.named_parameters():
        if k not in new_ckpt.keys():
            print('new', k)
            new_ckpt[k] = v
    
    model.load_state_dict(new_ckpt, strict=True)

    return model

def load_model_from_clip_video(ckpt, model):
    new_ckpt = {}
    for key,item in ckpt.items():
        if "vision_model" in key and 'position_ids' not in key:
            new_ckpt[key] = item
            new_ckpt['touch_mae_model.'+key.replace("vision_model","touch_model")] = copy.deepcopy(item)

            if "embeddings.patch_embedding" in key:
                new_ckpt[key] = item
                new_item = copy.deepcopy(item)
                new_item = new_item.unsqueeze(1)
                new_item = new_item.repeat(1,3,1,1,1)
                new_ckpt['touch_mae_model.'+key.replace("vision_model.embeddings.","video_")] = new_item
            
            if "embeddings.position_embedding" in key:
                new_ckpt[key] = item
                new_ckpt['touch_mae_model.'+key.replace("vision_model.embeddings.","video_")] = copy.deepcopy(item)
        
        if "visual_projection" in key:
            new_ckpt[key] = item
            new_ckpt['touch_mae_model.'+key.replace("visual","touch")] = copy.deepcopy(item)

        if ("text" in key or "logit" in key) and 'position_ids' not in key:
            new_ckpt[key] = item
    
    for k,v in model.named_parameters():
        if k not in new_ckpt.keys():
            new_ckpt[k] = v
    
    model.load_state_dict(new_ckpt, strict=True)

    return model

def random_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args):

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)


    datasets_train_image = [PretrainDataset_integrate(args), PretrainDataset_cross()]
    if args.use_video:
        datasets_train_video = [PretrainDataset_video_integrate(args), PretrainDataset_cross_video()]
    else:
        datasets_train_video = []

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    samplers_train_image = []
    samplers_train_video = []

    for dataset in datasets_train_image:
        samplers_train_image.append(torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True))
    
    if args.use_video:
        for dataset in datasets_train_video:
            samplers_train_video.append(torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True))

    #print("Sampler_train = %s" % str(sampler_train_image))


    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loaders_train_image = []
    data_loaders_train_video = []

    for i in range(len(datasets_train_image)):
        if i == len(datasets_train_image) - 1:
            data_loaders_train_image.append(torch.utils.data.DataLoader(
                datasets_train_image[i], sampler=samplers_train_image[i],
                batch_size=args.batch_size // 2,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            ))
        else:
            data_loaders_train_image.append(torch.utils.data.DataLoader(
                datasets_train_image[i], sampler=samplers_train_image[i],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            ))


    for i in range(len(datasets_train_video)):
        if i == len(datasets_train_video) - 1:
            data_loaders_train_video.append(torch.utils.data.DataLoader(
                datasets_train_video[i], sampler=samplers_train_video[i],
                batch_size=args.batch_size // 2,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            ))
        else:
            data_loaders_train_video.append(torch.utils.data.DataLoader(
                datasets_train_video[i], sampler=samplers_train_video[i],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            ))

    config = AutoConfig.from_pretrained('CLIP-ViT-L-14-DataComp.XL-s13B-b90K/config.json')
    print(args)

    decoder_config = ViTConfig()
    decoder_config.encoder_stride = 14
    decoder_config.hidden_size = 512
    decoder_config.intermediate_size = 2048
    decoder_config.num_attention_heads = 16
    decoder_config.num_hidden_layers = 8
    decoder_config.patch_size = 14


    model = CLIPModel(args, config, decoder_config, 1, False, 1)
    model.touch_mae_model.initialize_decoder()
    ckpt = torch.load('CLIP-ViT-L-14-DataComp.XL-s13B-b90K/pytorch_model.bin', map_location='cpu')
    if args.mae_dir is not None:
        mae_ckpt = torch.load(args.mae_dir, map_location='cpu')['model']
        model = load_from_mae_video(mae_ckpt,ckpt, model)
    else:
        model = load_model_from_clip_video(ckpt, model)

    if not args.no_text:
        convert_model_to_lora(args, model)

    if args.init_temp is not None:
        with torch.no_grad():
            model.logit_scale.fill_(np.log(1 / float(args.init_temp)))
        print('logit scale:', model.logit_scale)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, betas = (0.9, 0.99))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for i in range(len(data_loaders_train_image)):
                data_loaders_train_image[i].sampler.set_epoch(epoch)
            for i in range(len(data_loaders_train_video)):   
                data_loaders_train_video[i].sampler.set_epoch(epoch)

        # print(data_loaders_train_image)
        # print(data_loaders_train_video)
        # exit(0)
        train_stats_image, train_stats_video = train_one_epoch(
            model, data_loaders_train_image, data_loaders_train_video,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        # if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
        misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=epoch)

        log_stats_image = {**{f'train_{k}': v for k, v in train_stats_image.items()},
                        'epoch': epoch,}
        if args.use_video:
            log_stats_video = {**{f'train_{k}': v for k, v in train_stats_video.items()},
                            'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats_image) + "\n")
                if args.use_video:
                    f.write(json.dumps(log_stats_video) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    

if __name__ == "__main__":
    args = parse_args()
    args = args.parse_args()
    main(args)