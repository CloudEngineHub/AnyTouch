import os
# os.environ['CUDA_VISIBLE_DEVICES']= '0, 1, 2, 3'
import torch
from transformers import AutoConfig
from transformers.models.vit.configuration_vit import ViTConfig
from model.mae_model import TactileMAE, TactileVideoMAE
from config import parse_args
import random
import numpy as np
import torch.nn as nn
import sys
from dataloader.stage1_dataset import PretrainDataset_Contact, PretrainDataset_Contact_video
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm
import timm.optim.optim_factory as optim_factory
from stage1_engine import train_one_epoch

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

def load_model_from_clip(ckpt, model):
    new_ckpt = {}
    for key,item in ckpt.items():
        if "vision_model" in key and 'position_ids' not in key:
            #new_ckpt[key] = item
            new_ckpt[key.replace("vision_model","touch_model")] = copy.deepcopy(item)
        
        if "visual_projection" in key:
            #new_ckpt[key] = item
            new_ckpt[key.replace("visual","touch")] = copy.deepcopy(item)
    
    for k,v in model.named_parameters():
        if k not in new_ckpt.keys():
            new_ckpt[k] = v
    
    model.load_state_dict(new_ckpt, strict=True)

    return model

def load_model_from_clip_video(ckpt, model):
    new_ckpt = {}
    for key,item in ckpt.items():
        if "vision_model" in key and 'position_ids' not in key:
            #new_ckpt[key] = item
            new_ckpt[key.replace("vision_model","touch_model")] = copy.deepcopy(item)

            if "embeddings.patch_embedding" in key:
                new_item = copy.deepcopy(item)
                new_item = new_item.unsqueeze(1)
                new_item = new_item.repeat(1,3,1,1,1)
                new_ckpt[key.replace("vision_model.embeddings.","video_")] = new_item
            
            if "embeddings.position_embedding" in key:
                new_ckpt[key.replace("vision_model.embeddings.","video_")] = copy.deepcopy(item)
        
        if "visual_projection" in key:
            #new_ckpt[key] = item
            new_ckpt[key.replace("visual","touch")] = copy.deepcopy(item)
        
    
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


    dataset_train_image = PretrainDataset_Contact()
    dataset_train_video = None
    if args.use_video:
        dataset_train_video = PretrainDataset_Contact_video()
    # print(dataset_train)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train_image = torch.utils.data.DistributedSampler(
        dataset_train_image, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    if args.use_video:
        sampler_train_video = torch.utils.data.DistributedSampler(
            dataset_train_video, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    print("Sampler_train = %s" % str(sampler_train_image))


    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train_image = torch.utils.data.DataLoader(
        dataset_train_image, sampler=sampler_train_image,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if args.use_video:
        data_loader_train_video = torch.utils.data.DataLoader(
            dataset_train_video, sampler=sampler_train_video,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        data_loader_train_video = None

    config = AutoConfig.from_pretrained('CLIP-ViT-L-14-DataComp.XL-s13B-b90K/config.json')
    print(args)

    decoder_config = ViTConfig()
    decoder_config.encoder_stride = 14
    decoder_config.hidden_size = 512
    decoder_config.intermediate_size = 2048
    decoder_config.num_attention_heads = 16
    decoder_config.num_hidden_layers = 8
    decoder_config.patch_size = 14


    model = TactileVideoMAE(args, config, decoder_config, 1, False, 1)
    model.initialize_decoder()
    ckpt = torch.load('CLIP-ViT-L-14-DataComp.XL-s13B-b90K/pytorch_model.bin', map_location='cpu')
    model = load_model_from_clip_video(ckpt, model)


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
            data_loader_train_image.sampler.set_epoch(epoch)
            if args.use_video:
                data_loader_train_video.sampler.set_epoch(epoch)
        train_stats_image, train_stats_video = train_one_epoch(
            model, data_loader_train_image, data_loader_train_video,
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