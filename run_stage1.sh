#!/bin/bash

### 将本次作业计费到导师课题组，tutor_project 改为导师创建的课题组名
#SBATCH --comment=GSAI_GeWuLab

### 给您这个作业起个名字，方便识别不同的作业
#SBATCH --job-name=testgpu_full

### 指定该作业需要多少个节点
### 注意！没有使用多机并行（MPI/NCCL 等），下面参数写 1！不要多写，多写了也不会加速程序！
#SBATCH --nodes=1

### 指定该作业需要多少个 CPU 核心
### 注意！一般根据队列的 CPU 核心数填写，比如 cpu 队列 64 核，这里申请 64 核，并在您的程序中尽量使用多线程充分利用 64 核资源！
#SBATCH --ntasks=32

### 指定该作业在哪个队列上执行
#SBATCH --partition=gpu-a800

#SBATCH --gres=gpu:4

#SBATCH --output=mae_video_all+multi_new_samepatchemb_kecheng_0.75_new_12.9.out

### export PATH=/home/u2023100841/anaconda3/bin:$PATH

source ~/.bashrc
conda activate touch

python -m torch.distributed.launch --master_port=22234 --nproc_per_node=4 main_video.py --distributed --accum_iter 1 \
    --batch_size 64 \
    --epochs 20 \
    --mask_ratio 0.75 \
    --weight_decay 0.1 \
    --lr 2e-4 \
    --warmup_epochs 1 \
    --output_dir output_mae_video_all+multi_new_samepatchemb_kecheng_0.75_new_12.9 \
    --log_dir output_mae_video_all+multi_new_samepatchemb_kecheng_0.75_new_12.9 \
    --use_sensor_token \
    --use_video \
    --use_same_patchemb \
    --num_workers 12 \
    --sensor_token_for_all \
    --beta_start 0.0 \
    --beta_end 0.75 \
    # --resume /home/ruoxuan_feng/tactile/output_mae_video_all_samepatchemb/checkpoint-1.pth
    