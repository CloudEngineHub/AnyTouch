python -m torch.distributed.launch --master_port=22234 --nproc_per_node=4 main_video.py --distributed --accum_iter 1 \
    --batch_size 64 \
    --epochs 20 \
    --mask_ratio 0.75 \
    --weight_decay 0.1 \
    --lr 2e-4 \
    --warmup_epochs 1 \
    --output_dir log/output_stage1 \
    --log_dir log/output_stage1 \
    --use_sensor_token \
    --use_video \
    --use_same_patchemb \
    --num_workers 12 \
    --sensor_token_for_all \
    --beta_start 0.0 \
    --beta_end 0.75 \
    