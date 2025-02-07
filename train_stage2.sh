python -m torch.distributed.launch --master_port=10234 --nproc_per_node=4 main_stage2.py --distributed --accum_iter 1 \
    --batch_size 64 \
    --epochs 12 \
    --mask_ratio 0.75 \
    --weight_decay 0.1 \
    --lr 2e-4 \
    --warmup_epochs 1 \
    --output_dir log/output_stage2 \
    --log_dir log/output_stage2 \
    --use_sensor_token \
    --use_same_patchemb \
    --num_workers 10 \
    --init_temp 0.07 \
    --sensor_token_for_all \
    --beta_start 0.0 \
    --beta_end 0.75 \
    --alpha_vt 1.0 \
    --TAG_times 2 \
    --use_video \
    --cross_alpha 0.1 \
    --mae_dir log/output_stage1/checkpoint-19.pth
    
    