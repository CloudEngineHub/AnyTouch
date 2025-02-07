CUDA_VISIBLE_DEVICES=2,3 python -u -m torch.distributed.launch --master_port=32234 --nproc_per_node=2 main_probe.py --distributed --accum_iter 1 \
    --batch_size 128 \
    --epochs 50 \
    --weight_decay 0.05 \
    --lr 1e-3 \
    --warmup_epochs 0 \
    --output_dir log/output_probe_Feel \
    --log_dir log/output_probe_Feel \
    --pooling cls \
    --dataset feel \
    --split 3 \
    --use_same_patchemb \
    --load_from_align \
    --load_path log/checkpoint.pth \
    --use_sensor_token \
    