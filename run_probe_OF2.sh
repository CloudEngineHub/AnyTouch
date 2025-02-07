CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port=10234 --nproc_per_node=2 main_probe.py --distributed --accum_iter 1 \
    --batch_size 128 \
    --epochs 50 \
    --weight_decay 0.01 \
    --beta_2 0.99 \
    --lr 1e-3 \
    --warmup_epochs 0 \
    --output_dir log/output_probe_OF2 \
    --log_dir log/output_probe_OF2 \
    --pooling cls \
    --dataset obj2 \
    --use_sensor_token \
    --use_same_patchemb \
    --load_from_align \
    --load_path log/checkpoint.pth \
    --use_universal \
    