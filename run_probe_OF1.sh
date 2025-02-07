CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=6234 --nproc_per_node=2 main_probe.py --distributed --accum_iter 1 \
    --batch_size 64 \
    --epochs 50 \
    --weight_decay 0.01 \
    --lr 1e-3 \
    --warmup_epochs 0 \
    --output_dir log/output_probe_OF1 \
    --log_dir log/output_probe_OF1 \
    --pooling global \
    --dataset obj1 \
    --use_sensor_token \
    --use_same_patchemb \
    --load_from_align \
    --load_path log/checkpoint.pth \
    --use_universal \
    