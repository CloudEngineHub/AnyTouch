CUDA_VISIBLE_DEVICES=0,1 python -u -m torch.distributed.launch --master_port=10234 --nproc_per_node=2 main_probe.py --distributed --accum_iter 1 \
    --batch_size 64 \
    --epochs 50 \
    --weight_decay 0.05 \
    --lr 2e-4 \
    --warmup_epochs 0 \
    --output_dir log/output_probe_TAG \
    --log_dir log/output_probe_TAG \
    --pooling cls \
    --dataset rough \
    --use_same_patchemb \
    --load_path log/checkpoint.pth \
    --use_sensor_token \
    --load_from_align \
    