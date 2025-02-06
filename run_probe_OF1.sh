CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=6234 --nproc_per_node=2 main_probe.py --distributed --accum_iter 1 \
    --batch_size 64 \
    --epochs 50 \
    --weight_decay 0.01 \
    --lr 1e-3 \
    --warmup_epochs 0 \
    --output_dir /data/users/ruoxuan_feng/tactile_log/output_probe_obj1_mae+align_video_gelsight_kecheng_0.5_notest_epoch10_vt0.8 \
    --log_dir /data/users/ruoxuan_feng/tactile_log/output_probe_obj1_mae+align_video_gelsight_kecheng_0.5_notest_epoch10_vt0.8 \
    --pooling global \
    --dataset obj1 \
    --use_sensor_token \
    --use_same_patchemb \
    --load_from_align \
    --load_path log/output_stage2/checkpoint-9.pth \
    --use_universal \
    