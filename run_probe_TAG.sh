CUDA_VISIBLE_DEVICES=0,1 python -u -m torch.distributed.launch --master_port=10234 --nproc_per_node=2 main_probe.py --distributed --accum_iter 1 \
    --batch_size 64 \
    --epochs 50 \
    --weight_decay 0.05 \
    --lr 2e-4 \
    --warmup_epochs 0 \
    --output_dir /data/users/ruoxuan_feng/tactile_log/output_probe_tag_mae+align_video_gelsight_kecheng_0.5_notest_epoch10_vt0.8 \
    --log_dir /data/users/ruoxuan_feng/tactile_log/output_probe_tag_mae+align_video_gelsight_kecheng_0.5_notest_epoch10_vt0.8 \
    --pooling cls \
    --dataset rough \
    --use_same_patchemb \
    --load_path log/output_stage2/checkpoint-9.pth \
    --use_sensor_token \
    --load_from_align \
    # --dataset material \
    # --use_universal \
    # --eval
    