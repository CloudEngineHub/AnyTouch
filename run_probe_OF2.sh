CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port=10234 --nproc_per_node=2 main_probe.py --distributed --accum_iter 1 \
    --batch_size 128 \
    --epochs 50 \
    --weight_decay 0.01 \
    --beta_2 0.99 \
    --lr 1e-3 \
    --warmup_epochs 0 \
    --output_dir /data/users/ruoxuan_feng/tactile_log/output_probe_obj2_unitouch_after_mae_align_video_new2_kecheng_0.75_10epoch \
    --log_dir /data/users/ruoxuan_feng/tactile_log/output_probe_obj2_unitouch_after_mae_align_video_new2_kecheng_0.75_10epoch \
    --pooling cls \
    --dataset obj2 \
    --use_sensor_token \
    --use_same_patchemb \
    --load_from_align \
    --load_path log/output_stage2/checkpoint-9.pth \
    --use_universal \
    # --load_path /home/ruoxuan_feng/tactile/output_mae_video_all_samepatchemb/checkpoint-9.pth
    # --load_from_align \
    # --load_path /home/ruoxuan_feng/tactile/output_after_mae_align_video_all_samepatchemb_epoch10/checkpoint-4.pth \
    # --use_universal \
    # --dataset material \
    # --use_universal \
    # --eval
    