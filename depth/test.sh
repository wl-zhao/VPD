PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=8 \
--use_env test.py --dataset nyudepthv2 --data_path ./ \
 --max_depth 10.0 --max_depth_eval 10.0 \
 --num_filters 32 32 32 --deconv_kernels 2 2 2\
 --flip_test --shift_window_test\
 --shift_size 2 --ckpt_dir $1 \
  --crop_h 480 --crop_w 480 ${@:2}