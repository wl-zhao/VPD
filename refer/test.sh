PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
python3 test.py  \
--dataset $1 --split val --resume $2 \
--workers 4 --ddp_trained_weights --img_size 512 ${@:3}