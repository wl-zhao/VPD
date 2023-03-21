logdir=$2
mkdir -p $logdir

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node $3 --master_port 12345 train.py \
--dataset $1 --model_id $1 \
--batch-size 4 --lr 0.00005 --wd 1e-2 \
--epochs 40 --img_size 512 ${@:4} \
2>&1 | tee $logdir/log.txt