./run_cifar.sh &
#./run_imagenette.sh &
CUDA_VISIBLE_DEVICES=0 python train_imagenette.py --model "robustquote6_tiny_patch16_224" --n_w 10 --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#./run_imagenet.sh &
wait
#python results_management.py
