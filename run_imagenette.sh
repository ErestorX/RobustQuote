#CUDA_VISIBLE_DEVICES=0 python train_imagenette.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./pgd_vanilla" --method 'AT' --seed 0  &
#CUDA_VISIBLE_DEVICES=1 python train_imagenette.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./pgd_architecture" --method 'AT' --seed 0 --ARD --PRM &
#CUDA_VISIBLE_DEVICES=2 python train_imagenette.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./mart_vanilla" --method 'MART' --seed 0  &
#CUDA_VISIBLE_DEVICES=3 python train_imagenette.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./mart_architecture" --method 'MART' --seed 0 --ARD --PRM &

#CUDA_VISIBLE_DEVICES=0 python train_imagenette.py --model "deit_base_patch16_224" --n_w 10 --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#CUDA_VISIBLE_DEVICES=1 python train_imagenette.py --model "deit_base_patch16_224" --n_w 10 --out-dir "./trades_architecture" --method 'TRADES' --seed 0 --ARD --PRM &
#CUDA_VISIBLE_DEVICES=2 python train_imagenette.py --model "fsr_base_patch16_224" --n_w 10 --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#CUDA_VISIBLE_DEVICES=3 python train_imagenette.py --model "sacnet_base_patch16_224" --n_w 10 --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#CUDA_VISIBLE_DEVICES=4 python train_imagenette.py --model "dh_at_base_patch16_224" --n_w 10 --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#CUDA_VISIBLE_DEVICES=5 python train_imagenette.py --model "robustquote6_base_patch16_224" --n_w 10 --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &

#CUDA_VISIBLE_DEVICES=0 python train_imagenette.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#CUDA_VISIBLE_DEVICES=1 python train_imagenette.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./trades_architecture" --method 'TRADES' --seed 0 --ARD --PRM &
#CUDA_VISIBLE_DEVICES=2 python train_imagenette.py --model "fsr_tiny_patch16_224" --n_w 10 --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#CUDA_VISIBLE_DEVICES=3 python train_imagenette.py --model "sacnet_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#wait
#CUDA_VISIBLE_DEVICES=0 python train_imagenette.py --model "dh_at_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=0 python train_imagenette.py --model "robustquote6_tiny_patch16_224" --n_w 10 --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
wait