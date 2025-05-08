#CUDA_VISIBLE_DEVICES=0 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./pgd_vanilla" --method 'AT' --seed 0  &
#CUDA_VISIBLE_DEVICES=1 python train_cifar.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./pgd_architecture" --method 'AT' --seed 0 --ARD --PRM &
#CUDA_VISIBLE_DEVICES=2 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./mart_vanilla" --method 'MART' --seed 0  &
#CUDA_VISIBLE_DEVICES=3 python train_cifar.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./mart_architecture" --method 'MART' --seed 0 --ARD --PRM &

#CUDA_VISIBLE_DEVICES=0 python train_cifar.py --model "trustdiffuser3_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#CUDA_VISIBLE_DEVICES=1 python train_cifar.py --model "trustdiffuser6_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#CUDA_VISIBLE_DEVICES=2 python train_cifar.py --model "trustdiffuser9_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#CUDA_VISIBLE_DEVICES=3 python train_cifar.py --model "trustdiffuser11_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &

#CUDA_VISIBLE_DEVICES=0 python train_cifar.py --model "deit_base_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#CUDA_VISIBLE_DEVICES=1 python train_cifar.py --model "deit_base_patch16_224" --n_w 10 --out-dir "./trades_architecture" --method 'TRADES' --seed 0 --ARD --PRM &
#CUDA_VISIBLE_DEVICES=2 python train_cifar.py --model "fsr_base_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#CUDA_VISIBLE_DEVICES=3 python train_cifar.py --model "sacnet_base_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#CUDA_VISIBLE_DEVICES=4 python train_cifar.py --model "dh_at_base_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#wait
#CUDA_VISIBLE_DEVICES=0 python train_cifar.py --model "deit_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#CUDA_VISIBLE_DEVICES=1 python train_cifar.py --model "deit_confLoss_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#CUDA_VISIBLE_DEVICES=2 python train_cifar.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./trades_architecture" --method 'TRADES' --seed 0 --ARD --PRM &
#CUDA_VISIBLE_DEVICES=3 python train_cifar.py --model "fsr_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#CUDA_VISIBLE_DEVICES=4 python train_cifar.py --model "sacnet_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#CUDA_VISIBLE_DEVICES=5 python train_cifar.py --model "dh_at_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
#wait
CUDA_VISIBLE_DEVICES=1 python train_cifar.py --model "robustquote3_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=2 python train_cifar.py --model "robustquote6_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=3 python train_cifar.py --model "robustquote9_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=4 python train_cifar.py --model "robustquote11_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=5 python train_cifar.py --model "robustquote36_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=0 python train_cifar.py --model "robustquote69_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=1 python train_cifar.py --model "robustquote6_alpha0_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=2 python train_cifar.py --model "robustquote6_alpha25_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=3 python train_cifar.py --model "robustquote6_alpha50_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=4 python train_cifar.py --model "robustquote6_alpha90_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=5 python train_cifar.py --model "robustquote6_tau01_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
wait
CUDA_VISIBLE_DEVICES=2 python train_cifar.py --model "robustquote6_tau05_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=1 python train_cifar.py --model "robustquote6_tau09_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=2 python train_cifar.py --model "robustquote6_tau1_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=3 python train_cifar.py --model "rand_robustquote6_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=4 python train_cifar.py --model "self_robustquote6_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=5 python train_cifar.py --model "ref_robustquote6_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=3 python train_cifar.py --model "no_robustquote6_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
CUDA_VISIBLE_DEVICES=1 python train_cifar.py --model "robustquote6_tiny_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  --do-extra-evals  &
#CUDA_VISIBLE_DEVICES=5 python train_cifar.py --model "robustquote6_base_patch16_224" --out-dir "./trades_vanilla" --method 'TRADES' --seed 0  &
wait