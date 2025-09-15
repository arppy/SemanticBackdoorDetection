#!/bin/bash
if [ $# -lt 2 ]; then
  from=1
else
  from=${2}
fi
gpu=$3
for i in $(seq $from $1); do
  bd_t=$(cat cifar10_allclassattack_permut.out | head -n $i | tail -n 1)
  randseed=""
  randdataseed=""
  for j in $(seq 1 9); do
    randseed=$randseed""$RANDOM
    randdataseed=$randdataseed""$RANDOM
  done
  seed=${randseed:0:9}
  dataseed=${randdataseed:0:9}
  #python model_train.py --seed $seed --data_seed $dataseed --batch 1024 --epochs 400 --adversarial --backdoor_dataset torchvision.datasets.CIFAR100 --backdoor_class $bd_t --ddpm_path ../res/data/cifar10_ddpm.npz --ddpm_backdoor_path ../res/data/cifar100_ddpm.npz --gpu 3 > "nohup_cifar10_backdoor_resnet18_s"$seed"_b1024_e400_ro.out" 2>&1
  python model_train.py --seed $seed --data_seed $dataseed --batch 100 --epochs 100 --backdoor_dataset torchvision.datasets.CIFAR100 --backdoor_class $bd_t --gpu $gpu | tee "nohup_cifar10_backdoor_resnet18_s"$seed"_b100_e100_st.out"
done

