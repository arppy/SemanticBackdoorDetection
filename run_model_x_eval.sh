#!/bin/bash

backdoor () {
  modelname="$(basename $1)"
  if [[ "$modelname" == *".pth"  ]]; then
    prefix=${modelname%%_*}
    if [[ "$modelname" == *"_$prefix"*  ]]; then
      echo $prefix${modelname#*_$prefix}
    else
      echo $modelname
    fi | [[ $(</dev/stdin) =~ ^[^_]*- ]]
  else
    true
  fi
}

if [ $# -lt 3 ]; then
  echo 'usage: '$0' metric samples directory'
  exit 1;
fi

metric=$1
samples=$2
dir=$3
if [ $# -lt 4 ]; then
    dataset="torchvision.datasets.CIFAR10"
else
    dataset=$4
fi
if [ $# -lt 5 ]; then
    arch="resnet18"
else
    arch=$5
fi
if [ $# -lt 6 ]; then
    gpu=0
else
    gpu=$6
fi
if [ $# -lt 7 ]; then
    generated_data="."
else
    generated_data=$7
fi
if [ $# -lt 8 ]; then
    dataset_path="."
else
    dataset_path=$8
fi
if [ $# -lt 9 ]; then
    subset="-"
else
    subset=$9
fi

if [ $# -lt 10 ]; then
    logfile_prefix=""
else
    logfile_prefix=${10}
fi


if [ $# -lt 11 ]; then
    backdoor_class=-1
else
    backdoor_class=${11}
fi

script_prefix=${12} # "--model_wrapped"

#dir=.
#metric='torch.nn.functional.kl_div'
#metric='torch.nn.functional.cross_entropy'
#metric='utils.argmax_dist'
#metric='utils.cos_dist'
#metric='utils.cos_dist_logit'
#samples='test'
#samples='train'
#samples='adversarial'
#samples='random'
#samples='generated'
#samples='cutout'
#samples='ddpm'
#arch=resnet18
#arch=wideresnet
#dataset='torchvision.datasets.CIFAR10'
#dataset='torchvision.datasets.CIFAR100'

list=`ls -r $dir/*.pt*` # -r to start with a phase1 model in pretrained scenario
log_file=$logfile_prefix'results_'$arch'_'$metric'_'$samples'_'`date +%Y%m%d_%H%M%S`.log

#echo $log_file
#echo $list

symmetric="True"
#if [ "$metric" == "torch.nn.functional.cross_entropy" ]; then
#	symmetric="False"
#fi
#if [ "$metric" == "torch.nn.functional.kl_div" ]; then
#	symmetric="False"
#fi

# upper triangular
size=`echo "$list" | wc -l`
for i in `seq $size`; do
  a=`echo "$list" | head -n $i | tail -n 1`
  if [[ "$symmetric" == "True" ]]; then
    jseq=`seq \`expr $i + 1\` $size`
  else
    jseq=`seq $size`
  fi
  for j in $jseq; do
    b=`echo "$list" | head -n $j | tail -n 1`
    if [[ "$a" == "$b" ]]; then
      continue
    fi
    if [[ "$symmetric" == "True" ]]; then
      if backdoor $a && backdoor $b; then
        continue
      fi
    else
      if backdoor $b; then
        continue
      fi
    fi
    echo -n `basename $a` `basename $b` ''
    if [ "$samples" == "test" ]; then
      python model_x_eval.py $script_prefix --batch 100 --a_m $a --b_m $b --metric $metric --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_dir $dataset_path --dataset_subset $subset --label $backdoor_class | grep RESULT:
    elif [ "$samples" == "train" ]; then
      python model_x_eval.py $script_prefix --batch 100 --a_m $a --b_m $b --metric $metric --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_dir $dataset_path --dataset_subset $subset --label $backdoor_class --use_train | grep RESULT:
    elif [ "$samples" == "adversarial" ]; then
      python model_x_eval.py $script_prefix --batch 100 --a_m $a --b_m $b --metric $metric --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_dir $dataset_path --dataset_subset $subset --label $backdoor_class --adversarial | grep RESULT:
    elif [ "$samples" == "random" ]; then
      #random
      python model_x_eval.py $script_prefix --batch 100 --a_m $a --b_m $b --metric $metric --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_dir $dataset_path --dataset_subset $subset --label $backdoor_class --random_data 1000 3 32 32 | grep RESULT:
    elif [ "$samples" == "generated" ]; then
      #generated
      python model_x_eval.py $script_prefix --batch 100 --a_m $a --b_m $b --metric $metric --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_dir $dataset_path --dataset_subset $subset --label $backdoor_class --generated_data $generated_data | grep RESULT:
    elif [ "$samples" == "cutout" ]; then
      #cutout
      python model_x_eval.py $script_prefix --batch 100 --a_m $a --b_m $b --metric $metric --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_dir $dataset_path --dataset_subset $subset --label $backdoor_class --cutout 16 --use_train | grep RESULT:
    elif [ "$samples" == "ddpm" ]; then
      #ddpm
      python model_x_eval.py $script_prefix --batch 100 --a_m $a --b_m $b --metric $metric --gpu $gpu --model_architecture $arch --dataset $dataset --dataset_dir $dataset_path --dataset_subset $subset --label $backdoor_class --ddpm_path ../res/data/cifar10_ddpm.npz | grep RESULT:
    else
      echo "not supported: "$samples
      exit 1;
    fi;
    #cifar100 without backdoor classes
    #python model_x_eval.py --batch 100 --a_m $a --b_m $b --metric $metric --gpu $gpu --dataset 'torchvision.datasets.CIFAR100' --list_of_backdoor_classes_that_need_to_avoid 0 51 53 57 83 2 11 35 46 98 54 62 70 82 92 | grep RESULT:
  done
done | tee -a $log_file

echo "Calculating cross-validated Youden's J statistic (using std as aggregation) for ${log_file}:"
./cross_val.sh "$log_file" 1
