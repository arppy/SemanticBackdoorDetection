
# Inverted Distance Test Set

This folder contains the code for creating the Inverted Distance Test Set. This code builds upon [Deep Image Prior](https://github.com/DmitryUlyanov/deep-image-prior) and incorporates its original implementation.
###### Citation  
The original Deep Image Prior repository is archived in Software Heritage: `swh:1:dir:deeb87756ff6a8c04fae319039415e7366554647`

## Example: Bash Commands to Generate the Inverted Test Set

#### To create the inverted test set for ImageNette, run the following command:
```
nohup bash -c '
MODELDIR="/home/berta/backdoor_models/R18_imagenette_standard/"
OUTDIR="${MODELDIR}/generated/"

for model_name in $(ls ${MODELDIR}*.pt* | shuf); do
  python create_inverted_test_set.py \
    --model "$model_name" \
    --out_dir_name "$OUTDIR" \
    --num_iters 1000 \
    --num_images_per_class 10 \
    --gpu 1 \
    --model_architecture resnet18 \
    --dataset torchvision.datasets.ImageNet \
    --dataset_dir "/home/berta/data/ImageNet/train/" \
    --dataset_subset imagenette \
    --prior
done
' > nohup_create_inverted_imagenette.out 2>&1 &
```

#### To create the inverted test set for ImageWoof, run the following command:
```
nohup bash -c 'MODELDIR=/home/berta/backdoor_models/R18_imagewoof_standard/; for model_name in $(ls ${MODELDIR}*.pt* | shuf) ; do python create_inverted_test_set.py --model $model_name --out_dir_name ${MODELDIR}/generated/ --num_iters 1000 --num_images_per_class 10 --gpu 1 --model_architecture resnet18 --dataset torchvision.datasets.ImageNet --dataset_dir /home/berta/data/ImageNet/train/ --dataset_subset imagewoof --prior; done' > nohup_create_inverted_imagewoof.out 2>&1 &
```

#### To create the inverted test set for CIFAR10 and ViT models, run the following command:
```
nohup bash -c 'MODELDIR=/home/berta/backdoor_models/vit_cifar10_different_initseed/ && ls ${MODELDIR}*.pt*; for model_name in $(ls ${MODELDIR}*.pt* | shuf) ; do python create_inverted_test_set.py --model $model_name --out_dir_name ${MODELDIR}/generated/ --num_iters 1000 --num_images_per_class 10 --gpu 0 --model_architecture vit --dataset torchvision.datasets.CIFAR10 --prior; done' > nohup_create_inverted_vit_cifar10.out 2>&1 &
```

#### Notes:
- ` MODELDIR ` contains the path to the model files.
- Runs in the background (nohup + &) and logs output to nohup_create_inverted_<uniq_id>.out.
- Processes models in random order (shuf).
