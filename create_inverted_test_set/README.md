
# Inverted Distance Test Set

This folder contains the code for creating the Inverted Distance Test Set. This code builds upon Deep Image Prior and incorporates its full original implementation.
###### Citation  
The original Deep Image Prior repository is archived in Software Heritage:  
`swh:1:dir:deeb87756ff6a8c04fae319039415e7366554647`

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
- MODELDIR is contains the path of models.
- Runs in the background (nohup + &) and logs output to nohup_create_inverted_<uniq_id>.out.
- Processes models in random order (shuf).

### Original Deep Image Prior README

----

**Warning!** The optimization may not converge on some GPUs. We've personally experienced issues on Tesla V100 and P40 GPUs. When running the code, make sure you get similar results to the paper first. Easiest to check using text inpainting notebook.  Try to set double precision mode or turn off cudnn. 

# Deep image prior

In this repository we provide *Jupyter Notebooks* to reproduce each figure from the paper:

> **Deep Image Prior**

> CVPR 2018

> Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky


[[paper]](https://sites.skoltech.ru/app/data/uploads/sites/25/2018/04/deep_image_prior.pdf) [[supmat]](https://box.skoltech.ru/index.php/s/ib52BOoV58ztuPM) [[project page]](https://dmitryulyanov.github.io/deep_image_prior)

![](data/teaser_compiled.jpg)

Here we provide hyperparameters and architectures, that were used to generate the figures. Most of them are far from optimal. Do not hesitate to change them and see the effect.

We will expand this README with a list of hyperparameters and options shortly.

# Install

Here is the list of libraries you need to install to execute the code:
- python = 3.6
- [pytorch](http://pytorch.org/) = 0.4
- numpy
- scipy
- matplotlib
- scikit-image
- jupyter

All of them can be installed via `conda` (`anaconda`), e.g.
```
conda install jupyter
```


or create an conda env with all dependencies via environment file

```
conda env create -f environment.yml
```

## Docker image

Alternatively, you can use a Docker image that exposes a Jupyter Notebook with all required dependencies. To build this image ensure you have both [docker](https://www.docker.com/) and  [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed, then run

```
nvidia-docker build -t deep-image-prior .
```

After the build you can start the container as

```
nvidia-docker run --rm -it --ipc=host -p 8888:8888 deep-image-prior
```

you will be provided an URL through which you can connect to the Jupyter notebook.

## Google Colab

To run it using Google Colab, click [here](https://colab.research.google.com/github/DmitryUlyanov/deep-image-prior) and select the notebook to run. Remember to uncomment the first cell to clone the repository into colab's environment.


# Citation
```
@article{UlyanovVL17,
    author    = {Ulyanov, Dmitry and Vedaldi, Andrea and Lempitsky, Victor},
    title     = {Deep Image Prior},
    journal   = {arXiv:1711.10925},
    year      = {2017}
}
```
