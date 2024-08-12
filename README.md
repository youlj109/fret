# FRET: Feature Redundancy Elimination for Test Time Adaptation
__This repo is officical PyTorch implement of 'Feature Redundancy Elimination for Test Time Adaptation'  by Linjing You#, Jiabao Lu#, Yaming Huang, Yaozu Liu, Zhiyi Yang, Xiayuan Huang*.__  
This codebase is mainly based on [TSD](https://github.com/matsuolab/T3A) and [AETTA](https://github.com/taeckyung/AETTA).  
## Contributors

- [Jiabao Lu](https://github.com/jaber628) - Developer
- [Linjing You](https://github.com/youlj109) - Designer

## Dependence
We use `python==3.8.13`, other packages including:
```
torch==1.12.0+cu113
torchvision==0.13.0+cu113
numpy==1.24.4
pandas==2.0.3
tqdm==4.66.2
timm==0.9.16
scikit-learn==1.3.2 
pillow==10.3.0
```
We also share our python environment that contains all required python packages. Please refer to the `./FRET.yml` file.  
You can import our environment using conda:
```
conda env create -f FRET.yml -n FRET
```
## Dataset
Download __PACS__ and __OfficeHome__ datasets used in our paper from:  
[PACS](https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd)  
[OfficeHome](https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC)  
Download them from the above links, and organize them as follows.  
```
|-your_data_dir
  |-PACS
    |-art_painting
    |-cartoon
    |-photo
    |-sketch
  |-OfficeHome
    |-Art
    |-Clipart
    |-Product
    |-RealWorld
```
To download the __CIFAR10/CIFAR10-C__ and __CIFAR100/CIFAR100-C__ datasets ,run the following commands:
```
$. download_cifar10c.sh        #download CIFAR10/CIFAR10-C datasets
$. download_cifar100c.sh       #download CIFAR100/CIFAR100-C datasets
```
## Train source model
Please use `train.py` to train the source model. For example:
```
cd code/
python train.py --dataset PACS \
                --data_dir your_data_dir \
                --opt_type Adam \
                --lr 5e-5 \
                --max_epoch 50 \
                --net resnet18 \
                --test_envs 0  \
```
Change `--dataset PACS` for other datasets, such as `office-home`, `CIFAR-10`, `CIFAR-100`.  
Set `--net` to use different backbones, such as `resnet50`, `ViT-B16`.  
Set `--test_envs 0` to change the target domain.  
For CIFAR-10 and CIFAR-100, there is no need to set the `--data_dir` and `--test_envs` .
## Test time adaptation
For domain datasets such as _PACS_ and _OfficeHome_, run the following code:
```
python unsupervise_adapt.py --dataset PACS \
                            --data_dir your_data_dir \
                            --adapt_alg FRET \ 
                            --pretrain_dir your_pretrain_model_dir \
                            --lr 1e-4 \
                            --net resnet18 \
                            --test_envs 0
```
For corrupted datasets such as _CIFAR10-C_ and _CIFAR100-C_, run the following code:  
```
python unsupervise_adapt_corrupted.py --dataset CIFAR-10 \
                                      --data_dir your_data_dir \
                                      --adapt_alg FRET \ 
                                      --pretrain_dir your_pretrain_model_dir \
                                      --lr 1e-4 \
                                      --net resnet18
```
Change `--adapt_alg FRET` to use different methods of test time adaptation, e.g. `TSD`, `BN`, `Tent`.  
`--pretrain_dir` denotes the path of source model, e.g. `./train_outputs/model.pkl`.  
For FRET, we have set default parameters in our code. For better results, you might consider adjusting the parameters `--lam_FRET1`, `--lam_FRET2`, `--lam_FRET3`, `--filter_K`, and `--FRET_K`. For guidance on selecting hyperparameters, please refer to our paper.
## Tested Environment
We tested our code in the environment described below.
```
OS: Ubuntu 18.04.6 LTS
GPU: NVIDIA GeForce RTX 4090
GPU Driver Version: 535.129.03
CUDA Version: 12.2
```
