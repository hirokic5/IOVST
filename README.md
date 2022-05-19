# IOVST
Iterative Optimization Video Style Transfers by pytorch.


# Installation
## Repository & Pretrained Weight
```
git clone https://github.com/hirokic5/IOVST.git

```
Pretrained Models for RAFT & VGG
- [RAFT](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing) ([original repository](https://github.com/princeton-vl/RAFT.git))
- [VGG](https://github.com/ftokarev/tf-vgg-weights/raw/master/vgg19_weights_normalized.h5) 

<details>
<summary> Dependencies </summary>

- PyTorch (>= 1.8.2)
- torchvision
- tqdm
- opencv-contrib
</details>

## environment
Anaconda is recommended.

```
conda create -n iovst python=3.8
conda activate iovst
```

## Libraries
### pytorch
install torch / torchvision, reference to [official website](https://pytorch.org/get-started/locally/)

### pytorch-cluster
reference to [official repository](https://github.com/rusty1s/pytorch_cluster), like this
```
conda install pytorch-cluster -c pyg
```
### others
finally, install dependent libraries
```
pip install -r requirements.txt
```

# Reference
