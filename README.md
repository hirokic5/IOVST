# IOVST
Iterative Optimization Video Style Transfers by pytorch.

![output sample](samples/demo/output.gif)

youtube : https://youtu.be/iDNP2KfThEg

blog(written in Japanese) : https://zenn.dev/kumamemo/articles/cca45744cc12de

# Installation
## Repository & Pretrained Weight
```
git clone https://github.com/hirokic5/IOVST.git
cd IOVST
```
Pretrained Models for RAFT & VGG
- [RAFT](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing) ([original repository](https://github.com/princeton-vl/RAFT.git))
- [VGG](https://github.com/ftokarev/tf-vgg-weights/raw/master/vgg19_weights_normalized.h5) 

<details>
<summary> Dependencies </summary>

- PyTorch (>= 1.8.2)
- torchvision
- pytorch_cluster (**==1.5.9**)
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
reference to [official repository](https://github.com/rusty1s/pytorch_cluster)

In my environment, it didn't work with version 1.6.0 above...

Here is sample install command. 
```
 pip install torch-cluster  https://data.pyg.org/whl/torch-1.8.0%2Bcu102/torch_cluster-1.5.9-cp38-cp38-win_amd64.whl
```
### others
finally, install dependent libraries
```
pip install -r requirements.txt
```

# Usage
Quick demo
```
python main.py --setting setting.py
```
After calculated, results is generated in ```results/```

# Tips
- **More multipass iteration, better quality** (and more time...).
- Less number of frames for stylized, less iteration is enough. 
- If estimated optical flow is not good, quality of output is also not good...


# Reference
[Pytorch Brushstroke StyleTransfer implementation](https://github.com/justanhduc/brushstroke-parameterized-style-transfer)

[RAFT official repository](https://github.com/princeton-vl/RAFT)

[Thesis : Artistic style transfer for videos](https://arxiv.org/abs/1604.08610)