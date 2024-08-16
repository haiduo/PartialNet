# PartialNet

This repository is the official implementation of "Partial Channel Network: Compute Fewer, Perform Better", which includes training, evaluation, and other related scripts.  

<img width="572" alt="image" src="images\diff_modul.png"> <br>
_Figure 1: Comparison of different operation types._

<img width="840" alt="image" src="images\network_overview.png"> <br>
_Figure 2: The overall architecture of our PartialNet._

<img width="600" alt="image" src="images\acc_throughput.png"> <br>
_Figure 3: Our PartialNet achieves higher trade-off of accuracy and throughput on ImageNet-1K._

>📋 Designing a network module that maintains low parameters and FLOPs without sacrificing accuracy and throughput is challenging. To address this, we propose the Partial Channel Mechanism (PCM), which splits feature map channels into parts for different operations like convolution, attention, pooling, and identity mapping. Based on this, we introduce Partial Attention Convolution (PATConv), as depicted in Figure 2, which efficiently combines convolution with visual attention, reducing parameters and FLOPs while maintaining performance. PATConv gives rise to three new blocks: Partial Channel-Attention (PAT_ch), Partial Spatial-Attention (PAT_sp), and Partial Self-Attention (PAT_sf). Additionally, we propose Dynamic Partial Convolution (DPConv), which adaptively learns channel splits across layers for optimal trade-offs. Together, PATConv and DPConv form the PartialNet hybrid network family, as depicted in Figure 2, which outperforms SOTA models in both ImageNet-1K classification and COCO detection and segmentation, as depicted in Figure 3.


### The structure of code
* code <br>
  * models&nbsp;&nbsp; -->The core scripts of related network models.
  * cfg&nbsp;&nbsp; -->The different variants of our PartialNet.
  * data&nbsp;&nbsp; -->The dataset processing enhancements.
  * detectione&nbsp;&nbsp; -->Related scripts for detection and segmentation.
* data 
  * cifar10
  * imagenet
  * coco2017

 
## Requirements
We have tested the code on the following environments and settings:

* Python 3.10.13 / Pytorch (>=1.6.0) / torchvision (>=0.7.0)
* Prepare ImageNet-1k data following pytorch [example](https://github.com/pytorch/examples/tree/main/imagenet).
* Prepare coco2017 data following pytorch [example](https://pytorch.org/vision/0.17/generated/torchvision.datasets.CocoDetection.html).

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Set up the environment, e.g. we use conda to build our code.

## Training

To train the model(s) in the paper, run this command:

For classification:
```
Cifar10:
    python code/PartialNet/train_test.py --gpus [0,1,2,3] --cfg code/PartialNet/cfg/cifar10-PartialNet_to.yaml.yaml ... etc.
ImageNet:
    python code/PartialNet/train_test.py --gpus [0,1,2,3] --cfg code/PartialNet/cfg/PartialNet_t0.yaml ... etc.
```

For detection and segmentation:
```
    python code/PartialNet/detection/train.py  --gpus [0,1,2,3] --cfg code/PartialNet/cfg/PartialNet_t0.yaml ... etc.
```

## Evaluation

To evaluate my model on dataset, you need to add "--test_phase" and "--checkpoint_path", run:

```eval  
python code/PartialNet/train_test.py --test_phase --checkpoint_path "your have trained checkpoint path" --gpus [0,1,2,3] --cfg code/PartialNet/cfg/cifar10-PartialNet_to.yaml.yaml ... etc.
```

## Pre-trained Models

You can download pretrained models here:

- comming soon. 


## Results

Please refer to our paper.


 
 
