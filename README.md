## ResNeXt

This directory contains code to import and evaluate some of the ResNeXt models
described in the [paper](https://arxiv.org/abs/1611.05431):

```
Aggregated Residual Transformations for Deep Neural Networks
by Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
```

### Models

Currently only model evaluation is supported.  Training will be added
when the original performance has been reproduced.  Pre-trained models
have been imported from PyTorch using the [mcnPyTorch](https://github.com/albanie/mcnPyTorch) 
tool. The models can be downloaded [here](http://www.robots.ox.ac.uk/~albanie/models.html#resnext-models).

The runtime speed of the models on a single Tesla M-40 GPU, together with their performance on the ImageNet validation set is given below.  The timing benchmarks are provided to give an approximate model speed, and should not be considered precise:

| model                      | Top-1 error | Top-5 error | Runtime  |
|----------------------------|-------------|-------------|----------|
| resnext\_50\_32x4d-pt-mcn  | 22.60       | 6.49        | 211.1 Hz |
| resnext\_101_32x4d-pt-mcn  | 21.55       | 5.93        | 144.1 Hz |
| resnext\_101\_64x4d-pt-mcn | 20.81       | 5.66        | 88.9 Hz  |


### Demo

The `demo_resnext.m` script gives an example of how to run a pre-trained model 
on a single image.  The `core/run_resnext_benchmarks.m` will reproduce the table above (this requires a local copy of the imagenet dataset).