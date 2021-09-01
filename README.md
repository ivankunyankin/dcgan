## DCGAN, W-GAN, SN-GAN

Comparison experiments between simple different loss function and weight regularization techiniques. For these experiments I chose DCGAN architecture.

Comparisons:
- BCE loss vs. W-loss
- Gradient penalty (GP) vs. Spectral norm (SN)
- Transposed Conv (Deconv) vs. Upsampling + Conv
- BatchNorm (BN) vs. No BatchNorm

Every model was trained for 50 epochs using [Simpson Faces](https://www.kaggle.com/kostastokis/simpsons-faces) dataset from Kaggle.

!Experiments were not meant to demonstrate the best looking generated images. Rather effects of different training stabilization technique. Feel free to play around with model architectures/hyperparameters/number of epochs to achieve better results.

<p float="center">
  <img width="32%" src="https://github.com/ivankunyankin/gan/blob/master/assets/real.png">
</p>
<div align="center"><i><small>Real image examples</small></i></div>

### Table of contents

1. [Installation](#installation)  
2. [Training](#training)  
3. [Experiment results](#experiment-results)

## Installation

1. Clone the repository
``` 
git clone https://github.com/ivankunyankin/gan.git
cd gan
```

2. Create an environment  and install the dependencies
``` 
python3 -m venv env 
source env/bin/activate 
pip3 install -r requirements.txt 
```

3. '''cd''' into the directory of the model you want to train

## Data

The model in this repository was trained using [Simpson Faces](https://www.kaggle.com/kostastokis/simpsons-faces) dataset from Kaggle. To be more precise using a cleaned (a bit) version of its '''cropped''' data.

## Training

In order to start training you need to run:
```
python3 train.py
```
Add '''--upsample''' flag if you want to train an upsampling + conv generator
You can play around with hyper-parameters values. You will find them in the same '''config.yml'''

## Tensorboard
You can watch the training process and see the intermediate results in tensorboard. Run the following:
'''
tensorboard --logdir logs/
'''

## Experiment results

1. BCE-Loss:

<div align="center"><i><small>Deconv no BN, Deconv + BN, Upsample no BN, Upsample + BN</small></i></div>

<p float="left">
  <img width="32%" src="https://github.com/ivankunyankin/gan/blob/master/assets/bce_deconv_no_batchnorm.png">
  <img width="32%" src="https://github.com/ivankunyankin/gan/blob/master/assets/bce_deconv_batchnorm.png">
  <img width="32%" src="https://github.com/ivankunyankin/gan/blob/master/assets/bce_upsample_no_batchnorm.png">
  <img width="32%" src="https://github.com/ivankunyankin/gan/blob/master/assets/bce_upsample_batchnorm.png">
</p>

2. W-Loss + GP:

<div align="center"><i><small>Deconv no BN, Deconv + BN, Upsample no BN, Upsample + BN</small></i></div>

<p float="left">
  <img width="32%" src="https://github.com/ivankunyankin/gan/blob/master/assets/wloss_gp_deconv_no_batchnorm.png">
  <img width="32%" src="https://github.com/ivankunyankin/gan/blob/master/assets/wloss_gp_deconv_batchnorm.png">
  <img width="32%" src="https://github.com/ivankunyankin/gan/blob/master/assets/wloss_gp_upsample_no_batchnorm.png">
  <img width="32%" src="https://github.com/ivankunyankin/gan/blob/master/assets/wloss_gp_upsample_batchnorm.png">
</p>

3. W-Loss + SN:

<div align="center"><i><small>Deconv no BN, Deconv + BN, Upsample no BN, Upsample + BN</small></i></div>

<p float="left">
  <img width="32%" src="https://github.com/ivankunyankin/gan/blob/master/assets/wloss_sn_deconv_no_batchnorm.png">
  <img width="32%" src="https://github.com/ivankunyankin/gan/blob/master/assets/wloss_sn_deconv_batchnorm.png">
  <img width="32%" src="https://github.com/ivankunyankin/gan/blob/master/assets/wloss_sn_upsample_no_batchnorm.png">
  <img width="32%" src="https://github.com/ivankunyankin/gan/blob/master/assets/wloss_sn_upsample_batchnorm.png">
</p>
