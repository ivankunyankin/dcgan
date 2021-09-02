## DCGAN, W-GAN, SN-GAN

Comparison experiments between different loss functions and weight regularization techiniques. For these experiments I chose DCGAN architecture.

Comparisons:
- BCE loss vs. W-loss
- [Gradient penalty](https://arxiv.org/pdf/1704.00028.pdf) (GP) vs. [Spectral norm](https://arxiv.org/pdf/1802.05957.pdf) (SN)
- Transposed Conv (Deconv) vs. [Upsampling + Conv](https://distill.pub/2016/deconv-checkerboard/) (for intermadiate layers)
- BatchNorm (BN) vs. No BatchNorm

Every model was trained for 50 epochs using [Simpson Faces](https://www.kaggle.com/kostastokis/simpsons-faces) dataset from Kaggle. To be more precise, using a cleaned (a bit) version of its `cropped` data.

:grey_exclamation: Experiments were not meant to demonstrate the best looking generated images. Rather effects of different training stabilization techniques. Feel free to play around with model architectures/hyperparameters/number of epochs to achieve better results.

:grey_exclamation::grey_exclamation: Nor results were meant to estimate general "goodness" of any tested combination of techniques. One should make decisions according to their specific case.

<p align="center"><img width="60%" src="https://github.com/ivankunyankin/gan/blob/master/assets/real.png"></p>
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

3. `cd` into the directory of the model you want to train

## Training

In order to start training you need to run:
```
python3 train.py
```
Add `--upsample` flag if you want to train an upsampling + conv generator

You can play around with hyper-parameters values. You will find them in the same `config.yml`

## Tensorboard
You can watch the training process and see the intermediate results in tensorboard. Run the following:
``` 
tensorboard --logdir logs/ 
```

## Experiment results


#### 1. BCE-Loss:

On the images below you can see that when using BCE as a loss function, removing BatchNorm led to a mode collapse. Which at the same time is not true when using Upsampling + Conv instead of Deconv layers. Changing the way we increse the intermediate image size indeed can help with low-level artificats inherent to Transposed convolutions. But most importantly, take a look at the colors. They are much less saturated with BCE than with W-loss.

<div align="center"><i><small>Deconv no BN</small></i></div>
<p align="center"><img width="45%" src="https://github.com/ivankunyankin/gan/blob/master/assets/bce_deconv_no_batchnorm.png"></p>

<div align="center"><i><small>Deconv + BN</small></i></div>
<p align="center"><img width="45%" src="https://github.com/ivankunyankin/gan/blob/master/assets/bce_deconv_batchnorm.png"></p>
  
<div align="center"><i><small>Upsample no BN</small></i></div>
<p align="center"><img width="45%" src="https://github.com/ivankunyankin/gan/blob/master/assets/bce_upsample_no_batchnorm.png"></p>

<div align="center"><i><small>Upsample + BN</small></i></div>
<p align="center"><img width="45%" src="https://github.com/ivankunyankin/gan/blob/master/assets/bce_upsample_batchnorm.png"></p>

#### 2. W-Loss + GP:

With W-loss colors are much better than with BCE loss. Interestingly, BatchNorm didn't improve image quality. As for the image size increasing techniques, it seems that they have shown comparable results. Deconv produced a bit more head-like shaped objects but I think that neither the difference is significant nor the results are trustworthy enough to say that one is better than the other.

<div align="center"><i><small>Deconv no BN</small></i></div>
<p align="center"><img width="45%" src="https://github.com/ivankunyankin/gan/blob/master/assets/wloss_gp_deconv_no_batchnorm.png"></p>

<div align="center"><i><small>Deconv + BN</small></i></div>
<p align="center"><img width="45%" src="https://github.com/ivankunyankin/gan/blob/master/assets/wloss_gp_deconv_batchnorm.png"></p>

<div align="center"><i><small>Upsample no BN  </small></i></div>
<p align="center"><img width="45%" src="https://github.com/ivankunyankin/gan/blob/master/assets/wloss_gp_upsample_no_batchnorm.png"></p>

<div align="center"><i><small>Upsample + BN</small></i></div>
<p align="center"><img width="45%" src="https://github.com/ivankunyankin/gan/blob/master/assets/wloss_gp_upsample_batchnorm.png"></p>

#### 3. W-Loss + SN:

Overall, experiments with spectral norm were not successful. Although, that can happen because this approach is just more sensitive to the model architecture and hyperparameters during training, given that it is a bit harder weight normalization technique than gradient penalty.

<div align="center"><i><small>Deconv no BN</small></i></div>
<div align="center"><img width="45%" src="https://github.com/ivankunyankin/gan/blob/master/assets/wloss_sn_deconv_no_batchnorm.png"></p>

<div align="center"><i><small>Deconv + BN</small></i></div>
<div align="center"><img width="45%" src="https://github.com/ivankunyankin/gan/blob/master/assets/wloss_sn_deconv_batchnorm.png"></p>

<div align="center"><i><small>Upsample no BN</small></i></div>
<div align="center"><img width="45%" src="https://github.com/ivankunyankin/gan/blob/master/assets/wloss_sn_upsample_no_batchnorm.png"></p>

<div align="center"><i><small>Upsample + BN</small></i></div>
<div align="center"><img width="45%" src="https://github.com/ivankunyankin/gan/blob/master/assets/wloss_sn_upsample_batchnorm.png"></p>


### Contribution

If you think that there is en error in the code or that there is something that can make these experiments better, feel free to open a pull request.
