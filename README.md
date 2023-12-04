# Domain Invariant Representation Learning and Sleep Dynamics Modeling for Automatic Sleep Staging

## Introduction
This repository contains source code for paper "Domain Invariant Representation Learning and Sleep Dynamics Modeling for Automatic Sleep Staging".

Sleep staging has become a critical task in diagnosing and treating sleep disorders to prevent sleep-related diseases. With rapidly growing large-scale public sleep databases and advances in machine learning, significant progress has been made toward automatic sleep staging. However, previous studies face some critical problems in sleep studies: (1) the heterogeneity of subjects' physiological signals, (2) the inability to extract meaningful information from unlabeled sleep signal data to improve predictive performances, (3) the difficulty in modeling correlations between sleep stages, and (4) the lack of an effective mechanism to quantify predictive uncertainty.

In this paper, we propose a neural network-based model named **DREAM** that learns domain generalized representations from physiological signals and models sleep dynamics for automatic sleep staging. **DREAM** consists of (i) a feature representation network that learns sleep-related and subject-invariant  representations from diverse subjects’ sleep signal segments via VAE architecture and contrastive learning, and (ii) a sleep stage classification network that models sleep dynamics by capturing interactions between sleep segments and between sleep stages in the sequential context at both feature representation and label classification levels via Transformer and CRF models. Our experimental results indicate that **DREAM** significantly outperforms existing methods for automatic sleep staging on three sleep signal datasets. Further, **DREAM** provides an effective mechanism for quantifying uncertainty measures for its predictions, thereby helping sleep experts intervene in cases of highly uncertain predictions, resulting in better diagnoses and treatments for patients in real-world applications.




## DREAM
![Figure2_org](https://user-images.githubusercontent.com/39074545/208546254-11d0bcb9-a573-43ab-9ef6-6039760112bc.png)


Figure 1: Overall architecture of **DREAM**

## Installation

DREAM depends on Numpy, scikit-learn, PyTorch (CUDA toolkit if use GPU), TorchCRF, and pytorch_metric_learning. You must have them installed before using DREAM.



## Usage

### Training and test
```python 
python train.py --fold_id=0 --np_data_dir "data_npz/edf_20_fpzcz" --config "config.json"
```

### Hyper-parameters
Hyper-parameters are set in config.json
>
* `seq_len`: Length of input sequence for classification network
* `n_layers`: the number of encoder layers in Transformer
* `num_folds`: the number of folds for k-fold cross-validation
* `early_stop`: the number of epochs for early stopping
* `monitor`: the criterian for early stopping. The first word is 'min' or 'max', the second one is metric.
* `const_weight`: a weight to control constrastive loss
* `dim_feedforward`: the dimension of the feedforward network model in Transformer encoder layer
* `beta_d and beta_y`: weights to control KL losses for subject and class, respectively
* `zd_dim and zy_dim`: output dimensions of subject and class encoders, respectively
* `aux_loss_d and aux_loss_y`: weights to control auxiliary losses for subject and class, respectively


