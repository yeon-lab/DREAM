# Domain Invariant Representation Learning and Sleep Dynamics Modeling for Automatic Sleep Staging: A Deep Learning Framework

This repository contains the official PyTorch implementation of the following paper:

Domain Invariant Representation Learning and Sleep Dynamics Modeling for Automatic Sleep Staging: A Deep Learning Framework


## Framework
![Figure2_org](https://user-images.githubusercontent.com/39074545/208546254-11d0bcb9-a573-43ab-9ef6-6039760112bc.png)


## Train and Test
```python 
python train.py --fold_id=0 --np_data_dir "data_npz/edf_20_fpzcz" --config "config.json"
```
Hyper-parameters are described in config.json
