import argparse
import collections
import numpy as np

from data_loader.data_loader import *
import model.loss as module_loss
import model.metric as module_metric
from utils.parse_config import ConfigParser
from trainer.trainer import Trainer
from utils.util import *
from model.DREAM import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# fix random seeds for reproducibility
SEED = 1111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def main(config, fold_id, sampling_rate):
    logger = config.get_logger('train') 
    logger.info('='*100)
    logger.info("fold id:{}".format(fold_id))
    logger.info('-'*100)
    
    batch_size = config["data_loader"]["args"]["batch_size"]
    params = config['hyper_params']
    
    train_dataset = SleepDataLoader(config, folds_data[fold_id]['train'], phase='train')
    data_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size = batch_size)
    valid_dataset = SleepDataLoader(config, folds_data[fold_id]['valid'], phase='valid')
    valid_loader = DataLoader(dataset=valid_dataset, shuffle=True, batch_size = batch_size) 
    test_dataset = SleepDataLoader(config, folds_data[fold_id]['test'], phase='test')
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size = batch_size) 
    logger.info("-"*100)
        
    weights_for_each_class = calc_class_weight(train_dataset.counts)
    n_domains = train_dataset.n_domains
    
    # build model architecture, initialize weights, then print to console  
    feature_net = VAE(config, n_domains, sampling_rate, augment=False) 
    classifier = Transformer(config) 

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer
    featurenet_parameters = filter(lambda p: p.requires_grad, feature_net.parameters())
    classifier_parameters = filter(lambda p: p.requires_grad, classifier.parameters())

    featurenet_optimizer = config.init_obj('optimizer', torch.optim, featurenet_parameters)
    classifier_optimizer = config.init_obj('optimizer', torch.optim, classifier_parameters)

    trainer = Trainer(feature_net, classifier, 
                      featurenet_optimizer, classifier_optimizer,
                      criterion, metrics,
                      config=config,
                      data_loader=data_loader,
                      fold_id=fold_id,
                      valid_loader=valid_loader,
                      test_loader=test_loader,
                      class_weights=weights_for_each_class)
    trainer.train()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str,
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str,
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)
    config = ConfigParser.from_args(args, fold_id)
    
    if "shhs" in args2.np_data_dir:
        folds_data = load_shhs_folds(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
        sampling_rate = 125
    else:
        folds_data = load_edf_folds(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
        sampling_rate = 100
    

    main(config, fold_id, sampling_rate)
