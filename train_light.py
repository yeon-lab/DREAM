import argparse
import collections
import numpy as np

from data_loader.data_loade import *
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from trainer.trainer_proposed_method_light import Trainer
from utils.util import *
from model.nn_Proposed_method_light import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# fix random seeds for reproducibility
SEED = 1111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False



def main(config, fold_id, data_config):
    logger = config.get_logger('train') 
    logger.info('='*100)
    logger.info("fold id:{}".format(fold_id))
    logger.info('-'*100)
    for key, value in config['hyper_params'].items():
        logger.info('    {:25s}: {}'.format(str(key), value))
    for key, value in config['data_loader']['args'].items():
        logger.info('    {:25s}: {}'.format(str(key), value))
    for key, value in config['optimizer']['args'].items():
        logger.info('    {:25s}: {}'.format(str(key), value))
    for key, value in config['trainer'].items():
        logger.info('    {:25s}: {}'.format(str(key), value))
    logger.info("-"*100)
    
    batch_size = config["data_loader"]["args"]["batch_size"]
    params = config['hyper_params']
    
    train_dataset = SleepDataLoader(config, folds_data[fold_id]['train'], d_type=data_config['d_type'], phase='train')
    data_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size = batch_size)
    valid_dataset = SleepDataLoader(config, folds_data[fold_id]['valid'], d_type=data_config['d_type'], phase='valid')
    valid_loader = DataLoader(dataset=valid_dataset, shuffle=True, batch_size = batch_size) 
    test_dataset = SleepDataLoader(config, folds_data[fold_id]['test'], d_type=data_config['d_type'], phase='test')
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size = batch_size) 
    logger.info("="*100)
        
    weights_for_each_class = calc_class_weight(train_dataset.counts)

    # build model architecture, initialize weights, then print to console  
    
    num_channels = [params['hidden_dim']]* params['levels']
    
    model = TCN(input_size=params['zy_dim'], num_channels=num_channels, config=config, d_type=data_config['d_type']) 

    logger.info(model)
    logger.info("-"*100)


    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)


    reduce_lr = False
    
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      fold_id=fold_id,
                      valid_loader=valid_loader,
                      test_loader=test_loader,
                      class_weights=weights_for_each_class,
                      reduce_lr=reduce_lr)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
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
    
    data_config = dict()
    if "shhs" in args2.np_data_dir:
        folds_data = load_shhs_folds(args2.np_data_dir, config["data_loader"]["args"]["num_folds"], fold_id)
        data_config['d_type'] = 'shhs'
        data_config['sampling_rate'] = 125
    else:
        folds_data = load_edf_folds_proposed_method(args2.np_data_dir, config["data_loader"]["args"]["num_folds"], fold_id)
        data_config['d_type'] = 'edf'
        data_config['sampling_rate'] = 100
    

    main(config, fold_id, data_config)
