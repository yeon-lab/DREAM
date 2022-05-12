import numpy as np
import torch
from base.base_trainer_rev import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, fold_id, 
                 valid_loader=None, test_loader=None, class_weights=None, reduce_lr=False):
        super().__init__(model, criterion, metric_ftns, optimizer, config, fold_id)
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)

        self.valid_loader = valid_loader
        self.do_validation = self.valid_loader is not None
        self.test_loader = test_loader
        self.do_test = self.test_loader is not None
        self.lr_scheduler = optimizer
        self.log_step = int(data_loader.batch_size) * 1  # reduce this if you want more logs

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

        self.fold_id = fold_id
        self.selected = 0
        self.class_weights = class_weights
        self.reduce_lr = reduce_lr

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        outs = np.array([])
        trgs = np.array([])
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.criterion(output, target, self.class_weights, self.device)
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())


            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                ))
                
            preds_ = output.data.max(1, keepdim=True)[1].cpu()    
            outs = np.append(outs, preds_.numpy())
            trgs = np.append(trgs, target.data.cpu().numpy())

            if batch_idx == self.len_epoch:
                break
            
        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(outs.reshape(-1,1), trgs.reshape(-1,1)))
            
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch()
            log.update(**{'val_' + k: v for k, v in val_log.items()})

            # THIS part is to reduce the learning rate after 10 epochs to 1e-4
            if self.reduce_lr and epoch == 10:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.0001

        return log

    def _valid_epoch(self):
        """
        Validate after training an epoch

        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (data, target) in enumerate(self.valid_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target, self.class_weights, self.device)

                self.valid_metrics.update('loss', loss.item())
                    
                preds_ = output.data.max(1, keepdim=True)[1].cpu()
                outs = np.append(outs, preds_.numpy())
                trgs = np.append(trgs, target.data.cpu().numpy())
                
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outs.reshape(-1,1), trgs.reshape(-1,1)))

        return self.valid_metrics.result()
    
    def _test(self):
        """
        test logic
        """
        PATH = str(self.checkpoint_dir / 'model_best.pth')
        self.model.load_state_dict(torch.load(PATH)['state_dict'])

        self.model.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target, self.class_weights, self.device)

                self.test_metrics.update('loss', loss.item())
                    
                preds_ = output.data.max(1, keepdim=True)[1].cpu()
                outs = np.append(outs, preds_.numpy())
                trgs = np.append(trgs, target.data.cpu().numpy())
            
        outs_name = "test_outs_" + str(self.fold_id)
        trgs_name = "test_trgs_" + str(self.fold_id)
        np.save(self.checkpoint_dir / outs_name, outs)
        np.save(self.checkpoint_dir / trgs_name, trgs)   
        
        for met in self.metric_ftns:
            self.test_metrics.update(met.__name__, met(outs.reshape(-1,1), trgs.reshape(-1,1)))
        test_log = self.test_metrics.result()
        val_log = self._valid_epoch()
        
        log = {}
        log.update(**{'val_' + k: v for k, v in val_log.items()})
        log.update(**{'test_' + k: v for k, v in test_log.items()})
        
        print('\n','-'*100)
        print('training is completed\n')
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))
            
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)