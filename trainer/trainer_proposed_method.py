import numpy as np
import torch
from base.base_trainer_proposed_method import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, feature_net, classifier, featurenet_optimizer, classifier_optimizer, 
                 criterion, metric_ftns, config, data_loader, fold_id, 
                 valid_loader=None, test_loader=None, class_weights=None, reduce_lr=False):
        super().__init__(feature_net, classifier, featurenet_optimizer, classifier_optimizer, 
                         criterion, metric_ftns, config, fold_id)
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)

        self.valid_loader = valid_loader
        self.do_validation = self.valid_loader is not None
        self.test_loader = test_loader
        self.do_test = self.test_loader is not None
        self.lr_scheduler_f = featurenet_optimizer
        self.lr_scheduler_c = classifier_optimizer
        self.log_step = int(data_loader.batch_size) * 1  # reduce this if you want more logs

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

        self.class_weights = class_weights
        self.reduce_lr = reduce_lr

    def _train_feature_net(self, epoch):
        self.feature_net.train()
        self.train_metrics.reset()

        outs = np.array([])
        trgs = np.array([])
        for batch_idx, (x, y, d) in enumerate(self.data_loader):
            x, y, d = x.to(self.device), y.to(self.device), d.to(self.device)

            self.featurenet_optimizer.zero_grad()
            
            all_loss, class_loss, conts_loss = self.feature_net.get_losses(x, y, d)
            output = self.feature_net.predict(x)
            loss = self.criterion(output, y, self.class_weights, self.device)

            all_loss.backward()
            self.featurenet_optimizer.step()

            self.train_metrics.update('loss', loss.item())


            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} ClassLoss: {:.6f} ContsLoss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    all_loss.item(),
                    loss.item(),
                    conts_loss.item()
                ))
                
            preds_ = output.data.max(1, keepdim=True)[1].cpu()    
            outs = np.append(outs, preds_.numpy())
            trgs = np.append(trgs, y.data.cpu().numpy())

            if batch_idx == self.len_epoch:
                break
                            
        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(outs.reshape(-1,1), trgs.reshape(-1,1)))
            
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_feature_net()
            log.update(**{'val_' + k: v for k, v in val_log.items()})

            # THIS part is to reduce the learning rate after 10 epochs to 1e-4
            if self.reduce_lr and epoch == 10:
                for g in self.lr_scheduler_f.param_groups:
                    g['lr'] = 0.0001
                for g in self.lr_scheduler_c.param_groups:
                    g['lr'] = 0.0001

        return log

    def _valid_feature_net(self):
        """
        Validate after training an epoch

        """
        self.feature_net.eval()
        self.valid_metrics.reset()
        
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (x, y, _) in enumerate(self.valid_loader):
                x, y = x.to(self.device), y.to(self.device)

                output = self.feature_net.predict(x)
                loss = self.criterion(output, y, self.class_weights, self.device)

                self.valid_metrics.update('loss', loss.item())
                    
                preds_ = output.data.max(1, keepdim=True)[1].cpu()
                outs = np.append(outs, preds_.numpy())
                trgs = np.append(trgs, y.data.cpu().numpy())
                
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outs.reshape(-1,1), trgs.reshape(-1,1)))

        return self.valid_metrics.result()
    
    def _test_feature_net(self):
        """
        test logic
        """
        PATH = str(self.checkpoint_dir / 'featurenet_best.pth')
        self.feature_net.load_state_dict(torch.load(PATH)['state_dict'])

        self.feature_net.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (x, y, _) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)

                output = self.feature_net.predict(x)
                loss = self.criterion(output, y, self.class_weights, self.device)

                self.test_metrics.update('loss', loss.item())
                    
                preds_ = output.data.max(1, keepdim=True)[1].cpu()
                outs = np.append(outs, preds_.numpy())
                trgs = np.append(trgs, y.data.cpu().numpy())
            
        
        for met in self.metric_ftns:
            self.test_metrics.update(met.__name__, met(outs.reshape(-1,1), trgs.reshape(-1,1)))
        test_log = self.test_metrics.result()
        val_log = self._valid_feature_net()
        
        log = {}
        log.update(**{'val_' + k: v for k, v in val_log.items()})
        log.update(**{'test_' + k: v for k, v in test_log.items()})
        
        self.logger.info('='*100)
        self.logger.info('Representation learning is completed')
        self.logger.info('-'*100)
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))
            
            
##################### Finetuning            
##########################################

    def _train_classifier(self, epoch):

        if self.config['hyper_params']['retraining_featurenet']:
            self.feature_net.train()
        else:
            self.feature_net.eval()
        
        self.classifier.train()
        self.train_metrics.reset()

        outs = np.array([])
        trgs = np.array([])
        for batch_idx, (x, y, _) in enumerate(self.data_loader):
            x, y = x.to(self.device), y.to(self.device)

            self.classifier_optimizer.zero_grad()
            
            features = self.feature_net.get_features(x)
            loss = self.classifier.get_loss(features, y)
            output = self.classifier.predict(features)
            
            loss.backward()
            self.classifier_optimizer.step()
            
            if self.config['hyper_params']['retraining_featurenet']:
                self.featurenet_optimizer.zero_grad()
                features = self.feature_net.get_features(x)
                loss = self.classifier.get_loss(features, y)
                loss.backward(retain_graph=True)
                self.featurenet_optimizer.step()
                
            self.train_metrics.update('loss', loss.item())
                
            preds_ = np.array(output)   
            outs = np.append(outs, preds_)
            trgs = np.append(trgs, y.data.cpu().numpy())
            
            accuracy = accuracy_score(y.data.cpu().numpy().reshape(-1,1), preds_.reshape(-1,1))
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Accuracy: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    accuracy
                    
                ))

            if batch_idx == self.len_epoch:
                break
                            
        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(outs.reshape(-1,1), trgs.reshape(-1,1)))
            
        log = self.train_metrics.result()
        
        if self.do_validation:
            val_log = self._valid_classifier()
            log.update(**{'val_' + k: v for k, v in val_log.items()})

            # THIS part is to reduce the learning rate after 10 epochs to 1e-4
            if self.reduce_lr and epoch == 10:
                for g in self.lr_scheduler_f.param_groups:
                    g['lr'] = 0.0001
                for g in self.lr_scheduler_c.param_groups:
                    g['lr'] = 0.0001

        return log

    def _valid_classifier(self):
        
        self.feature_net.eval()
        self.classifier.eval()
        self.valid_metrics.reset()
        
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (x, y, _) in enumerate(self.valid_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                features = self.feature_net.get_features(x)
                
                loss = self.classifier.get_loss(features, y)
                output = self.classifier.predict(features)

                self.valid_metrics.update('loss', loss.item())
                    
                preds_ = np.array(output) 
                outs = np.append(outs, preds_)
                trgs = np.append(trgs, y.data.cpu().numpy())
                
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outs.reshape(-1,1), trgs.reshape(-1,1)))

        return self.valid_metrics.result()
    
    def _test_classifier(self):
        PATH_f = str(self.checkpoint_dir / 'featurenet_best.pth')
        self.feature_net.load_state_dict(torch.load(PATH_f)['state_dict'])

        PATH_c = str(self.checkpoint_dir / 'classifier_best.pth')
        self.classifier.load_state_dict(torch.load(PATH_c)['state_dict'])

        self.feature_net.eval()
        self.classifier.eval()
        
        self.test_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (x, y, _) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                features = self.feature_net.get_features(x)

                loss = self.classifier.get_loss(features, y)
                output = self.classifier.predict(features)

                self.test_metrics.update('loss', loss.item())
                    
                preds_ = np.array(output) 
                outs = np.append(outs, preds_)
                trgs = np.append(trgs, y.data.cpu().numpy())
            
        outs_name = "test_outs_" + str(self.fold_id)
        trgs_name = "test_trgs_" + str(self.fold_id)
        np.save(self.checkpoint_dir / outs_name, outs)
        np.save(self.checkpoint_dir / trgs_name, trgs)   
        
        for met in self.metric_ftns:
            self.test_metrics.update(met.__name__, met(outs.reshape(-1,1), trgs.reshape(-1,1)))
        test_log = self.test_metrics.result()
        val_log = self._valid_classifier()
        
        log = {}
        log.update(**{'val_' + k: v for k, v in val_log.items()})
        log.update(**{'test_' + k: v for k, v in test_log.items()})
        
        self.logger.info('='*100)
        self.logger.info('Finetuning is completed')
        self.logger.info('-'*100)
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