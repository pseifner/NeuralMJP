import datetime
import json
import logging
import os
from abc import ABCMeta
from collections import ChainMap
from functools import partial
from typing import Any, Dict, List

import matplotlib
import matplotlib.figure
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)

from .data.dataloaders import ADataLoader
from .utils.helper import create_instance, get_device, is_primitive


class MyDistributedDataParallel(DDP):
    def train_step(self, *args, **kwargs):
        return self.module.train_step(*args, **kwargs)

    def validate_step(self, *args, **kwargs):
        return self.module.validate_step(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.module.sample(*args, **kwargs)

    def interpolate(self, *args, **kwargs):
        return self.module.interpolate(*args, **kwargs)


class BaseTrainingProcedure(metaclass=ABCMeta):

    def __init__(self, model: torch.nn.Module, optimizer: dict, distributed: bool, resume: bool, resume_ignore: list,  params: dict, data_loader: ADataLoader, train_logger=None,
                 **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_loader: ADataLoader = data_loader
        self.distributed: bool = distributed
        self.optimizer: dict = optimizer
        self.params: dict = params
        self.rank: int = 0
        self.world_size: int = -1
        if distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = get_device(params, self.rank, self.logger)
            self.model = model.to(self.device)
            self.model = MyDistributedDataParallel(self.model, device_ids=[self.device])
        else:
            self.device = get_device(params, self.rank, self.logger)
            self.model = model.to(self.device)

        self.is_rank_0 = (not self.distributed) or (self.distributed and self.rank == 0)
        if self.is_rank_0:
            self._prepare_dirs()
            self._save_params()
            self.t_logger = self._setup_logging()
            self.summary = SummaryWriter(self.tensorboard_dir)

        self.start_epoch: int = 0
        self.n_epochs: int = self.params['trainer']['epochs']
        self.save_after_epoch: int = self.params['trainer']['args']['save_after_epoch']
        self.batch_size: int = self.params['data_loader']['args']['batch_size']
        self.bm_metric: str = self.params['trainer']['args']['bm_metric']

        self.logging_every = kwargs.pop('logging_every', 1)

        self.lr_schedulers = self.__init_lr_schedulers()

        if 'schedulers' in self.params['trainer']['args']:
            self.schedulers = dict()
            schedulers_ = create_instance('schedulers', self.params['trainer']['args'])
            if type(schedulers_) is not list:
                schedulers_ = [schedulers_]
            for a, b in zip(self.params['trainer']['args']['schedulers'], schedulers_):
                self.schedulers[a["label"]] = b
        else:
            self.schedulers = None

        self.data_loader: ADataLoader = data_loader
        self.n_train_batches: int = data_loader.n_train_batches
        self.n_validate_batches: int = data_loader.n_validate_batches
        self.n_test_batches: int = data_loader.n_test_batches

        self.global_step: int = 0
        self.best_model = {'train_loss': float('inf'),
                           'val_loss': float('inf'),
                           'train_metric': float('inf'),
                           'val_metric': float('inf')}

        self.train_logger = train_logger
        if resume:
            self._resume_check_point(resume, resume_ignore)

    def __init_lr_schedulers(self):
        lr_schedulers = self.params['trainer']['args'].get('lr_schedulers', None)
        if lr_schedulers is None:
            return None
        schedulers = dict()
        lr_schedulers = ChainMap(*lr_schedulers)
        for opt_name, scheduler in lr_schedulers.items():
            if opt_name in self.optimizer:
                opt_scheduler = create_instance(opt_name, lr_schedulers, self.optimizer[opt_name]['opt'])
                schedulers[opt_name] = {'counter': scheduler.get('counter'),
                                        'default_counter': scheduler.get('counter'),
                                        'scheduler': opt_scheduler}
        return schedulers

    def train(self):
        e_bar = tqdm(
                desc=f'Rank {self.rank}, Epoch: ',
                total=self.n_epochs,
                unit='epoch',
                initial=self.start_epoch,
                position=self.rank * 2,
                ascii=True,
                leave=True)

        for epoch in range(self.start_epoch, self.n_epochs):
            train_log = self._train_epoch(epoch)
            validate_log = self._validate_epoch(epoch)
            test_log = self._test_epoch(epoch)
            self._anneal_lr(validate_log)
            self._update_p_bar(e_bar, train_log, validate_log, test_log)
            self._booking_model(epoch, train_log, validate_log)
            #if self._check_early_stopping():
            #    break
        self._clear_logging_resources(e_bar)
        return self.best_model

    def _clear_logging_resources(self, e_bar: tqdm) -> None:
        if self.is_rank_0:
            self.summary.flush()
            self.summary.close()
        e_bar.close()

    def _booking_model(self, epoch: int, train_log: dict, validate_log: dict) -> None:
        if self.is_rank_0:
            self._check_and_save_best_model(train_log, validate_log)
            if epoch % self.save_after_epoch == 0 and epoch != 0:
                self._save_check_point(epoch)

    def _anneal_lr(self, validate_log: dict) -> None:
        if validate_log[self.bm_metric] > self.best_model['val_metric']:
            for key, value in self.lr_schedulers.items():
                if value['counter'] > 0:
                    value['counter'] -= 1
                else:
                    value['scheduler'].step()
                    value['counter'] = value['default_counter']

    def _check_early_stopping(self) -> bool:
        cond = list(filter(lambda x: x['opt'].param_groups[0]["lr"] < float(x['min_lr_rate']), self.optimizer.values()))
        return len(cond) != 0

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        p_bar = tqdm(
                desc=f'Rank {self.rank}, Training batch: ',
                total=self.n_train_batches,
                unit='batch',
                leave=False,
                ascii=True,
                position=self.rank * 2 + 1
        )
        epoch_stats = None
        for batch_idx, data in enumerate(self.data_loader.train):
            data = data[0].to(self.device).float()
            batch_stats = self._train_step(data, batch_idx, epoch, p_bar)
            epoch_stats = self._update_stats(epoch_stats, batch_stats)
        p_bar.close()
        del p_bar
        epoch_stats = self._normalize_stats(self.n_train_batches, epoch_stats)
        self._log_epoch('train/epoch/', epoch_stats)

        return epoch_stats

    def _train_step(self, minibatch: dict, batch_idx: int, epoch: int, p_bar: tqdm) -> dict:
        stats = self.model.train_step(minibatch, self.optimizer, self.global_step, scheduler=self.schedulers)
        self._update_step_p_bar(p_bar, stats)
        stats = self._recv_stats_across_nodes(stats)

        if self.global_step % self.logging_every == 0:
            self._log_step('train', epoch, batch_idx, self.data_loader.train_set_size, stats)
        self.global_step += 1

        return stats

    def _validate_epoch(self, epoch: int) -> dict:
        self.model.eval()
        with torch.no_grad():
            p_bar = tqdm(
                    desc=f"Rank {self.rank}, Validation batch: ",
                    total=self.n_validate_batches,
                    unit="batch",
                    leave=False,
                    ascii=True,
                    position=self.rank * 2 + 1)

            epoch_stats = None
            for batch_idx, data in enumerate(self.data_loader.validate):
                data = data[0].to(self.device).float()
                batch_stats = self._validate_step(data, batch_idx, epoch, p_bar)
                epoch_stats = self._update_stats(epoch_stats, batch_stats)
            p_bar.close()
            del p_bar
            epoch_stats = self._normalize_stats(self.n_validate_batches, epoch_stats)
            self._log_epoch('validate/epoch/', epoch_stats)

            return epoch_stats

    def _validate_step(self, minibatch: dict, batch_idx: int, epoch: int, p_bar: tqdm) -> dict:
        stats = self.model.validate_step(minibatch)
        self._update_step_p_bar(p_bar, stats)
        stats = self._recv_stats_across_nodes(stats)

        self._log_step('validate', epoch, batch_idx, self.data_loader.validation_set_size, stats)

        return stats

    def _test_epoch(self, epoch: int) -> dict:
        self.model.eval()
        with torch.no_grad():
            p_bar = tqdm(
                    desc=f'Rank {self.rank}, Test batch: ',
                    total=self.n_test_batches,
                    unit='batch',
                    ascii=True,
                    position=self.rank * 2 + 1,
                    leave=False)

            epoch_stats = None
            for batch_idx, data in enumerate(self.data_loader.test):
                data = data[0].to(self.device).float()
                batch_stat = self._test_step(data, batch_idx, epoch, p_bar)
                epoch_stats = self._update_stats(epoch_stats, batch_stat)
            p_bar.close()
            del p_bar
            self._normalize_stats(self.n_test_batches, epoch_stats)

            self._log_epoch('test/epoch/', epoch_stats)

        return epoch_stats

    def _test_step(self, minibatch: dict, batch_idx: int, epoch: int, p_bar: tqdm) -> dict:
        stats = self.model.validate_step(minibatch)
        self._update_step_p_bar(p_bar, stats)
        stats = self._recv_stats_across_nodes(stats)
        self._log_step('test', epoch, batch_idx, self.data_loader.test_set_size, stats)

        return stats

    def _recv_stats_across_nodes(self, stats: dict) -> dict:
        if self.world_size != -1:
            stats = self._average_across_nodes(stats, self.world_size)
        else:
            stats = self.tensor_2_item(stats)
        return stats

    @staticmethod
    def _update_stats(epoch_stat: dict, batch_stat: dict) -> dict:
        if epoch_stat is None:
            return batch_stat.copy()
        for k, v in batch_stat.items():
            epoch_stat[k] += v

        return epoch_stat

    @staticmethod
    def _normalize_stats(n_batches: int, statistics: dict) -> dict:
        for k, v in statistics.items():
            if is_primitive(v):
                statistics[k] /= n_batches
        return statistics

    @staticmethod
    def _average_across_nodes(statistics: dict, world_size: int) -> dict:
        avg_stats = dict()
        for k, v in statistics.items():
            if not isinstance(v, tuple):
                dist.all_reduce(v, dist.ReduceOp.SUM)
                avg_stats[k] = v.item() / world_size
            else:
                avg_stats[k] = v
        return avg_stats

    def _log_epoch(self, log_label: str, statistics: dict) -> None:
        if not self.is_rank_0:
            return None

        for k, v in statistics.items():
            if is_primitive(v):
                self.summary.add_scalar(log_label + k, v, self.global_step)
            elif isinstance(v, list) and isinstance(v[0], int):
                self.summary.add_histogram(log_label + k, v, self.global_step)
            elif isinstance(v, matplotlib.figure.Figure):
                self.summary.add_figure(log_label + k, figure=v, global_step=self.global_step)

    def _prepare_dirs(self) -> None:
        trainer_par = self.params['trainer']
        start_time = datetime.datetime.now().strftime('%d%m_%H%M%S')
        name = self.params['name']
        if len(name) > 200:
            name = "_".join([i if i.isdigit() else i[0:3] for i in name.split('_')])
        self.checkpoint_dir = os.path.join(trainer_par['save_dir'], name, start_time)
        self.logging_dir = os.path.join(trainer_par['logging']['logging_dir'], name, start_time)
        self.tensorboard_dir = os.path.join(trainer_par['logging']['tensorboard_dir'], name, start_time)

        os.makedirs(self.logging_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

    def _save_params(self):
        params_path = os.path.join(self.logging_dir, 'config.yaml')
        self.logger.info(f'saving config into {params_path}')
        yaml.dump(self.params, open(params_path, 'w'), default_flow_style=False)

    def _save_model(self, file_name: str, **kwargs) -> None:
        model_type = type(self.model).__name__
        state = {
            'model_type': model_type,
            'epoch': kwargs.get('epoch'),
            'model_state': self.model.state_dict(),
            'params': self.params
        }
        for key in self.optimizer:
            state[key] = self.optimizer[key]['opt'].state_dict()

        torch.save(state, file_name)

    def _save_model_parameters(self, file_name):
        """
        Args:
            file_name:
        """
        with open(file_name, 'w') as f:
            json.dump(self.params, f, indent=4)

    def _save_check_point(self, epoch: int) -> None:
        """

        :param epoch:
        :returns:
        :rtype:^^

        """

        file_name = os.path.join(self.checkpoint_dir,
                                 'checkpoint-epoch{}.pth'.format(epoch))
        self.t_logger.info('Saving checkpoint: {} ...'.format(file_name))
        self._save_model(file_name, epoch=epoch)

    def _save_best_model(self) -> None:
        file_name = os.path.join(self.checkpoint_dir,
                                 'best_model.pth')
        self.t_logger.info('Saving best model ...')
        self._save_model(file_name)

    def _resume_check_point(self, path: str, resume_ignore: list) -> None:
        """

        :param path:
        :returns:
        :rtype:

        """
        self.logger.info('Loading checkpoint: {} ...'.format(path))
        if torch.cuda.is_available() is False:
            state = torch.load(path, map_location='cpu')
        else:
            state = torch.load(path)
        self.params = state['params']
        if state['epoch'] is None:
            self.start_epoch = 1
        else:
            self.start_epoch = state['epoch'] + 1

        state_dict_keys = list(state['model_state'].keys())
        self.logger.info('Not loading states starting with: {}'.format(resume_ignore))
        for ignore in resume_ignore:
            for key in state_dict_keys:
                if key.startswith(ignore):
                    self.logger.info('Key removed from state dict: {}'.format(key))
                    state['model_state'].pop(key)

        self.model.load_state_dict(state['model_state'], strict=False)

        if len(resume_ignore) == 0:
            for key in self.optimizer:
                try: self.optimizer[key]['opt'].load_state_dict(state[key])
                except Exception: print("Optimizer {} could not be loaded from checkpoint.".format(key))
        self.logger.info('Finished loading checkpoint: {} ...'.format(path))

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('train_logger')
        logger.propagate = False
        logger.setLevel(logging.INFO)

        file_name = os.path.join(self.logging_dir, 'train.log')
        fh = logging.FileHandler(file_name)
        formatter = logging.Formatter(
                self.params['trainer']['logging']['formatters']['simple'])
        fh.setLevel(logging.INFO)

        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def _log_step(self, step_type: str, epoch: int, batch_idx: int, data_len: int, stats: dict) -> None:
        if not self.is_rank_0:
            return None
        log = self._build_raw_log_str(f'{step_type} epoch', batch_idx, epoch, stats, float(data_len), self.batch_size)
        self.t_logger.info(log)
        for k, v in stats.items():
            if is_primitive(v):
                self.summary.add_scalar(f'{step_type}/batch/' + k, v, self.global_step)

    @staticmethod
    def _build_raw_log_str(prefix: str, batch_idx: int, epoch: int, logs: dict, data_len: float, batch_size: int):
        sb = prefix + ': {} [{}/{} ({:.0%})]'.format(
                epoch,
                batch_idx * batch_size,
                data_len,
                100.0 * batch_idx / data_len)
        for k, v in logs.items():
            if is_primitive(v):
                sb += ' {}: {:.6f}'.format(k, v)
        return sb

    def _check_and_save_best_model(self, train_log: dict, validate_log: dict) -> None:
        if validate_log[self.bm_metric] < self.best_model['val_metric']:
            self._save_best_model()
            self._update_best_model_flag(train_log, validate_log)

    def _update_p_bar(self, e_bar: tqdm, train_log: dict, validate_log: dict, test_log: dict) -> None:
        e_bar.set_postfix_str(
                f"train loss: {train_log[self.bm_metric]:4.4g} train {self.bm_metric}: {train_log[self.bm_metric]:4.4g}, "
                f"validation loss: {validate_log[self.bm_metric]:4.4g}, validation {self.bm_metric}: {validate_log[self.bm_metric]:4.4g} "
                f"test loss: {test_log[self.bm_metric]:4.4g}, test {self.bm_metric}: {test_log[self.bm_metric]:4.4g}")
        e_bar.update()

    @staticmethod
    def _update_step_p_bar(p_bar: tqdm, stats: dict):
        log_str = ''
        for key, value in stats.items():
            if isinstance(value, tuple):
                continue
            log_str += f"{key}: {value.item():4.6g} "

        p_bar.update()
        p_bar.set_postfix_str(log_str)

    def _update_best_model_flag(self, train_log: dict, validate_log: dict) -> None:
        self.best_model['train_loss'] = train_log[self.bm_metric]
        self.best_model['val_loss'] = validate_log[self.bm_metric]
        self.best_model['train_metric'] = train_log[self.bm_metric]
        self.best_model['val_metric'] = validate_log[self.bm_metric]
        self.best_model['name'] = self.params['name']

    @staticmethod
    def tensor_2_item(stats):
        for key, value in stats.items():
            if type(value) is torch.Tensor:
                stats[key] = value.item()
        return stats

    def generate(self, n: int) -> List[Any]:
        return self.model.generate(n=n)

class LotkaVolterraTrainer(BaseTrainingProcedure):
    
    def __init__(self, model: torch.nn.Module, optimizer: dict, distributed: bool, resume: bool, resume_ignore: list,  params: dict, data_loader: ADataLoader, train_logger=None,
                 **kwargs):
        super(LotkaVolterraTrainer, self).__init__(model, optimizer, distributed, resume, resume_ignore, params, data_loader,
                                                   train_logger=train_logger,
                                                   **kwargs)

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        p_bar = tqdm(
                desc=f'Rank {self.rank}, Training batch: ',
                total=self.n_train_batches,
                unit='batch',
                leave=False,
                ascii=True,
                position=self.rank * 2 + 1
        )
        epoch_stats = None
        for batch_idx, data in enumerate(self.data_loader.train):
            batch_stats = self._train_step(data, batch_idx, epoch, p_bar)
            epoch_stats = self._update_stats(epoch_stats, batch_stats)
        p_bar.close()
        del p_bar
        epoch_stats = self._normalize_stats(self.n_train_batches, epoch_stats)
        self._log_epoch('train/epoch/', epoch_stats)

        return epoch_stats

    def _validate_epoch(self, epoch: int) -> dict:
        self.model.eval()
        with torch.no_grad():
            p_bar = tqdm(
                    desc=f"Rank {self.rank}, Validation batch: ",
                    total=self.n_validate_batches,
                    unit="batch",
                    leave=False,
                    ascii=True,
                    position=self.rank * 2 + 1)

            epoch_stats = None
            for batch_idx, data in enumerate(self.data_loader.validate):
                batch_stats = self._validate_step(data, batch_idx, epoch, p_bar)
                epoch_stats = self._update_stats(epoch_stats, batch_stats)
            p_bar.close()
            del p_bar
            epoch_stats = self._normalize_stats(self.n_validate_batches, epoch_stats)
            self._log_epoch('validate/epoch/', epoch_stats)

            return epoch_stats


    def _test_epoch(self, epoch: int) -> dict:
        self.model.eval()
        with torch.no_grad():
            p_bar = tqdm(
                    desc=f'Rank {self.rank}, Test batch: ',
                    total=self.n_test_batches,
                    unit='batch',
                    ascii=True,
                    position=self.rank * 2 + 1,
                    leave=False)

            epoch_stats = None
            for batch_idx, data in enumerate(self.data_loader.test):
                batch_stat = self._test_step(data, batch_idx, epoch, p_bar)
                epoch_stats = self._update_stats(epoch_stats, batch_stat)
            p_bar.close()
            del p_bar
            self._normalize_stats(self.n_test_batches, epoch_stats)

            self._log_epoch('test/epoch/', epoch_stats)

        return epoch_stats
    