import math
import os
# import tensorboardX
from tqdm import tqdm

import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import transformers
from transformers import AdamW
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from cser import models
from cser import sampling
from cser import util
from cser.dataset import Dataset
from cser.evaluator import Evaluator
from cser.loss import Loss
from cser.reader import JsonReader


class BaseTrainer:
    """ Trainer base class with common methods """
    def __init__(self, cfg, logger):
        self._cfg = cfg

        # Logger
        self._logger = logger

        # Model saving
        self._label = self._logger.label
        self._timestamp = self._logger.timestamp
        self._log_path = self._logger.log_path

        if self._cfg.has_option('logger', 'save_path'):
            self._save_path = os.path.join(
                self._cfg.get('logger', 'save_path'),
                self._label, self._timestamp)
            util.create_directorie(self._save_path)

        # CUDA devices
        self._gpu = self._cfg.getint('model', 'gpu')
        self._cpu = self._cfg.getboolean('model', 'cpu')
        self._device = torch.device(
            'cuda:' + str(self._gpu) if torch.cuda.is_available()
            and not self._cpu else 'cpu')
        self._gpu_count = torch.cuda.device_count()

        # tensorboard summary
        # self._summary_writer = tensorboardX.SummaryWriter(self._log_path)
        self._log_configs()

    def _log_configs(self):
        util.save_config(self._log_path, self._cfg, 'config')

    def _save_model(
        self, save_path: str, model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer, iteration: int,
        save_as_best: bool = False, extra: dict = None,
        include_iteration: int = True, name: str = 'model'): # noqa

        extra_state = dict(iteration=iteration)

        if extra:
            extra_state.update(extra)

        if save_as_best:
            dir_path = os.path.join(save_path, '%s_best' % name)
        else:
            dir_name = '%s_%s' % (name, iteration) if include_iteration else name
            dir_path = os.path.join(save_path, dir_name)

        util.create_directorie(dir_path)

        # save model
        if isinstance(model, DataParallel):
            model.module.save_pretrained(dir_path)
        else:
            model.save_pretrained(dir_path)

        # save vocabulary
        tokenizer.save_pretrained(dir_path)

        # save extra
        state_path = os.path.join(dir_path, 'extra.state')
        torch.save(extra_state, state_path)

    def _get_lr(self, optimizer):
        lrs = []
        for group in optimizer.param_groups:
            lr_scheduled = group['lr']
            lrs.append(lr_scheduled)
        return lrs


class SpanTrainer(BaseTrainer):
    """ Joint entity extraction training and evaluation """
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)

        # byte-pair encoding
        self._tokenizer_path = self._cfg.get(
            'preprocessing', 'tokenizer_path')
        self._lowercase = self._cfg.getboolean(
            'preprocessing', 'lowercase')
        self._tokenizer = BertTokenizer.from_pretrained(
            self._tokenizer_path, do_lower_case=self._lowercase)

        # Input reader
        self._types_path = self._cfg.get(
            'input', 'types_path')
        self._neg_entity_count = None if not self._cfg.has_option(
            'model', 'neg_entity_count') else self._cfg.getint(
            'model', 'neg_entity_count')
        self._max_span_size = self._cfg.getint(
            'preprocessing', 'max_span_size')
        self._reader = JsonReader(
            self._types_path,
            self._tokenizer,
            neg_entity_count=self._neg_entity_count,
            max_span_size=self._max_span_size)
        self._log_reader()

        # Create model
        self._model_type = self._cfg.get('model', 'model_type')
        model_class = models.get_model(self._model_type)

        self._model_path = self._cfg.get('model', 'model_path')
        config = BertConfig.from_pretrained(self._model_path)

        self._dropout = self._cfg.getfloat(
            'model', 'dropout')
        self._size_embedding = self._cfg.getint(
            'model', 'size_embedding')
        self._freeze_transformer = self._cfg.getboolean(
            'model', 'freeze_transformer')
        self._model = model_class.from_pretrained(
            self._model_path,
            config=config,
            # Span model parameters
            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
            entity_types=self._reader.entity_type_count,
            prop_drop=self._dropout,
            size_embedding=self._size_embedding,
            freeze_transformer=self._freeze_transformer)


        # If you want to predict Spans on multiple GPUs, uncomment the following lines
        # # parallelize model
        # self._model = self._model.cuda()
        # if not self._cpu and self._gpu_count > 1:
        self._model = torch.nn.DataParallel(self._model, device_ids=[2, 3])
        self._model.to(self._device)

        # path to export predictions to
        self._jpredictions_path = os.path.join(
            self._log_path, 'predictions_%s_epoch_%s.json')

        # path to export entity extraction examples to
        self._hpredictions_path = os.path.join(
            self._log_path, 'predictions_%s_%s_epoch_%s.html')

    def train(self, train_path: str, valid_path: str):
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % self._model_type)

        # Read datasets
        self._reader.read(train_label, train_path, Dataset.TRAIN_MODE)
        self._reader.read(valid_label, valid_path, Dataset.EVAL_MODE)
        # self._log_dataset()

        self._epochs = self._cfg.getint('model', 'epochs')
        self._train_batch_size = self._cfg.getint('model', 'train_batch_size')
        train_dataset = self._reader.get_dataset(train_label)
        train_doc_count = train_dataset.did
        updates_epoch = train_doc_count // self._train_batch_size
        updates_total = updates_epoch * self._epochs

        valid_dataset = self._reader.get_dataset(valid_label)

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # Create optimizer
        self._lr = self._cfg.getfloat('model', 'lr')
        self._weight_decay = self._cfg.getfloat('model', 'weight_decay')
        optimizer_params = self._get_optimizer_params(self._model)
        optimizer = AdamW(
            optimizer_params, lr=self._lr,
            weight_decay=self._weight_decay, correct_bias=False)

        # Create scheduler
        self._lr_warmup = self._cfg.getfloat('model', 'lr_warmup')
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self._lr_warmup * updates_total,
            num_training_steps=updates_total)

        # Create loss function
        self._max_grad_norm = self._cfg.getfloat('model', 'max_grad_norm')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        compute_loss = Loss(
            entity_criterion, self._model, optimizer,
            scheduler, self._max_grad_norm)

        # Eval validation set
        self._init_eval = self._cfg.getboolean('logger', 'init_eval')
        if self._init_eval:
            self._eval(
                self._model, valid_dataset,
                self._reader, 0, updates_epoch)

        # Train
        self._final_eval = self._cfg.getboolean('logger', 'final_eval')
        for epoch in range(self._epochs):
            # Train epoch
            self._train(
                self._model, compute_loss, optimizer,
                train_dataset, updates_epoch, epoch)

            # Eval validation sets
            if not self._final_eval or (epoch == self._epochs - 1):
                self._eval(
                    self._model, valid_dataset, self._reader,
                    epoch + 1, updates_epoch)

        # save final model
        extra = dict(
            epoch=self._epochs, updates_epoch=updates_epoch,
            epoch_iteration=0)
        global_iteration = self._epochs * updates_epoch
        self._save_model(
            self._save_path, self._model, self._tokenizer,
            global_iteration, extra=extra,
            include_iteration=False, name='final_model')

        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)

    def _train(
        self, model: torch.nn.Module, compute_loss: Loss,
        optimizer: Optimizer, dataset: Dataset,
        updates_epoch: int, epoch: int): # noqa

        self._logger.info("Train epoch: %s" % epoch)

        # Create data loader
        self._sampling_processes = self._cfg.getint(
            'preprocessing', 'sampling_processes')
        data_loader = DataLoader(
            dataset,
            batch_size=self._train_batch_size,
            shuffle=True,
            num_workers=self._sampling_processes,
            collate_fn=sampling.collate_fn_padding)

        model.zero_grad()

        iteration = 0
        total = dataset.did // self._train_batch_size
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = util.to_device(batch, self._device)

            # Forward step
            entity_logits = model(
                encodings=batch['encodings'],
                context_masks=batch['context_masks'],
                entity_masks=batch['entity_masks'],
                entity_sizes=batch['entity_sizes'])

            # Compute loss and optimize parameters
            # batch_loss = compute_loss.compute(
            compute_loss.compute(
                entity_logits=entity_logits,
                entity_types=batch['entity_types'],
                entity_sample_masks=batch['entity_sample_masks'])

            iteration += 1
            # global_iteration = epoch * updates_epoch + iteration

            # if global_iteration % self.args.train_log_iter == 0:
            #     self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def eval(self, test_path: str):
        test_label = 'test'

        self._logger.info("Datasets: %s" % test_path)
        self._logger.info("Model: %s" % self._model_type)

        # Read datasets
        self._reader.read(test_label, test_path, Dataset.EVAL_MODE)
        # self._log_dataset()

        # evaluate
        test_dataset = self._reader.get_dataset(test_label)
        self._eval(self._model, test_dataset, self._reader)

        self._logger.info("Logged in: %s" % self._log_path)

    def _eval(
        self, model: torch.nn.Module,
        dataset: Dataset, epoch: int = 0,
        updates_epoch: int = 0, iteration: int = 0): # noqa

        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        self._overlapping = self._cfg.getboolean('model', 'overlapping')
        evaluator = Evaluator(
            dataset, self._reader, self._tokenizer,
            self._overlapping, self._jpredictions_path,
            self._hpredictions_path, epoch, dataset.label)

        # create data loader
        self._eval_batch_size = self._cfg.getint(
            'model', 'eval_batch_size')
        self._sampling_processes = self._cfg.getint(
            'preprocessing', 'sampling_processes')
        data_loader = DataLoader(
            dataset,
            batch_size=self._eval_batch_size,
            shuffle=False,
            num_workers=self._sampling_processes,
            collate_fn=sampling.collate_fn_padding)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.did / self._eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % 0):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(
                    encodings=batch['encodings'],
                    context_masks=batch['context_masks'],
                    entity_masks=batch['entity_masks'],
                    entity_sizes=batch['entity_sizes'],
                    train=False)
                entity_clf = result

                # evaluate batch
                evaluator.eval_batch(entity_clf, batch)

        evaluator.compute_scores()

        self._store_predictions = self._cfg.getboolean(
            'logger', 'store_predictions')
        if self._store_predictions:
            evaluator.store_predictions()

        self._store_examples = self._cfg.getboolean(
            'logger', 'store_examples')
        self._template_path = self._cfg.get(
            'template', 'template_path')
        if self._store_examples:
            evaluator.store_examples(self._template_path)

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': self._weight_decay
        }, {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }]

        return optimizer_params

    def _log_reader(self):
        self._logger.info("Entity type count: %s" % self._reader.entity_type_count)

        self._logger.info("Entities:")
        for e in self._reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        for k, d in self._reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.did)
            self._logger.info("Entity count: %s" % d.eid)

        self._logger.info("Context size: %s" % self._reader.context_size)

    def _log_dataset(self):
        self._logger.info("Entity type count: %s" % self._reader.entity_type_count)

        self._logger.info("Entities:")
        for e in self._reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        for k, d in self._reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.did)
            self._logger.info("Entity count: %s" % d.eid)

        self._logger.info("Context size: %s" % self._reader.context_size)
