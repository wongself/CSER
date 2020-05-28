import math
import os
import sys
import tensorboardX
from tqdm import tqdm
from typing import List, Dict, Tuple

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

# from ner.model import models
# from ner.model import sampling
from cser import util
# from ner.model.entity import Dataset
# from ner.model.evaluator import Evaluator
# from ner.model.reader import JsonInputReader


class BaseTrainer:
    """ Trainer base class with common methods """
    def __init__(self, cfg, logger, timestamp):
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
        tokenizer: PreTrainedTokenizer,
        iteration: int, optimizer: Optimizer = None,
        save_as_best: bool = False,
        extra: dict = None, include_iteration: int = True,
        name: str = 'model'): # noqa

        extra_state = dict(iteration=iteration)

        if optimizer:
            extra_state['optimizer'] = optimizer.state_dict()

        if extra:
            extra_state.update(extra)

        if save_as_best:
            dir_path = os.path.join(save_path, '%s_best' % name)
        else:
            dir_name = '%s_%s' % (name, iteration) if include_iteration else name
            dir_path = os.path.join(save_path, dir_name)

        util.create_directories_dir(dir_path)

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

        # path to export predictions to
        self._jpredictions_path = os.path.join(
            self._log_path, 'predictions_%s_epoch_%s.json')

        # path to export entity extraction examples to
        self._hpredictions_path = os.path.join(
            self._log_path, 'predictions_%s_%s_epoch_%s.html')

        # Input reader
        self._input_reader = JsonInputReader(
            self._types_path,
            self._tokenizer,
            max_span_size=self._max_span_size,
            logger=self._logger)

        # Create model
        model_class = models.get_model(self._model_type)

        config = BertConfig.from_pretrained(self._model_path)

        self._model = model_class.from_pretrained(
            self._model_path,
            config=config,
            # Span model parameters
            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
            entity_types=self._input_reader.entity_type_count,
            max_pairs=self._max_pairs,
            prop_drop=self._prop_drop,
            size_embedding=self._size_embedding,
            freeze_transformer=self._freeze_transformer)

        # If you want to predict Spans on multiple GPUs, uncomment the following lines
        # # parallelize model
        # if self._device.type != 'cpu' and self._gpu_count > 1:
        #     self._model = torch.nn.DataParallel(self._model, device_ids=[0,])
        self._model.to(self._device)

        # path to export predictions to
        self._predictions_path = os.path.join(
            self._log_path, 'predictions_%s_epoch_%s.json')

    def eval(self, jdoc: list):
        dataset_label = 'prediction'

        self._logger.info("Model: %s" % self._model_type)

        # Read datasets
        self._input_reader.read({dataset_label: jdoc})
        self._log_datasets()

        # evaluate
        jpredictions = self._eval(
            self._model,
            self._input_reader.get_dataset(dataset_label),
            self._input_reader)

        self._logger.info("Logged in: %s" % self._log_path)

        return jpredictions

    def _eval(
        self, model: torch.nn.Module,
        dataset: Dataset, epoch: int = 0,
        updates_epoch: int = 0, iteration: int = 0): # noqa

        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        evaluator = Evaluator(
            dataset, self._input_reader, self._tokenizer,
            self._no_overlapping, self._predictions_path,
            epoch, dataset.label)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(
            dataset,
            batch_size=self._eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self._sampling_processes,
            collate_fn=sampling.collate_fn_padding)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self._eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(
                    encodings=batch['encodings'],
                    context_masks=batch['context_masks'],
                    entity_masks=batch['entity_masks'],
                    entity_sizes=batch['entity_sizes'],
                    entity_spans=batch['entity_spans'],
                    entity_sample_masks=batch['entity_sample_masks'])
                entity_clf = result

                # evaluate batch
                evaluator.eval_batch(entity_clf, batch)

        jpredictions = evaluator.store_predictions()

        return jpredictions

    def _log_datasets(self):
        self._logger.info("Entity type count: %s" % self._input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in self._input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        for k, d in self._input_reader.datasets.items():
            self._logger.info('Document: %s' % k)
            self._logger.info("Sentences count: %s" % d.document_count)
            # self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % self._input_reader.context_size)
