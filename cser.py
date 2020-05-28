import argparse
from configparser import ConfigParser
import multiprocessing as mp

from cser.logger import Logger
from cser.trainer import SpanTrainer


def _train(cfg, logger):
    trainer = SpanTrainer(cfg, logger)
    trainer.train()


def _eval(cfg, logger):
    trainer = SpanTrainer(cfg, logger)
    trainer.eval()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()

    cfg = ConfigParser()
    if args.mode == 'train':
        configuration_path = 'configs/cser_train.conf'
        target = _train
    elif args.mode == 'eval':
        configuration_path = 'configs/cser_eval.conf'
        target = _eval
    else:
        raise Exception(
            "Mode not in ['train', 'eval'], e.g. 'python cser.py train ...'")
    cfg.read(configuration_path)

    logger = Logger(cfg)
    logger.info('Configuration Parsed: %s' % cfg.sections())

    ctx = mp.get_context('fork')
    p = ctx.Process(target=target, args=(cfg, logger, ))
    p.start()
    p.join()
