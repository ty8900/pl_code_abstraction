from codes.datamodule import DATAMODULE
from codes.dataset import DATASET
from codes.model import MODEL
from codes.trainer import TRAINER

import argparse
import yaml


class TemplateParser:
    def __init__(self, sys_argv):
        self.sys_argv = sys_argv
        
    def parse(self):
        conf = {}
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--templates', type=str)
        args = parser.parse_known_args(self.sys_argv)[0]
        conf.update(vars(args))
        
        conf = self.set_template(conf)
        return conf

    @staticmethod
    def set_template(conf):
        template_name = conf['templates']
        if template_name is None:
            raise NameError
        yaml_path = f'templates/{template_name}.yaml'
        template = yaml.safe_load(open(yaml_path))
        return template



"""

    def parse_dataset(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--templates', type=str)
        parser.add_argument('--dataset_name', type=str, choices=DATASET.keys(), help='Select dataset')
        parser.add_argument('--force_update', type=bool, default=False, help='If true, force update dataset pkl')
        parser.add_argument('--dataset_path', type=str, default='./datasets', help='dataset path')

        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_datamodule(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--datamodule_name', type=str, choices=DATAMODULE.keys(), help='Select datamodule')
        parser.add_argument('--train_batch_size', type=int, help='training batch size')
        parser.add_argument('--val_batch_size', type=int, help='validation batch size')
        parser.add_argument('--test_batch_size', type=int, help='test batch size')

        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_model(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--model_name', type=str, choices=MODEL.keys(), help='Select model')
        # parser.add_argument('--model_init_seed', type=int, help='seed for initialize model param')
        # parser.add_argument('--model_init_range', type=int, help='range for initialize model param')
        parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], help='optimizer type')
        parser.add_argument('--lr', type=int, default=1e-3, help='set learning rate')

        parser.add_argument('--network_name', type=str, help='network name_ex) resnet, efficientnet')
        parser.add_argument('--hidden_size', type=int, help='hidden_size')
        parser.add_argument('--num_attr', type=int, help='output attr size')
        parser.add_argument('--loss_type', type=str, choices=['bce', 'asl'], help='set loss function')

        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)

    def parse_trainer(self):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--trainer_name', type=str, choices=TRAINER.keys(), help='Select trainer')
        parser.add_argument('--train_from_ckpt', type=bool, help='if True, train from checkpoint below')
        parser.add_argument('--pretrained_model_path', type=str, help='checkpoint path')
        parser.add_argument('--gpus', type=int, help='amount of gpu to use')
        parser.add_argument('--max_epochs', type=int, help='train max epochs')
        parser.add_argument('--bar_refresh_rate', type=int, help='logging bar refresh rate')

        args = parser.parse_known_args(self.sys_argv)[0]
        return vars(args)
"""

