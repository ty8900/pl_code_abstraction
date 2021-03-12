from codes.utils.args import TemplateParser
from codes.trainer import trainer_factory

import sys
from dotmap import DotMap
from typing import List


def main(sys_argv: List[str] = None):
    if sys_argv is None:
        sys_argv = sys.argv[1:]
    conf = TemplateParser(sys_argv).parse()
    args = DotMap(conf, _dynamic=False)
    trainer = trainer_factory(args)
    trainer.run()
