from .base import AbstractModel
from codes.utils.factory import all_subclasses, import_all_classes
import_all_classes(__file__, __name__, AbstractModel)

MODEL = {c.code(): c
         for c in all_subclasses(AbstractModel)
         if c.code() is not None}


def model_factory(args):
    model = MODEL[args.model_name]
    return model(args)
