from .base import AbstractDataset
from codes.utils.factory import all_subclasses, import_all_classes
import_all_classes(__file__, __name__, AbstractDataset)

DATASET = {c.code(): c
           for c in all_subclasses(AbstractDataset)
           if c.code() is not None}


def dataset_factory(args):
    dataset = DATASET[args.dataset_name]
    return dataset(args)
