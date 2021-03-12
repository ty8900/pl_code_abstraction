import os
import sys
import pkgutil
import inspect
from importlib import import_module


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


def import_all_classes(_file, _name, _class):
    modules = get_all_modules(_file, _name)
    for m in modules:
        for i in dir(m):
            attr = getattr(m, i)
            if inspect.isclass(attr) and issubclass(attr, _class):
                setattr(sys.modules[_name], i, attr)


def get_all_modules(_file, _name):
    modules = []
    _dir = os.path.dirname(_file)
    for _, name, ispkg in pkgutil.iter_modules([_dir]):
        module = import_module('.' + name, package=_name)
        modules.append(module)
        if ispkg:
            modules.extend(get_all_modules(module.__file__, module.__name__))
    return modules
