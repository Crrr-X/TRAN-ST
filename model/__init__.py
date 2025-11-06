from importlib import import_module
from src.model.base_model import BaseModel
import torch.nn as nn


def create_model(args) -> BaseModel:
    print('making model……')
    module_lib = import_module('src.model.' + args.model_type)
    # target_model_name = args.model_name.replace("_", "") if "_" in  args.model_name else args.model_name 
    model = None
    for name,cls in module_lib.__dict__.items():
        if name.lower() == args.model_type and issubclass(cls, BaseModel):
           model = cls
    return model(args=args) # type: ignore