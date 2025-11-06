import os
import torch
import itertools
from src.model.modules import *
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler
from src.loss import Loss
from .base_model import BaseModel

class Supervised(BaseModel):
    def __init__(self, args) -> None:
        BaseModel.__init__(self, args)
        self.args = args
        model = globals()[args.model_name]
        local_rank = int(os.environ["LOCAL_RANK"])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        device = torch.device("cuda", int(local_rank))
        model = DistributedDataParallel(model(scale=args.scale, mean=args.mean, std=args.std).to(device), device_ids=[local_rank], output_device=local_rank)

        setattr(self, args.model_name, model)
        self.model_names = [args.model_name]
        self.initialization(args)

    def set_input(self, lr, hr, feats):
        self.hr = hr
        self.lr = lr
        self.feats = feats 

    def forward(self, mode="train") -> None:
        self.output = getattr(self, self.args.model_name)(self.lr, self.feats)

    def initialization(self, args):
        super().initialization()
        if args.isTrain:
            self.optimizers.append(torch.optim.Adam(itertools.chain(
                getattr(self, self.args.model_name).parameters()), lr=args.lr, betas=[0.9, args.beta2], eps=args.epsilon))  # type: ignore
            self.loss = Loss(args.weight)
            self.schedulers = [self.get_scheduler(
                optimizer)for optimizer in self.optimizers]
        elif args.resume_SR:
            self.load_networks(getattr(self, self.args.model_name), args.resume_SR)

    def optimize_parameters(self, epoch) -> None:
        
        self.optimizers[0].zero_grad()
        self.forward()
        self.backward(epoch)
        self.optimizers[0].step()

    def backward(self, epoch) -> None:
        loss_total = self.loss.l1_loss(self.output, self.hr, self.feats, epoch)
        self.loss_G = loss_total
        self.loss_D = loss_total
        loss_total.backward()

    def get_scheduler(self, optimizer):
        return super().get_scheduler(optimizer, self.args)