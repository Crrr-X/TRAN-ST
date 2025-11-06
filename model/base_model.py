# -*- encoding: utf-8 -*-

from __future__ import annotations
import os
import torch
import glob
from abc import ABC, abstractmethod
from src.model.common import *
from torch.optim import lr_scheduler


class BaseModel(ABC): 
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <>:           calculate losses, gradients, and update network weights.
    """

    def __init__(self, args) -> None:
        self.params_num = 0
        self.args = args
        self.optimizers: list[torch.optim.Adam] = [] 
        self.model_names = []
        self.schedulers = []
        self.isTrain = args.isTrain
        self.gpu_ids = args.gpu_ids
        self.loss_G, self.loss_D = torch.Tensor([0.0]), torch.Tensor([0.0])

    def initialization(self) -> None:
        """initialize optimizers and the initial weights of network"""
        for model in self.model_names:
            net = getattr(self, model)
            weight_init(net)

    @abstractmethod
    def set_input(self, lr, hr) -> None:
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self, mode: str = "train") -> None:
        """Run forward pass; called by both functions <> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self) -> None:
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def set_requires_grad(self, nets: list[nn.Module], requires_grad=False) -> None:
        """Set requires_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def eval(self):
        """Make models eval mode during test time
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def update_learning_rate(self) -> float:
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.optimizers[0].param_groups[0]['lr'] > 5e-6:
                scheduler.step()
            else:
                continue

        return self.optimizers[0].param_groups[0]['lr']

    def save_networks(self, epoch, save_dir) -> None:
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name 'net_%s.pth' % (name)
        """
        for name in self.model_names:
            save_filename = 'net_%s.pth' % (name)
            save_path = os.path.join(save_dir, save_filename)
            net = getattr(self, name)

            torch.save(net.state_dict(), save_path)
    
    def load_networks(self, net, path) -> None:
        """init weight.

        Args:
            net (nn.Module): module of network to be initialized.
            path : pretrained weights of the module.
        """
        if isinstance(net, torch.nn.DataParallel):  # type: ignore
            net = net.module
        state_dict = torch.load(path, map_location="cpu")
        net.load_state_dict(state_dict)
    
        
    def get_params_num(self) -> int:
        """get total num of parameters of this network.

        Returns:
            int: params_num of this network.
        """
        self.params_num, params_num = 0, 0
        for name in self.model_names:
            net = getattr(self, name)
            params_num = sum(p.numel() for p in net.parameters())
            self.params_num += params_num
        return self.params_num

    def get_scheduler(self, optimizer, args):
        if args.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(
                optimizer, step_size=100, gamma=0.5)
        elif args.lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - 100) / float(100 + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif args.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif args.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.eta_min)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
        return scheduler
