# -*- encoding: utf-8 -*-

from __future__ import annotations
import os
from src.utils.saver import Saver
import logging
import datetime
from src.utils.metrics import Evaluator
from src.data import prefetcher
from src.utils.summaries import TensorboardSummary
import torch
from torch.cuda.streams import Stream

class Trainer():
    """An object for training and validation
    """
    def __init__(self, args, loader : dict, model) -> None:
        """Init an trainer Object.

        Args:
            args (NameSpace): a namespace contains the parameters set in the experiment.
            loader (dict): a dict containing dataloader and total num for training and validation.
            model (BaseModel): the inherited BaseModel after initialization.
        """
        
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.args = args
        self.scale = args.scale
        if self.local_rank == 0 and args.isTrain:
            self.saver = Saver(self.args)
            self.saver.save_experiment_config()
            logging.basicConfig(filename=os.path.join(self.saver.experiment_dir, "train.log"),
                                filemode="w", level=logging.INFO, format="%(levelname)s:%(asctime)s:%(message)s") 
            self.writer = self.summary.create_summart() 
        self.loader_train = loader['loader_train']
        self.loader_test = loader['loader_test']
        self.num_train_image = loader["num_train"]
        self.num_val_image = loader["num_val"]
        self.model = model
        self.current_epoch = 0
        self.best_pred = 1e6
        self.train_iters_epoch = len(self.loader_train)
        self.val_iters_epoch = len(self.loader_test)
        self.val_evaluator = Evaluator(
            self.args.test_batch_size, self.args.rgb_range, self.args.resolution)
        self.learning_rate = self.args.lr
        self.stream = Stream()


    def train(self, epoch : int) -> None:
        """training process.

        Args:
            epoch (int): current epoch of training process
        """
        train_loss_g = 0.0
        train_loss_d = 0.0

        for iter_per, (hr, lr, feats) in enumerate(self.loader_train):
            with torch.cuda.stream(self.stream):  # type: ignore
                hr = hr.cuda(non_blocking=True)
                lr = lr.cuda(non_blocking=True)
                if len(feats):
                    feats = [feat.cuda(non_blocking=True) for feat in feats]
            self.model.set_input(lr, hr, feats)
            self.model.optimize_parameters(epoch)

            if self.local_rank == 0:
                train_loss_g += self.model.loss_G.item()
                train_loss_d += self.model.loss_D.item()

                global_step = iter_per + self.train_iters_epoch * epoch + 1

                if global_step % 50 == 0:
                    self.writer.add_scalar(
                        "train/train_loss_D", train_loss_g/(iter_per + 1), global_step)
                    self.writer.add_scalar(
                        "train/train_loss_G", train_loss_d/(iter_per + 1), global_step)
        if self.local_rank == 0:
            self.writer.add_scalar("train/learning_rate",
                                   self.learning_rate, epoch)
            self.writer.add_scalar(
                "train/loss_epoch_D", train_loss_d/self.train_iters_epoch, epoch)
            self.writer.add_scalar(
                "train/loss_epoch_G", train_loss_d/self.train_iters_epoch, epoch)
            print("Train:")
            print(f'[Epoch: {epoch}, numImages: {self.num_train_image}]')
            print(f"learning_rate: {self.learning_rate}")
            print('Loss of Discriminator: %.3f    Loss of Generator:%.3f' % (
                train_loss_d/self.train_iters_epoch, train_loss_g/self.train_iters_epoch))
            params_num = self.model.get_params_num()
            print("Params: %.2fM" % (params_num / 1e6))
            logging.info("Train:")
            logging.info('[Epoch: %d, numImages: %5d]' %
                         (epoch, self.num_train_image))
            logging.info(f"learning_rate: {self.learning_rate}")
            logging.info('Loss of Discriminator: %.3f    Loss of Generator:%.3f' % (
                train_loss_d/self.train_iters_epoch, train_loss_g/self.train_iters_epoch))
            logging.info("Params: %.2fM" % (params_num / 1e6))
        self.learning_rate = self.model.update_learning_rate()
        

    def validation(self, epoch : int = 0) -> None:
        """validation process.

        Args:
            epoch (int, optional): training process. Defaults to 0.
        """
        self.val_evaluator.reset()
        self.model.eval() 

        for hr, lr, feats in self.loader_test:
            with torch.cuda.stream(self.stream):  
                hr = hr.cuda(non_blocking=True) 
                lr = lr.cuda(non_blocking=True) 
                if len(feats):
                    feats = [feat.cuda(non_blocking=True) for feat in feats]
            with torch.no_grad():
                self.model.set_input(lr, hr, feats)
                self.model.forward(mode="val")
                output = self.model.output
            self.val_evaluator.add_batch(sr=output, hr=hr)

        mse, mae, rmse, e_max,  psnr, slope_mae = self.val_evaluator.score(
            self.num_val_image)
        print("Validation:")
        print('[Epoch: %d, numImages: %5d]' % (epoch, self.num_val_image))
        print("MSE:{}, MAE:{}, RMSE:{}, E_MAX:{},  PSNR:{}, Slope_MAE:{}".format(
            mse, mae, rmse, e_max,  psnr, slope_mae))
        if self.args.isTrain:
            logging.info("Validation:")
            logging.info('[Epoch: %d, numImages: %5d]' % (epoch, self.num_val_image))
            logging.info("MSE:{}, MAE:{}, RMSE:{}, E_MAX:{},  PSNR:{}, Slope_MAE:{}".format(mse, mae, rmse, e_max,  psnr, slope_mae))
            self.writer.add_scalar("val/MSE", mse, epoch)
            self.writer.add_scalar("val/PSNR", psnr, epoch)
            self.writer.add_scalar("val/MAE", mae, epoch)
            self.writer.add_scalar("val/RMSE", rmse, epoch)
            self.writer.add_scalar("val/E_MAX", e_max, epoch)
            self.writer.add_scalar("val/Slope_MAE", slope_mae, epoch)
            new_pred = mae
            if new_pred < self.best_pred:
                is_best = True
                self.best_pred = new_pred
                metric_dict = {
                    "MSE": mse,
                    "MAE": mae,
                    "RMSE": rmse,
                    "E_MAX": e_max,
                    "Slope_MAE": slope_mae,
                }
                self.model.save_networks(epoch, self.saver.experiment_dir)
                self.saver.save_checkpoint({'epoch': epoch + 1,
                                            'optimizers': self.model.optimizers,
                                            'best_pred': self.best_pred}, is_best=is_best, results=metric_dict)
