import json
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import utils, loss, meter, scheduler
from data.dataset import Dataset_cifar10, Dataset_cifar100, Dataset_imagenet
from model.teacher.resnet_cifar import (
    resnet_56_cifar10,
    resnet_110_cifar10,
    resnet_56_cifar100,
)
from model.student.resnet_sparse_cifar import (
    resnet_56_sparse_cifar10,
    resnet_110_sparse_cifar10,
    resnet_56_sparse_cifar100,
)
from model.teacher.ResNet import ResNet_18_imagenet, ResNet_50_imagenet
from model.student.ResNet_sparse import (
    ResNet_18_sparse_imagenet,
    ResNet_50_sparse_imagenet,
)
from model.teacher.vgg_bn import VGG_16_bn_cifar10
from model.student.vgg_bn_sparse import VGG_16_bn_sparse_cifar10
from model.teacher.DenseNet import DenseNet_40_cifar10
from model.student.DenseNet_sparse import (
    DenseNet_40_sparse_cifar10,
)
from model.teacher.GoogLeNet import GoogLeNet_cifar10
from model.student.GoogLeNet_sparse import GoogLeNet_sparse_cifar10
from model.teacher.MobileNetV2 import MobileNetV2_imagenet
from model.student.MobileNetV2_sparse import MobileNetV2_sparse_imagenet

from get_flops_and_params import get_flops_and_params


class Finetune:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_type = args.dataset_type
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch
        self.device = args.device
        self.seed = args.seed
        self.result_dir = args.result_dir
        self.finetune_train_batch_size = args.finetune_train_batch_size
        self.finetune_eval_batch_size = args.finetune_eval_batch_size
        self.finetune_student_ckpt_path = args.finetune_student_ckpt_path
        self.finetune_num_epochs = args.finetune_num_epochs
        self.finetune_lr = args.finetune_lr
        self.finetune_warmup_steps = args.finetune_warmup_steps
        self.finetune_warmup_start_lr = args.finetune_warmup_start_lr
        self.finetune_lr_decay_T_max = args.finetune_lr_decay_T_max
        self.finetune_lr_decay_eta_min = args.finetune_lr_decay_eta_min
        self.finetune_weight_decay = args.finetune_weight_decay
        self.finetune_resume = args.finetune_resume
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path

        self.start_epoch = 0
        self.best_prec1_after_finetune = 0

    def result_init(self):
        self.writer = SummaryWriter(self.result_dir)

        # log
        self.logger = utils.get_logger(
            os.path.join(self.result_dir, "finetune_logger.log"), "finetune_logger"
        )

        # config
        self.logger.info("finetune config:")
        self.logger.info(str(json.dumps(vars(self.args), indent=4)))
        utils.record_config(
            self.args, os.path.join(self.result_dir, "finetune_config.txt")
        )

        self.logger.info("--------- Finetune -----------")

    def setup_seed(self):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = True

    def dataload(self):
        dataset = eval("Dataset_" + self.dataset_type)(
            self.dataset_dir,
            self.finetune_train_batch_size,
            self.finetune_eval_batch_size,
            self.num_workers,
            self.pin_memory,
        )
        self.train_loader, self.val_loader = (
            dataset.loader_train,
            dataset.loader_test,
        )
        self.logger.info("Dataset has been loaded!")

    def build_model(self):
        self.logger.info("==> Building model..")

        self.logger.info("Loading student model")
        self.student = eval(self.arch + "_sparse_" + self.dataset_type)()
        ckpt_student = torch.load(self.finetune_student_ckpt_path, map_location="cpu")
        self.student.load_state_dict(ckpt_student["student"])
        self.best_prec1_before_finetune = ckpt_student["best_prec1"]

    def define_loss(self):
        self.ori_loss = nn.CrossEntropyLoss()

    def define_optim(self):
        weight_params = map(
            lambda a: a[1],
            filter(
                lambda p: p[1].requires_grad and "mask" not in p[0],
                self.student.named_parameters(),
            ),
        )
        self.finetune_optim_weight = torch.optim.Adamax(
            weight_params,
            lr=self.finetune_lr,
            weight_decay=self.finetune_weight_decay,
            eps=1e-7,
        )
        self.finetune_scheduler_student_weight = scheduler.CosineAnnealingLRWarmup(
            self.finetune_optim_weight,
            T_max=self.finetune_lr_decay_T_max,
            eta_min=self.finetune_lr_decay_eta_min,
            last_epoch=-1,
            warmup_steps=self.finetune_warmup_steps,
            warmup_start_lr=self.finetune_warmup_start_lr,
        )

    def resume_student_ckpt(self):
        ckpt_student = torch.load(self.finetune_resume)

        self.best_prec1_after_finetune = ckpt_student["best_prec1_after_finetune"]
        self.start_epoch = ckpt_student["start_epoch"]
        self.student.load_state_dict(ckpt_student["student"])
        self.finetune_optim_weight.load_state_dict(
            ckpt_student["finetune_optim_weight"]
        )
        self.finetune_scheduler_student_weight.load_state_dict(
            ckpt_student["finetune_scheduler_student_weight"]
        )

        self.logger.info("=> Continue from epoch {}...".format(self.start_epoch))

    def save_student_ckpt(self, is_bset):
        folder = os.path.join(self.result_dir, "student_model")
        if not os.path.exists(folder):
            os.makedirs(folder)

        ckpt_student = {}
        ckpt_student["best_prec1_after_finetune"] = self.best_prec1_after_finetune
        ckpt_student["start_epoch"] = self.start_epoch
        ckpt_student["student"] = self.student.state_dict()
        ckpt_student["finetune_optim_weight"] = self.finetune_optim_weight.state_dict()
        ckpt_student[
            "finetune_scheduler_student_weight"
        ] = self.finetune_scheduler_student_weight.state_dict()

        if is_bset:
            torch.save(
                ckpt_student,
                os.path.join(folder, "finetune_" + self.arch + "_sparse_best.pt"),
            )
        torch.save(
            ckpt_student,
            os.path.join(folder, "finetune_" + self.arch + "_sparse_last.pt"),
        )

    def finetune(self):
        if self.device == "cuda":
            self.student = self.student.cuda()
            self.ori_loss = self.ori_loss.cuda()
        if self.finetune_resume:
            self.resume_student_ckpt()

        meter_oriloss = meter.AverageMeter("OriLoss", ":.4e")
        meter_loss = meter.AverageMeter("Loss", ":.4e")
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        meter_top5 = meter.AverageMeter("Acc@5", ":6.2f")

        for epoch in range(self.start_epoch + 1, self.finetune_num_epochs + 1):
            # train
            self.student.train()
            self.student.ticket = True
            meter_oriloss.reset()
            meter_loss.reset()
            meter_top1.reset()
            meter_top5.reset()
            finetune_lr = (
                self.finetune_optim_weight.state_dict()["param_groups"][0]["lr"]
                if epoch > 1
                else self.finetune_warmup_start_lr
            )

            with tqdm(total=len(self.train_loader), ncols=100) as _tqdm:
                _tqdm.set_description(
                    "epoch: {}/{}".format(epoch, self.finetune_num_epochs)
                )
                for images, targets in self.train_loader:
                    self.finetune_optim_weight.zero_grad()
                    if self.device == "cuda":
                        images = images.cuda()
                        targets = targets.cuda()
                    logits_student, _ = self.student(images)
                    # loss
                    ori_loss = self.ori_loss(logits_student, targets)

                    total_loss = ori_loss

                    total_loss.backward()
                    self.finetune_optim_weight.step()

                    prec1, prec5 = utils.get_accuracy(
                        logits_student, targets, topk=(1, 5)
                    )
                    n = images.size(0)
                    meter_oriloss.update(ori_loss.item(), n)

                    meter_loss.update(total_loss.item(), n)
                    meter_top1.update(prec1.item(), n)
                    meter_top5.update(prec5.item(), n)

                    _tqdm.set_postfix(
                        loss="{:.4f}".format(meter_loss.avg),
                        top1="{:.4f}".format(meter_top1.avg),
                    )
                    _tqdm.update(1)
                    time.sleep(0.01)

            self.finetune_scheduler_student_weight.step()

            self.writer.add_scalar(
                "finetune_train/loss/ori_loss",
                meter_oriloss.avg,
                global_step=epoch,
            )
            self.writer.add_scalar(
                "finetune_train/loss/total_loss",
                meter_loss.avg,
                global_step=epoch,
            )

            self.writer.add_scalar(
                "finetune_train/acc/top1",
                meter_top1.avg,
                global_step=epoch,
            )
            self.writer.add_scalar(
                "finetune_train/acc/top5",
                meter_top5.avg,
                global_step=epoch,
            )

            self.writer.add_scalar(
                "finetune_train/lr/lr",
                finetune_lr,
                global_step=epoch,
            )

            self.logger.info(
                "[Finetune_train] "
                "Epoch {0} : "
                "LR {lr:.6f} "
                "OriLoss {ori_loss:.4f} "
                "TotalLoss {total_loss:.4f} "
                "Prec@(1,5) {top1:.2f}, {top5:.2f}".format(
                    epoch,
                    lr=finetune_lr,
                    ori_loss=meter_oriloss.avg,
                    total_loss=meter_loss.avg,
                    top1=meter_top1.avg,
                    top5=meter_top5.avg,
                )
            )

            # valid
            self.student.eval()
            self.student.ticket = True
            meter_top1.reset()
            meter_top5.reset()
            with torch.no_grad():
                with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
                    _tqdm.set_description(
                        "epoch: {}/{}".format(epoch, self.finetune_num_epochs)
                    )
                    for images, targets in self.val_loader:
                        if self.device == "cuda":
                            images = images.cuda()
                            targets = targets.cuda()
                        logits_student, _ = self.student(images)
                        prec1, prec5 = utils.get_accuracy(
                            logits_student, targets, topk=(1, 5)
                        )
                        n = images.size(0)
                        meter_top1.update(prec1.item(), n)
                        meter_top5.update(prec5.item(), n)

                        _tqdm.set_postfix(
                            top1="{:.4f}".format(meter_top1.avg),
                            top5="{:.4f}".format(meter_top5.avg),
                        )
                        _tqdm.update(1)
                        time.sleep(0.01)

            self.writer.add_scalar(
                "finetune_val/acc/top1",
                meter_top1.avg,
                global_step=epoch,
            )
            self.writer.add_scalar(
                "finetune_val/acc/top5",
                meter_top5.avg,
                global_step=epoch,
            )

            self.logger.info(
                "[Finetune_val] "
                "Epoch {0} : "
                "Prec@(1,5) {top1:.2f}, {top5:.2f}".format(
                    epoch,
                    top1=meter_top1.avg,
                    top5=meter_top5.avg,
                )
            )

            masks = []
            for _, m in enumerate(self.student.mask_modules):
                masks.append(round(m.mask.mean().item(), 2))
            self.logger.info("[Mask avg] Epoch {0} : ".format(epoch) + str(masks))

            self.start_epoch += 1
            if self.best_prec1_after_finetune < meter_top1.avg:
                self.best_prec1_after_finetune = meter_top1.avg
                self.save_student_ckpt(True)
            else:
                self.save_student_ckpt(False)

            self.logger.info(
                " => Best top1 accuracy before finetune : "
                + str(self.best_prec1_before_finetune)
            )
            self.logger.info(
                " => Best top1 accuracy after finetune : "
                + str(self.best_prec1_after_finetune)
            )
        self.logger.info("Finetune finished!")
        self.logger.info("Best top1 accuracy : " + str(self.best_prec1_after_finetune))
        (
            Flops_baseline,
            Flops,
            Flops_reduction,
            Params_baseline,
            Params,
            Params_reduction,
        ) = get_flops_and_params(self.args)
        self.logger.info(
            "Params_baseline: %.2fM, Params: %.2fM, Params reduction: %.2f%%"
            % (Params_baseline, Params, Params_reduction)
        )
        self.logger.info(
            "Flops_baseline: %.2fM, Flops: %.2fM, Flops reduction: %.2f%%"
            % (Flops_baseline, Flops, Flops_reduction)
        )

    def main(self):
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        self.finetune()
