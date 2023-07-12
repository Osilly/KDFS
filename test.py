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


class Test:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_type = args.dataset_type
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path

    def dataload(self):
        dataset = eval("Dataset_" + self.dataset_type)(
            self.dataset_dir,
            self.test_batch_size,
            self.test_batch_size,
            self.num_workers,
            self.pin_memory,
        )
        self.train_loader, self.val_loader = (
            dataset.loader_train,
            dataset.loader_test,
        )
        print("Dataset has been loaded!")

    def build_model(self):
        print("==> Building model..")

        print("Loading student model")
        self.student = eval(self.arch + "_sparse_" + self.dataset_type)()
        ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu")
        self.student.load_state_dict(ckpt_student["student"])

    def test(self):
        if self.device == "cuda":
            self.student = self.student.cuda()

        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
        meter_top5 = meter.AverageMeter("Acc@5", ":6.2f")

        self.student.eval()
        self.student.ticket = True
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
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

        print(
            "[Test] "
            "Prec@(1,5) {top1:.2f}, {top5:.2f}".format(
                top1=meter_top1.avg,
                top5=meter_top5.avg,
            )
        )

        (
            Flops_baseline,
            Flops,
            Flops_reduction,
            Params_baseline,
            Params,
            Params_reduction,
        ) = get_flops_and_params(args=self.args)
        print(
            "Params_baseline: %.2fM, Params: %.2fM, Params reduction: %.2f%%"
            % (Params_baseline, Params, Params_reduction)
        )
        print(
            "Flops_baseline: %.2fM, Flops: %.2fM, Flops reduction: %.2f%%"
            % (Flops_baseline, Flops, Flops_reduction)
        )

    def main(self):
        self.dataload()
        self.build_model()
        self.test()
