import os
import sys
import shutil
import time, datetime
import logging
import numpy as np
from pathlib import Path
import time, datetime
import re
import torch


def record_config(args, path):
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if args.resume:
        with open(path, "a") as f:
            f.write(now + "\n\n")
            for arg in vars(args):
                f.write("{}: {}\n".format(arg, getattr(args, arg)))
            f.write("\n")
    else:
        with open(path, "w") as f:
            f.write(now + "\n\n")
            for arg in vars(args):
                f.write("{}: {}\n".format(arg, getattr(args, arg)))
            f.write("\n")


def get_logger(file_path, name):
    logger = logging.getLogger(name)
    log_format = "%(asctime)s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%m/%d %I:%M:%S %p")
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def get_compress_rates(cprate_str):
    cprate_str_list = cprate_str.split("+")
    pat_cprate = re.compile(r"\d+\.\d*")
    pat_num = re.compile(r"\*\d+")
    cprate = []
    for x in cprate_str_list:
        num = 1
        find_num = re.findall(pat_num, x)
        if find_num:
            assert len(find_num) == 1
            num = int(find_num[0].replace("*", ""))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate) == 1
        cprate += [float(find_cprate[0])] * num
    return cprate


def get_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
