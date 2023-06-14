import torch
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os


class Dataset_cifar10:
    def __init__(
        self,
        dataset_dir,
        train_batch_size,
        eval_batch_size,
        num_workers=8,
        pin_memory=True,
        ddp=False,
    ):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        trainset = CIFAR10(
            root=dataset_dir, train=True, download=True, transform=transform_train
        )
        if ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset, shuffle=True
            )
            self.loader_train = DataLoader(
                trainset,
                batch_size=train_batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=train_sampler,
            )
        else:
            self.loader_train = DataLoader(
                trainset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        testset = CIFAR10(
            root=dataset_dir, train=False, download=True, transform=transform_test
        )
        self.loader_test = DataLoader(
            testset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


class Dataset_cifar100:
    def __init__(
        self,
        dataset_dir,
        train_batch_size,
        eval_batch_size,
        num_workers=8,
        pin_memory=True,
        ddp=False,
    ):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        trainset = CIFAR100(
            root=dataset_dir, train=True, download=True, transform=transform_train
        )
        if ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset, shuffle=True
            )
            self.loader_train = DataLoader(
                trainset,
                batch_size=train_batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=train_sampler,
            )
        else:
            self.loader_train = DataLoader(
                trainset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        testset = CIFAR100(
            root=dataset_dir, train=False, download=True, transform=transform_test
        )
        self.loader_test = DataLoader(
            testset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


class Dataset_imagenet:
    def __init__(
        self,
        dataset_dir,
        train_batch_size,
        eval_batch_size,
        num_workers=8,
        pin_memory=True,
        ddp=False,
    ):
        scale_size = 224

        train_dir = os.path.join(dataset_dir, "train")
        test_dir = os.path.join(dataset_dir, "val")

        trainset = ImageFolder(
            train_dir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(scale_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
        if ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset, shuffle=True
            )
            self.loader_train = DataLoader(
                trainset,
                batch_size=train_batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=train_sampler,
            )
        else:
            self.loader_train = DataLoader(
                trainset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        testset = ImageFolder(
            test_dir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Resize(scale_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
        self.loader_test = DataLoader(
            testset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
