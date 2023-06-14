import torch
import argparse
from model.student.resnet_sparse_cifar import (
    resnet_56_sparse_cifar10,
    resnet_110_sparse_cifar10,
    resnet_56_sparse_cifar100,
)
from model.student.ResNet_sparse import (
    ResNet_18_sparse_imagenet,
    ResNet_50_sparse_imagenet,
)
from model.student.vgg_bn_sparse import VGG_16_bn_sparse_cifar10
from model.student.DenseNet_sparse import (
    DenseNet_40_sparse_cifar10,
)
from model.student.GoogLeNet_sparse import GoogLeNet_sparse_cifar10
from model.student.MobileNetV2_sparse import MobileNetV2_sparse_imagenet
from model.pruned_model.resnet_pruned_cifar import (
    resnet_56_pruned_cifar10,
    resnet_110_pruned_cifar10,
    resnet_56_pruned_cifar100,
)
from model.pruned_model.ResNet_pruned import (
    ResNet_18_pruned_imagenet,
    ResNet_50_pruned_imagenet,
)
from model.pruned_model.vgg_bn_pruned import VGG_16_bn_pruned_cifar10
from model.pruned_model.DenseNet_pruned import DenseNet_40_pruned_cifar10
from model.pruned_model.GoogLeNet_pruned import GoogLeNet_pruned_cifar10
from model.pruned_model.MobileNetV2_pruned import MobileNetV2_pruned_imagenet

from thop import profile

Flops_baselines = {
    "resnet_56": 125.49,
    "resnet_110": 252.89,
    "ResNet_18": 1820,
    "ResNet_50": 4134,
    "VGG_16_bn": 313.73,
    "DenseNet_40": 282.00,
    "GoogLeNet": 1520,
    "MobileNetV2": 327.55,
}
Params_baselines = {
    "resnet_56": 0.85,
    "resnet_110": 1.72,
    "ResNet_18": 11.60,
    "ResNet_50": 25.50,
    "VGG_16_bn": 14.98,
    "DenseNet_40": 1.04,
    "GoogLeNet": 6.15,
    "MobileNetV2": 3.50,
}
image_sizes = {"cifar10": 32, "cifar100": 32, "imagenet": 224}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="cifar10",
        choices=("cifar10", "cifar100", "imagenet"),
        help="The type of dataset",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet_56",
        choices=(
            "ResNet_18",
            "ResNet_50",
            "vgg_16_bn",
            "resnet_56",
            "resnet_110",
            "DenseNet_40",
            "GoogLeNet",
            "MobileNetV2",
        ),
        help="The architecture to prune",
    )
    parser.add_argument(
        "--sparsed_student_ckpt_path",
        type=str,
        default=None,
        help="The path where to load the sparsed student ckpt",
    )
    return parser.parse_args()


def get_flops_and_params(args):
    student = eval(args.arch + "_sparse_" + args.dataset_type)()
    ckpt_student = torch.load(args.sparsed_student_ckpt_path, map_location="cpu")
    student.load_state_dict(ckpt_student["student"])

    mask_weights = [m.mask_weight for m in student.mask_modules]
    masks = [
        torch.argmax(mask_weight, dim=1).squeeze(1).squeeze(1)
        for mask_weight in mask_weights
    ]
    pruned_model = eval(args.arch + "_pruned_" + args.dataset_type)(masks=masks)
    input = torch.rand(
        [1, 3, image_sizes[args.dataset_type], image_sizes[args.dataset_type]]
    )
    Flops, Params = profile(pruned_model, inputs=(input,), verbose=False)
    Flops_reduction = (
        (Flops_baselines[args.arch] - Flops / (10**6))
        / Flops_baselines[args.arch]
        * 100.0
    )
    Params_reduction = (
        (Params_baselines[args.arch] - Params / (10**6))
        / Params_baselines[args.arch]
        * 100.0
    )
    return (
        Flops_baselines[args.arch],
        Flops / (10**6),
        Flops_reduction,
        Params_baselines[args.arch],
        Params / (10**6),
        Params_reduction,
    )


def main():
    args = parse_args()
    (
        Flops_baseline,
        Flops,
        Flops_reduction,
        Params_baseline,
        Params,
        Params_reduction,
    ) = get_flops_and_params(args=args)
    print(
        "Params_baseline: %.2fM, Params: %.2fM, Params reduction: %.2f%%"
        % (Params_baseline, Params, Params_reduction)
    )
    print(
        "Flops_baseline: %.2fM, Flops: %.2fM, Flops reduction: %.2f%%"
        % (Flops_baseline, Flops, Flops_reduction)
    )


if __name__ == "__main__":
    main()
