# KDFS
The official pytorch implementation of KDFS.

Paper link: [Filter Pruning for Efficient CNNs via Knowledge-driven Differential Filter Sampler](https://arxiv.org/abs/2307.00198)

![KDFS](fig/KDFS.png)

The illustration of KDFS. Its key components lie in differential filter sampler, masked filter modeling between teacher output and student decoder, and FLOPs regularization term $\mathcal{R}$. The sampler with additional sampling parameters is proposed to generate binary masks to automatically select the filters, where dotted cubes denote the mask values are $0$. To better guide the sampling, masked filter modeling exploits the prior knowledge from a pre-trained model (a.k.a. teacher) to construct PCA-like knowledge (i.e., RL loss), which aligns the teacher intermediate features and the outputs of student decoder taking sampling features as the input. We leverage FLOPs regularization $\mathcal{R}$ into RL loss, CE loss, and KD loss and optimize the sampler and weights directly in an end-to-end manner.

## Citation

If you find KDFS useful in your research, please consider citing:

```
@article{lin2023filter,
  title={Filter Pruning for Efficient CNNs via Knowledge-driven Differential Filter Sampler},
  author={Lin, Shaohui and Huang, Wenxuan and Xie, Jiao and Zhang, Baochang and Shen, Yunhang and Yu, Zhou and Han, Jungong and Doermann, David},
  journal={arXiv preprint arXiv:2307.00198},
  year={2023}
}
```

## Requirements

- Pytorch 1.10
- tensorboard 2.11.0
- nvidia-dali-cuda110 1.23.0 (optional)

## Training

### 1. download pre-trained model

First you need to download the pre-trained model, you can get the pre-trained model we used at the following link:

[Baidu Wangpan](https://pan.baidu.com/s/16b_iA3GINmnn1xHHC6InQA ) (password: zshb)

The performance of the pre-trained models are shown below:

| model              | FLOPs   | Params | TOP-1 Acc. |
| ------------------ | ------- | ------ | ---------- |
| resnet_56_cifar10  | 125.49M | 0.85M  | 93.26%     |
| resnet_110_cifar10 | 252.89M | 1.72M  | 93.50%     |
| resnet_56_cifar100 | 125.49M | 0.85M  | 71.33%     |
| resnet_50_imagenet | 4134M   | 25.50M | 76.15%     |

Place the pre-trained model in folder `teacher_dir` .

### 2. Train

We provide training scripts for the models at different compression rates, they are in folder `run`.

You can train directly using the following script, for example, to train resnet56 on cifar10:

```
bash run/run_resnet56_cifar10/run_resnet56_cifar10_prune1.sh
```

The details of the script are as follows, you need to change the path and such parameters:

**run/run_resnet56_cifar10/run_resnet56_cifar10_prune1.sh:**

```
arch=resnet_56 # Model name
result_dir=result/run_resnet56_cifar10_prune1 # The path where you want to go to save the results
dataset_dir=dataset_cifar10 # dataset path
dataset_type=cifar10 # dataset type
teacher_ckpt_path=teacher_dir/resnet_56.pt # path of the pre-trained model
device=0 # gpu id
CUDA_VISIBLE_DEVICES=$device python main.py \
--phase train \
--dataset_dir $dataset_dir \
--dataset_type $dataset_type \
--num_workers 8 \
--pin_memory \
--device cuda \
--arch $arch \
--seed 3407 \
--result_dir $result_dir \
--teacher_ckpt_path $teacher_ckpt_path \
--num_epochs 350 \
--lr 1e-2 \
--warmup_steps 20 \
--warmup_start_lr 1e-4 \
--lr_decay_T_max 350 \
--lr_decay_eta_min 1e-4 \
--weight_decay 1e-4 \
--train_batch_size 256 \
--eval_batch_size 256 \
--target_temperature 3 \
--gumbel_start_temperature 1 \
--gumbel_end_temperature 0.1 \
--coef_kdloss 0.05 \
--coef_rcloss 1000 \
--coef_maskloss 10000 \
--compress_rate 0.57 \
&& \
CUDA_VISIBLE_DEVICES=$device python main.py \
--phase finetune \
--dataset_dir $dataset_dir \
--dataset_type $dataset_type \
--num_workers 8 \
--pin_memory \
--device cuda \
--arch $arch \
--seed 3407 \
--result_dir $result_dir \
--finetune_student_ckpt_path $result_dir"/student_model/"$arch"_sparse_last.pt" \
--finetune_num_epochs 50 \
--finetune_lr 1e-4 \
--finetune_warmup_steps 10 \
--finetune_warmup_start_lr 1e-6 \
--finetune_lr_decay_T_max 50 \
--finetune_lr_decay_eta_min 1e-6 \
--finetune_weight_decay 1e-4 \
--finetune_train_batch_size 256 \
--finetune_eval_batch_size 256 \
--sparsed_student_ckpt_path $result_dir"/student_model/finetune_"$arch"_sparse_best.pt" \
```

The performance corresponding to these scripts is as follows:

cifar10:

| script                          | baseline FLOPs | FLOPs (FLOPs reduction) | TOP-1 Acc. |
| ------------------------------- | -------------- | ----------------------- | ---------- |
| run_resnet56_cifar10_prune1.sh  | 125.49M        | 74.22M (40.85%)         | 93.78%     |
| run_resnet56_cifar10_prune2.sh  | 125.49M        | 61.25M (51.19%)         | 93.58%     |
| run_resnet56_cifar10_prune3.sh  | 125.49M        | 51.24M (59.17%)         | 93.19%     |
| run_resnet110_cifar10_prune1.sh | 252.89M        | 122.61M (51.52%)        | 94.23%     |
| run_resnet110_cifar10_prune2.sh | 252.89M        | 98.80M (60.93%)         | 93.65%     |

cifar100:

| script                          | baseline FLOPs | FLOPs (FLOPs reduction) | TOP-1 Acc. |
| ------------------------------- | -------------- | ----------------------- | ---------- |
| run_resnet56_cifar100_prune1.sh | 125.49M        | 60.26M (51.98%)         | 71.65%     |

imagenet:

| script                          | baseline FLOPs | FLOPs (FLOPs reduction) | TOP-1 Acc. | TOP-5 Acc. |
| ------------------------------- | -------------- | ----------------------- | ---------- | ---------- |
| run_resnet50_imagenet_prune1.sh | 4134M          | 2384M (42.32%)          | 76.26%     | 93.07%     |
| run_resnet50_imagenet_prune2.sh | 4134M          | 1845M (55.36%)          | 75.80%     | 92.66%     |

## Test

We provide trained weights and they can be accessed from the following links:

[Baidu Wangpan](https://pan.baidu.com/s/1RmAkmejtB37MSr9N10dG1A  ) (password: 9qyz)

You can use the test scripts we provide in folder `run` to get the performance of the trained weights, for example, to test resnet56 on cifar10:

```
bash run/run_resnet56_cifar10/test_resnet56_cifar10.sh
```

The details of the script are as follows, you need to change the path and such parameters:

**run/run_resnet56_cifar10/test_resnet56_cifar10.sh:**

```
arch=resnet_56 # Model name
dataset_dir=dataset_cifar10 # dataset path
dataset_type=cifar10 # dataset type
ckpt_path=ckpt_path # The weight path you want to test
device=0 # gpu id
CUDA_VISIBLE_DEVICES=$device python main.py \
--phase test \
--dataset_dir $dataset_dir \
--dataset_type $dataset_type \
--num_workers 8 \
--pin_memory \
--device cuda \
--arch $arch \
--test_batch_size 256 \
--sparsed_student_ckpt_path $ckpt_path \
```

## Tips

If you find any problems, please feel free to contact to the authors (osilly0616@gmail.com).
