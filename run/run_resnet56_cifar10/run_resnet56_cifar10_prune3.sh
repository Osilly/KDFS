arch=resnet_56
result_dir=result/run_resnet56_cifar10_prune3
dataset_dir=dataset_cifar10
dataset_type=cifar10
teacher_ckpt_path=teacher_dir/resnet_56.pt
device=0
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
--compress_rate 0.41 \
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