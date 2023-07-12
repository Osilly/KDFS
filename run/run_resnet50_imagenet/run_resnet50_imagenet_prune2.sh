arch=ResNet_50
result_dir=result/run_resnet50_imagenet_prune2
dataset_dir=dataset_imagenet
dataset_type=imagenet
teacher_ckpt_path=teacher_dir/resnet50-19c8e357.pth
device=0,1,2,3
master_port=6681
CUDA_VISIBLE_DEVICES=$device torchrun --nproc_per_node=4 --master_port $master_port main.py \
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
--num_epochs 250 \
--lr 4e-3 \
--warmup_steps 10 \
--warmup_start_lr 4e-5 \
--lr_decay_T_max 250 \
--lr_decay_eta_min 4e-5 \
--weight_decay 2e-5 \
--train_batch_size 256 \
--eval_batch_size 256 \
--target_temperature 3 \
--gumbel_start_temperature 1 \
--gumbel_end_temperature 0.1 \
--coef_kdloss 0.05 \
--coef_rcloss 1000 \
--coef_maskloss 10000 \
--compress_rate 0.56 \
--ddp \
&& \
CUDA_VISIBLE_DEVICES=$device torchrun --nproc_per_node=4 --master_port $master_port main.py \
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
--finetune_num_epochs 20 \
--finetune_lr 4e-6 \
--finetune_warmup_steps 5 \
--finetune_warmup_start_lr 4e-8 \
--finetune_lr_decay_T_max 20 \
--finetune_lr_decay_eta_min 4e-8 \
--finetune_weight_decay 2e-5 \
--finetune_train_batch_size 256 \
--finetune_eval_batch_size 256 \
--sparsed_student_ckpt_path $result_dir"/student_model/finetune_"$arch"_sparse_best.pt" \
--ddp \