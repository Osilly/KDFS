arch=ResNet_50
dataset_dir=dataset_imagenet
dataset_type=imagenet
ckpt_path=ckpt_path
device=0
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
