import os
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torchvision.datasets as datasets
from nvidia.dali.pipeline import Pipeline
import torchvision.transforms as transforms
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator


class HybridTrainPipeline_imagenet(Pipeline):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_threads,
        device_id=0,
        local_rank=0,
        world_size=1,
    ):
        # As we're recreating the Pipeline at every epoch, the seed must be -1 (random seed)
        super().__init__(batch_size, num_threads, device_id, seed=-1)
        dali_device = "gpu"
        self.input = ops.readers.File(
            file_root=data_dir,
            shard_id=local_rank,
            num_shards=world_size,
            random_shuffle=True,
        )
        self.decode = ops.decoders.Image(device="mixed", output_type=types.RGB)
        self.res = ops.RandomResizedCrop(
            device="gpu", size=224, random_area=[0.08, 1.25]
        )
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            image_type=types.RGB,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        self.coin = ops.random.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


class HybridValPipeline_imagenet(Pipeline):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_threads,
        device_id=0,
    ):
        super().__init__(batch_size, num_threads, device_id, seed=-1)
        self.input = ops.readers.File(
            file_root=data_dir,
            random_shuffle=False,
        )
        self.decode = ops.decoders.Image(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(
            device="gpu", resize_shorter=256, interp_type=types.INTERP_TRIANGULAR
        )
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(224, 224),
            image_type=types.RGB,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


class Dataset_imagenet_dali:
    def __init__(
        self,
        dataset_dir,
        train_batch_size,
        eval_batch_size,
        num_threads=8,
        device_id=0,
        local_rank=0,
        world_size=1,
    ):
        train_dir = os.path.join(dataset_dir, "train")
        test_dir = os.path.join(dataset_dir, "val")
        pipeline_train = HybridTrainPipeline_imagenet(
            data_dir=train_dir,
            batch_size=train_batch_size,
            num_threads=num_threads,
            device_id=device_id,
            local_rank=local_rank,
            world_size=world_size,
        )
        pipeline_train.build()
        self.loader_train = DALIClassificationIterator(
            pipelines=pipeline_train,
            size=pipeline_train.epoch_size("Reader") // world_size,
            auto_reset=True,
        )

        pipeline_test = HybridValPipeline_imagenet(
            data_dir=test_dir,
            batch_size=eval_batch_size,
            num_threads=num_threads,
            device_id=device_id,
        )
        pipeline_test.build()
        self.loader_test = DALIClassificationIterator(
            pipelines=pipeline_test,
            size=pipeline_test.epoch_size("Reader"),
            auto_reset=True,
        )
