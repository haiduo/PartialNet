import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from data.custom_imagenet_data import custom_DataModule
from data.custom_imagenet_data import build_transform

__all__ = ['LitDataModule']

def LitDataModule(hparams):
    dm =None
    CLASS_NAMES = None
    drop_last = True
    bs = hparams.batch_size
    bs_eva = hparams.batch_size_eva
    n_gpus = len(hparams.gpus) if isinstance(hparams.gpus, list)  else int(hparams.gpus)
    n_nodes = hparams.num_nodes
    batch_size = int(1.0*bs / n_gpus / n_nodes) if hparams.strategy == 'ddp' else bs
    batch_size_eva = int(1.0*bs_eva / n_gpus / n_nodes) if hparams.strategy == 'ddp' else bs_eva

    if hparams.dataset_name in ["imagenet", "cifar10"]:
        dm = custom_DataModule(
            dataset_name = hparams.dataset_name,
            data_dir=hparams.data_dir+hparams.dataset_name,
            image_size=hparams.image_size,
            num_workers=hparams.num_workers,
            batch_size=batch_size,
            batch_size_eva=batch_size_eva,
            # dist_eval= True if len(str2list(hparams.gpus))>1 else False,
            pin_memory=hparams.pin_memory,
            drop_last=drop_last,
            train_transforms=build_transform(is_train=True, args=hparams, image_size=hparams.image_size),
            val_transforms=build_transform(is_train=False, args=hparams, image_size=hparams.image_size),
            train_transforms_multi_scale=build_transform(is_train=True, args=hparams, image_size=int(hparams.multi_scale.split('_')[0])) if hparams.multi_scale else None,
            scaling_epoch=None if hparams.multi_scale is None else int(hparams.multi_scale.split('_')[1])
        )
    else:
        print("Invalid dataset name, exiting...")
        exit()

    return dm, CLASS_NAMES