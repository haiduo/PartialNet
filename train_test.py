import torch
from argparse import ArgumentParser
import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy 

from utils.utils import *
from utils.fuse_conv_bn import fuse_conv_bn
from data.data_api import LitDataModule
from models.model_api import LitModel


def main(args):
    if args.seed:
        seed_everything(args.seed)
    # Init data pipeline
    dm, _ = LitDataModule(hparams=args)

    # init LitModel
    if args.checkpoint_path:
        PATH = args.checkpoint_path
        if PATH[-5:]=='.ckpt':
            model = LitModel.load_from_checkpoint(PATH, map_location='cpu', num_classes=dm.num_classes, hparams=args)
            print('Successfully load the pl checkpoint file.')
            if args.pl_ckpt_2_torch_pth:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.model.to(device)
                torch.save(model.state_dict(), PATH[:-5]+'.pth')
                exit()
        elif PATH[-4:] == '.pth':
            model = LitModel(num_classes=dm.num_classes, hparams=args)
            missing_keys, unexpected_keys = model.model.load_state_dict(torch.load(PATH), False)
            # show for debug
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)
        else:
            raise TypeError
    else:
        model = LitModel(num_classes=dm.num_classes, hparams=args)

    flops, params = get_flops_params(model.model, args.image_size)

    if args.fuse_conv_bn:
        fuse_conv_bn(model.model)

    if args.measure_latency:
        dm.prepare_data()
        dm.setup(stage="test")
        for idx, (images, _) in enumerate(dm.test_dataloader()):
            model = model.model.eval()
            throughput, latency = measure_latency(images[:1, :, :, :], model, GPU=False, num_threads=1)
            if torch.cuda.is_available():
                throughput, latency = measure_latency(images, model, GPU=True)
            exit()

    # print_model(model)

    # Callbacks
    MONITOR = 'val_acc1'
    checkpoint_callback = ModelCheckpoint(
        monitor=MONITOR,
        dirpath=args.model_ckpt_dir,
        filename=args.model_name+'_'+str(args.num_model)+'-{epoch}-{val_acc1:.4f}',
        save_top_k=1,
        save_last=True,
        mode='max' if 'acc' in MONITOR else 'min'
    )
    refresh_callback = TQDMProgressBar(refresh_rate=20)
    callbacks = [
        checkpoint_callback, # save checkpoint
        refresh_callback     
    ]

    # Initialize wandb logger
    WANDB_ON = True if args.dev+args.test_phase == 0 else False
    # WANDB_ON = False
    if WANDB_ON:
        wandb_logger = WandbLogger(
            project=args.wandb_project_name+'_'+str(args.num_model), #Specify the name of the WandB project
            save_dir=args.wandb_save_dir,    #Specify the directory where WandB logs are saved
            offline=args.wandb_offline,      
            log_model=False,                 #no checkpoint is logged
            job_type='train')
        wandb_logger.log_hyperparams(args)
        wandb_logger.log_hyperparams({"flops": flops, "params": params})

    # wandb_logger.watch(model) #Auto record initial model structure

    # Initialize a trainer
    find_unused_para = False if args.distillation_type == 'none' else True
    if args.pre_epoch > 0:  #Start fixed PConv pre-training
        find_unused_para = True
    trainer = pl.Trainer(
        fast_dev_run=args.dev,
        logger=wandb_logger if WANDB_ON else None,
        max_epochs=args.epochs,
        accelerator="gpu", #Hardware type settings cpu, gpu, tpu and automatic selection auto
        devices=args.gpus, #the number of hardware to call. Pass in a list to call the GPU with id [0,1]
        # gpus=1, #Deprecated in 2.0
        sync_batchnorm=args.sync_batchnorm,
        num_nodes=args.num_nodes,
        gradient_clip_val=args.clip_grad,
        strategy= DDPStrategy(find_unused_parameters=find_unused_para) if args.strategy == 'ddp' else args.strategy,
        callbacks=callbacks,    
        precision=args.precision, # Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16)
        benchmark=args.benchmark  
    )

    if bool(args.test_phase):
        trainer.test(model, datamodule=dm)
    else:
        trainer.fit(model, dm)
        if args.dev==0:
            trainer.test(ckpt_path="best", datamodule=dm)

    # Close wandb run
    if WANDB_ON:
        wandb.finish()

if __name__ == "__main__":
    parser = ArgumentParser()
    name_run =str("PartialNet_ImageNet")
    parser.add_argument('--seed', default=42, choices=[0, 42, 3407], type=int, help='seed for initializing training')
    parser.add_argument('--cfg', type=str, default='code/PartialNet/cfg/cifar10-PartialNet_to.yaml.yaml')
    parser.add_argument("--gpus", default=[0,1,2,3] , help="Number of GPUs [0,] or [0,1,2,3] applied per node.")
    parser.add_argument("--num_model", default=name_run+'-'+str(1)) 
    parser.add_argument("--dev", type=int, default=0, help='fast_dev_run for debug')
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--model_ckpt_dir", type=str, default="code/PartialNet/model_ckpt")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument('--pin_memory', default=True)
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--pconv_fw_type", type=str, default='split_cat', help="use 'split_cat' for training/inference and 'slicing' only for inference")
    parser.add_argument('--measure_latency', default=False, action='store_true', help='measure latency or throughput')
    parser.add_argument('--test_phase', default=False, action='store_true')
    parser.add_argument('--fuse_conv_bn', default=False, action='store_true') 
    parser.add_argument("--wandb_project_name", type=str, default=name_run) 
    parser.add_argument('--wandb_offline', default=False, action='store_true') 
    parser.add_argument('--wandb_save_dir', type=str, default='code/PartialNet/wandb_log')
    parser.add_argument('--pl_ckpt_2_torch_pth', action='store_true', help='convert pl .ckpt file to torch .pth file, and then exit')

    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    args = merge_args_cfg(args, cfg)

    main(args)
