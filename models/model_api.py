import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from torch.optim.lr_scheduler import *
from models import *
from utils.utils import *
import torch
import pytorch_lightning as pl

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import accuracy
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.loss import DistillationLoss


global_gconv=[]          # Count the number of channels in each layer of all dynamic partial convolutions.
global_regularizer = []  # Count all convolutional layers regular loss
global_loss_gate = []    # Count all gate_loss of all layers of convolutions
global_n_div = []        # Count the number of groups in each layer of all convolutions
div_names = [f"{i+1}th-layer-n_div" for i in range(13)]
loss_names = [f"{i+1}th-layer-gate_loss" for i in range(13)]
global_epoch = 0

def build_criterion(args):
    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    return criterion


def build_mixup_fn(args, num_classes):
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=num_classes)
    return mixup_fn


class LitModel(pl.LightningModule):
    def __init__(self, num_classes, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        if 'partialnet' in hparams.model_name:
            self.model = create_model(
                hparams.model_name,
                mlp_ratio=hparams.mlp_ratio,
                embed_dim=hparams.embed_dim,
                depths=hparams.depths,
                pretrained=hparams.pretrained,
                n_div=hparams.n_div,
                print_n_div=hparams.print_n_div,
                auto_div=hparams.auto_div,
                u_regular=hparams.u_regular,
                l_gate=hparams.l_gate,
                index_div=hparams.index_div,
                feature_dim=hparams.feature_dim,
                patch_size=hparams.patch_size,
                patch_stride=hparams.patch_stride,
                patch_size2=hparams.patch_size2,
                patch_stride2=hparams.patch_stride2,
                num_classes=num_classes,
                layer_scale_init_value=hparams.layer_scale_init_value,
                drop_path_rate=hparams.drop_path_rate,
                norm_layer=hparams.norm_layer,
                act_layer=hparams.act_layer,
                pconv_fw_type=hparams.pconv_fw_type,
                pre_epoch = hparams.pre_epoch,
                use_channel_attn=hparams.use_channel_attn,
                use_spatial_attn=hparams.use_spatial_attn,
                patnet_t0=hparams.patnet_t0,
            )
        elif 'convnext_tiny' in hparams.model_name:
            self.model = create_model(
                hparams.model_name,
                pretrained=hparams.pretrained,
                num_classes=num_classes,
                n_div=hparams.n_div,
                pconv_fw_type=hparams.pconv_fw_type,
                use_channel_attn=hparams.use_channel_attn,
                use_spatial_attn=hparams.use_spatial_attn,
            )
        else:
            self.model = create_model(
                hparams.model_name,
                pretrained=hparams.pretrained,
                num_classes=num_classes
            )

        base_criterion = build_criterion(hparams)
        self.distillation_type = hparams.distillation_type
        if hparams.distillation_type == 'none':
            self.criterion = base_criterion
        else:
            # assert hparams.teacher_path, 'need to specify teacher-path when using distillation'
            print(f"Creating teacher model: {hparams.teacher_model}")
            teacher_model = create_model(
                hparams.teacher_model,
                pretrained=True,
                num_classes=num_classes,
                global_pool='avg',
            )
            for param in teacher_model.parameters():
                param.requires_grad = False
            teacher_model.eval()
            self.criterion = DistillationLoss(base_criterion,
                                              teacher_model,
                                              hparams.distillation_type,
                                              hparams.distillation_alpha,
                                              hparams.distillation_tau
                                              )
        self.criterion_eva = torch.nn.CrossEntropyLoss()
        self.mixup_fn = build_mixup_fn(hparams, num_classes)
        self.auto_div = hparams.auto_div
        self.u_regular=hparams.u_regular
        self.u_regular_theta=hparams.u_regular_theta
        self.l_gate=hparams.l_gate
        self.u_regular_b = hparams.u_regular_b
        self.loss_gate_alpha = hparams.loss_gate_alpha
        self.penalize_alpha = hparams.penalize_alpha
        self.print_n_div=hparams.print_n_div
        self.print_loss_gate=hparams.print_loss_gate
        self.pre_epoch = hparams.pre_epoch

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        if self.hparams.multi_scale is not None:
            if self.current_epoch == int(self.hparams.multi_scale.split('_')[1]):
                # image_size = self.hparams.image_size
                self.trainer.reset_train_dataloader(self)

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        if mode == "train" and self.mixup_fn is not None:
            global global_epoch
            global_epoch = self.current_epoch
            imgs, labels = self.mixup_fn(imgs, labels)
        preds = self.model(imgs) # forward proess

        if mode == "train":
            if self.distillation_type == 'none':
                loss = self.criterion(preds, labels)
            else:
                loss = self.criterion(imgs, preds, labels)

            if self.auto_div and self.u_regular:
                global global_regularizer, global_gconv
                if self.pre_epoch == 0 or self.current_epoch >= self.pre_epoch:
                    assert len(global_regularizer) == len(global_gconv)
                    # print(f"u_regular:{global_regularizer}")
                    all_global_regularizer = sum(global_regularizer)  
                    all_global_gconv = sum(global_gconv)/(self.u_regular_b**2)
                    # print(f"u_regular_sum:{all_global_regularizer.data}")
                    if self.u_regular_theta > 0:
                        if all_global_regularizer > all_global_gconv:
                            alpha = self.penalize_alpha 
                        elif all_global_regularizer >= all_global_gconv * self.u_regular_theta:
                            alpha = 0
                        else:
                            # alpha = -self.penalize_alpha * self.u_regular_theta
                            alpha = -self.penalize_alpha 
                    else:
                        alpha = 0 if all_global_regularizer <= all_global_gconv else self.penalize_alpha

                    u_loss = torch.pow(all_global_gconv/all_global_regularizer, alpha)
                    loss = loss*u_loss
                    global_regularizer = []

            if self.auto_div and self.l_gate:
                global global_loss_gate
                # print(f"loss_gate:{global_loss_gate}")
                loss_gate_sum = sum(global_loss_gate)
                # print(f"loss_gate_sum:{loss_gate_sum}")
                loss = loss + loss_gate_sum*self.loss_gate_alpha

                if self.global_rank == 0 and (self.print_n_div or self.print_loss_gate):
                    print(f"Epoch:{self.current_epoch}, Bacth:{self.global_step}") 
                    if self.print_n_div:
                        global global_n_div
                        global div_names
                        all_n_div = []
                        for i, x in enumerate(global_n_div):
                            self.logger.experiment.log({div_names[i]:x}) 
                            all_n_div.append(x.data.cpu().numpy().item())
                        print(f"global_n_div:{all_n_div}")
                        global_n_div = []
                    if self.print_loss_gate:
                        global loss_names
                        loss_gate = []
        
                        for i, x in enumerate(global_loss_gate):
                            self.logger.experiment.log({loss_names[i]:x})
                            loss_gate.append(x.data.cpu().numpy().item())
                        print(f"loss_gate:{loss_gate}")
                global_loss_gate = []
            self.log("%s_loss" % mode, loss)
        else:
            loss = self.criterion_eva(preds, labels)
            acc1, acc5 = accuracy(preds, labels, topk=(1, 5))
            sync_dist = True if torch.cuda.device_count() > 1 else False
            self.log("%s_loss" % mode, loss, sync_dist=sync_dist)
            self.log("%s_acc1" % mode, acc1, sync_dist=sync_dist)
            self.log("%s_acc5" % mode, acc5, sync_dist=sync_dist)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    def configure_optimizers(self):
        optimizer = create_optimizer(self.hparams, self.parameters())
        if self.hparams.sched == 'cosine':
            if self.hparams.warmup:
                scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                warmup_epochs=self.hparams.warmup_epochs,
                                max_epochs=self.hparams.epochs,
                                warmup_start_lr=self.hparams.warmup_lr,
                                eta_min=self.hparams.min_lr)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.epochs)
        else:
            raise NotImplementedError

        return [optimizer], [scheduler]

