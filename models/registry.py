import torch.nn as nn
from timm.models.registry import register_model
from .partialnet import PartialNet
from .convnext import create_convnext, ConvNeXt


@register_model
def partialnet(**kwargs):
    model = PartialNet(**kwargs)
    return model

@register_model
def convnext_tiny(pretrained=False, **kwargs) -> ConvNeXt:
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
    model = create_convnext('convnext_tiny', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


