import argparse
import os
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import timm
from torch.autograd import Variable


def reshape_transform_resmlp(tensor, height=14, width=14):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_swin(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default="./test/0bf7bfb05.png", help='Input image path')
    parser.add_argument('--output-image-path', type=str, default=None, help='Output image path')
    parser.add_argument('--model', type=str, default='partialnet', help='model name')
    parser.add_argument('--aug_smooth', action='store_true', help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true', help='Reduce noise by taking the first principle componenet of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam', help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam/eigencam/eigengradcam/layercam')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('checkpoints/ConvNext/best.pth', map_location='cpu')
    reshape_transform = None
    if 'poolformer' in args.model:
        target_layers = [model.network[-1]]  # [model.network[-1][-2]]
        print(target_layers)
    elif 'resnet' in args.model:
        target_layers = [model.layer4[-1]]
    elif 'partialnet' in args.model:
        target_layers = [model.stages[-1]]
    elif 'resmlp' in args.model:
        target_layers = [model.blocks[-1]]
        reshape_transform = reshape_transform_resmlp
    elif 'deit' in args.model:
        target_layers = [model.blocks[-1].norm1]
        reshape_transform = reshape_transform_vit
    elif 'swin' in args.model:
        target_layers = [model.layers[-1].blocks[-1]]
        reshape_transform = reshape_transform_swin
    print(target_layers)
    model.eval()
    model.to(DEVICE)
    img_path = args.image_path
    if args.image_path:
        img_path = args.image_path
    else:
        import requests

        image_url = ''
        img_path = image_url.split('/')[-1]
        if os.path.exists(img_path):
            img_data = requests.get(image_url).content
            with open(img_path, 'wb') as handler:
                handler.write(img_data)

    if args.output_image_path:
        save_name = args.output_image_path
    else:
        img_type = img_path.split('.')[-1]
        it_len = len(img_type)
        save_name = img_path.split('/')[-1][:-(it_len + 1)]
        save_name = save_name + '_' + args.model + '.' + img_type

    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    if args.model == 'resize':
        cv2.imwrite(save_name, img)
    else:
        rgb_img = img[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = Variable(preprocess_image(rgb_img,
                                                 mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]), requires_grad=True).to(DEVICE)
        targets = None
        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model,
                           target_layers=target_layers,
                           use_cuda=args.use_cuda,
                           reshape_transform=reshape_transform,
                           ) as cam:

            cam.batch_size = 1
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_name, cam_image)
