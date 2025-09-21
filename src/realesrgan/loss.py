from collections import OrderedDict

import config
import cv2
import numpy as np
import torch
import tensorflow as tf
from keras.models import Model
from keras.preprocessing.image import img_to_array
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models import VGG19_Weights, vgg19

from realesrgan.utils import load_model_functions, padding_functions, robust_consistent_crop

# phi_5,4 5th conv layer before maxpooling but after activation


NAMES = {
    'vgg11': [
        'conv1_1', 'relu1_1', 'pool1', 'conv2_1', 'relu2_1', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
        'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
        'pool5'
    ],
    'vgg13': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'
    ],
    'vgg16': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
        'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
        'pool5'
    ],
    'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    ]
}


class VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): Forward function returns the corresponding
            features according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image. Importantly,
            the input feature must in the range [0, 1]. Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
        remove_pooling (bool): If true, the max pooling operations in VGG net
            will be removed. Default: False.
        pooling_stride (int): The stride of max pooling operation. Default: 2.
    """

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 requires_grad=False,
                 remove_pooling=False,
                 pooling_stride=2):
        super(VGGFeatureExtractor, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        self.names = NAMES['vgg19']

        # only borrow layers that will be used to avoid unused params
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx

        vgg_net = vgg19(weights=VGG19_Weights)

        features = vgg_net.features[:max_idx + 1]

        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if 'pool' in k:
                # if remove_pooling is true, pooling operation will be removed
                if remove_pooling:
                    continue
                else:
                    # in some cases, we may want to change the default stride
                    modified_net[k] = nn.MaxPool2d(
                        kernel_size=2,
                        stride=pooling_stride
                    )
            else:
                modified_net[k] = v

        self.vgg_net = nn.Sequential(modified_net)

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                'mean',
                torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            # the std is for image with range [0, 1]
            self.register_buffer(
                'std',
                torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean.to(config.DEVICE)) / self.std.to(config.DEVICE)

        output = {}
        for key, layer in self.vgg_net.to(config.DEVICE)._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()

        return output


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss = percep_loss + torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss = percep_loss + self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss = percep_loss * self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss = style_loss + torch.norm(
                        self._gram_mat(
                            x_features[k]
                        ) - self._gram_mat(
                                gt_features[k]
                        ), p='fro'
                    ) * self.layer_weights[k]
                else:
                    style_loss = style_loss + self.criterion(
                        self._gram_mat(
                            x_features[k]
                        ), self._gram_mat(
                            gt_features[k]
                        )
                    ) * self.layer_weights[k]
            style_loss = style_loss * self.style_weight
        else:
            style_loss = None

        return percep_loss  # , style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


class VGGLoss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.vgg = (
            vgg19(weights=VGG19_Weights).features[:35].eval().to(config.DEVICE)
        )
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)

        return self.loss(vgg_input_features, vgg_target_features)


_reduction_modes = ['none', 'mean', 'sum']


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * nn.functional.l1_loss(pred, target, weight, reduction=self.reduction)


class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        self.loss = nn.BCEWithLogitsLoss()

    def get_target_label(self, input, target_is_real):
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)

        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


class OCRFeatureExtractor(nn.Module):
    def __init__(self, OCR_path=None):
        super().__init__()
        self.to_numpy = transforms.ToPILImage()
        self.criterion = torch.nn.L1Loss()
        if OCR_path is not None:
            self.OCR_path = OCR_path
            with tf.device('/CPU:0'):
                self.OCR = load_model_functions(self.OCR_path)
                self.OCR = Model(self.OCR.input, self.OCR.layers[-41].output)
                self.IMAGE_DIMS = self.OCR.input_shape[1:]
                self.parameters = np.load(
                    self.OCR_path + '/parameters.npy',
                    allow_pickle=True
                ).item()
                self.tasks = self.parameters['tasks']
                self.ocr_classes = self.parameters['ocr_classes']
                self.num_classes = self.parameters['num_classes']
                self.padding = True
                self.aspect_ratio = (
                    self.IMAGE_DIMS[1]
                    / self.IMAGE_DIMS[0]
                )
                self.min_ratio = self.aspect_ratio - 0.15
                self.max_ratio = self.aspect_ratio + 0.15
                self.OCR.compile()
        else:
            self.OCR_path = None

    def OCR_pred(self, img, fl=None, convert_to_bgr=True):
        if convert_to_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.padding:
            img, _, _ = padding_functions(
                img,
                self.min_ratio,
                self.max_ratio,
                color=(127, 127, 127)
            )

        img = cv2.resize(img, (self.IMAGE_DIMS[1], self.IMAGE_DIMS[0]))
        img = img_to_array(img)
        img = img.reshape(
            1,
            self.IMAGE_DIMS[0],
            self.IMAGE_DIMS[1],
            self.IMAGE_DIMS[2]
        )

        img = (img / 255.0).astype('float')
        predictions = self.OCR.predict(img)

        return predictions

    def forward(self, x, gt):
        batch_x = []
        batch_gt = []
        if self.OCR_path is not False:
            for imgX, imgGt in zip(x, gt):
                img_x = np.array(self.to_numpy(imgX.to('cpu'))).astype('uint8')
                img_gt = np.array(self.to_numpy(imgGt.to('cpu'))).astype('uint8')
                img_gt, img_x = robust_consistent_crop(img_gt, img_x)

                if self.OCR is not None:
                    features_x = self.OCR_pred(img_x)
                    features_gt = self.OCR_pred(img_gt)
                batch_x.append(features_x)
                batch_gt.append(features_gt)
        logits_x = Variable(torch.as_tensor(batch_x), requires_grad=True)
        logits_gt = Variable(torch.as_tensor(batch_gt), requires_grad=True)

        # logits = logits.view(logits.size(0), -1)

        return self.criterion(logits_x, logits_gt)
