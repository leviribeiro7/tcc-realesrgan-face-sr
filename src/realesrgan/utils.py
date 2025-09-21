import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import config
import cv2
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import torch
from keras.models import model_from_json


def find_crop_coordinates(image, threshold=10):
    """
    Encontra as coordenadas de recorte baseadas nas áreas não pretas
    Retorna: (x, y, w, h) para recorte
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Aplicar threshold para identificar áreas não pretas
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Se não encontrar contornos, retorna a imagem inteira
        return 0, 0, image.shape[1], image.shape[0]
    
    # Encontrar o maior contorno
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Adicionar pequena margem de segurança
    margin = 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)
    
    return x, y, w, h


def consistent_crop_pair(imgGt, imgX):
    """
    Recorta consistentemente o par ground-truth e fake
    baseado nas áreas não pretas da ground-truth
    """
    # Encontrar coordenadas de recorte na ground-truth
    x, y, w, h = find_crop_coordinates(imgGt)
    
    # Aplicar o mesmo recorte em ambas as imagens
    cropped_gt = imgGt[y:y+h, x:x+w]
    cropped_x = imgX[y:y+h, x:x+w]
    
    return cropped_gt, cropped_x


# Versão para processamento em lote
def batch_consistent_crop(gt_images, x_images):
    """
    Processa múltiplos pares de imagens
    gt_images: lista de imagens ground-truth
    x_images: lista de imagens fake correspondentes
    """
    cropped_pairs = []
    
    for imgGt, imgX in zip(gt_images, x_images):
        # Verificar se as imagens têm o mesmo tamanho
        if imgGt.shape != imgX.shape:
            raise ValueError("As imagens devem ter o mesmo tamanho")
        
        cropped_gt, cropped_x = consistent_crop_pair(imgGt, imgX)
        cropped_pairs.append((cropped_gt, cropped_x))
    
    return cropped_pairs


def get_crop_coordinates_from_gt(imgGt):
    """
    Retorna apenas as coordenadas de recorte da ground-truth
    Útil para aplicar o mesmo recorte posteriormente
    """
    return find_crop_coordinates(imgGt)


def apply_crop_coordinates(image, crop_coords):
    """
    Aplica coordenadas de recorte pré-calculadas
    """
    x, y, w, h = crop_coords
    return image[y:y+h, x:x+w]


def robust_consistent_crop(imgGt, imgX, min_area_ratio=0.1):
    """
    Versão robusta com verificação da qualidade do recorte
    """
    # Encontrar coordenadas
    x, y, w, h = find_crop_coordinates(imgGt)
    
    # Verificar se o recorte é muito pequeno (possível erro)
    total_area = imgGt.shape[0] * imgGt.shape[1]
    crop_area = w * h
    
    if crop_area < total_area * min_area_ratio:
        print("Aviso: Área recortada muito pequena, usando imagem completa")
        return imgGt, imgX
    
    # Aplicar recorte
    cropped_gt = imgGt[y:y+h, x:x+w]
    cropped_x = imgX[y:y+h, x:x+w]
    
    return cropped_gt, cropped_x


def padding_functions(img, min_ratio, max_ratio, color=(0, 0, 0)):
	img_h, img_w = np.shape(img)[:2]

	border_w = 0
	border_h = 0
	ar = float(img_w)/img_h

	if ar >= min_ratio and ar <= max_ratio:
		return img, border_w, border_h

	if ar < min_ratio:
		while ar < min_ratio:
			border_w += 1
			ar = float(img_w+border_w)/(img_h+border_h)
	else:
		while ar > max_ratio:
			border_h += 1
			ar = float(img_w)/(img_h+border_h)

	border_w = border_w//2
	border_h = border_h//2

	img = cv2.copyMakeBorder(img, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value = color)
	return img, border_w, border_h


def load_model_functions(path):
    with open(path + '/model.json', 'r') as f:
        json = f.read()

    with tf.device('/CPU:0'):
        with tf.keras.utils.custom_object_scope({'Model': tf.keras.Model}):
            model = model_from_json(json)
        model.load_weights(path + '/weights.hdf5')

    return model


def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, epoch, history_loss, filename='mycheckpoint.pth.tar'):
    print('-> Saving checkpoint')
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history_loss': history_loss
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print('-> Loading checkpoint')
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    last_epoch = checkpoint['epoch']
    history_loss = checkpoint['history_loss']

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return model, optimizer, last_epoch, history_loss


def plot_examples(low_res_folder, gen, epoch):
    files = os.listdir(low_res_folder)
    files.sort()

    gen.eval()
    for count, file in enumerate(files):
        img = cv2.imread(
            os.path.join(low_res_folder, file),
            cv2.IMREAD_COLOR
        ).astype(np.float32) / 255.
        img = torch.from_numpy(
            np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
        ).float()
        img = img.unsqueeze(0).to(config.DEVICE)

        with torch.inference_mode():
            hr_image = gen(img)

        hr_image = hr_image.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        hr_image = np.transpose(hr_image[[2, 1, 0], :, :], (1, 2, 0))
        hr_image = (hr_image * 255.0).round().astype(np.uint8)
        os.makedirs(os.path.join(config.EXPERIMENT_FOLDER, f'epoch_{epoch}'), exist_ok=True)
        cv2.imwrite(os.path.join(config.EXPERIMENT_FOLDER, f'epoch_{epoch}', file), hr_image)

        if count == config.MAX_IMAGES_EXAMPLES - 1:
            break
    gen.train()
