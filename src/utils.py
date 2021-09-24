import get_saliency
import get_gbp
import get_CAM

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import cv2
from io import BytesIO
from PIL import Image
import requests


def visualize(plots, top_pred, top_prob):
    """Plots the saved mappings
    Args:
        plots: dictionary of saved mappings
        top_pred: top prediction by index
        top_prob: probability of top prediction

    Plots from left to right:
        1) Input image with top predicted label and probability
        2) Saliency Map
        3) Guided Backpropagation Map
        4) Grad-CAM
    """
    # If only wanted to see image with prediction and no other maps
    if len(plots) == 1:
        plt.imshow(plots['Prediction'])
        plt.title(f'Top Prediction: {top_pred} ({100 * top_prob:.2f}%)', fontsize=15)
        plt.axis('off')
        plt.show()
        return

    fig, ax = plt.subplots(1, len(plots), figsize=(25,5), dpi=60)
    for idx, (key, value) in enumerate(plots.items()):

        # Gray cmap only for saliency map
        if key == 'Saliency Map':
            ax[idx].imshow(value, cmap='gray')
        else:
            ax[idx].imshow(value)

        # Specific title only for 'Prediction'
        if key == 'Prediction':
            ax[idx].set_title(f'Top Prediction: {top_pred} ({100 * top_prob:.2f}%)', fontsize=20)
        else:
            ax[idx].set_title(key, fontsize=20)

        ax[idx].axis('off')

    plt.tight_layout()
    plt.show()


def transform(image, device):
    """Transforms image
    Args:
        image: input image
        device: cuda or cpu

    Returns:
        transformed_image: image to be passed into model
    """
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # Unsqueeze to be used as 1 batch
    transformed_image = tfms(image).to(device).unsqueeze(0)
    return transformed_image


feature_maps = []
def hook_feature(module, input, output):
    feature_maps.append(output.data)


def predict(model, url, saliency, guided_bp, grad_cam, device):
    """ Predicts image label and visualize various maps
    Args:
        model: ResNet model to be used
        url: Image address
        saliency (bool): If True, acquire saliency map
        guided_bp (bool): If True, acquire guided backprop map
        grad_cam (bool): If True, acquire Grad-CAM
        device: cuda or cpu
    """
    classes = ['Chimpanzee', 'Gibbon', 'Gorilla', 'Human', 'Orangutan']

    # Read image from given url link
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
    except OSError as e:
        print(f'Error: {e}')
        print(f'Try another image.')
        exit()

    # Transform image
    img_tensor = transform(image, device)
    # For saliency, for finding the gradient with respect to the input image
    img_tensor.requires_grad_()
    # Put model to evaluation mode
    model.eval()

    # For Grad-CAM, hook last convolutional block 'layer4'
    if grad_cam:
         model.model._modules.get('layer4').register_forward_hook(hook_feature)

    output = model(img_tensor)

    # Get logits and indexes in descending order
    logits, idx = output.sort(1, True)
    probs = F.softmax(logits, dim=1)
    probs = probs.flatten().data
    idx_flattened = idx.flatten().data.cpu()

    # Dictionary to store maps
    plots = {}
    # Input image with prediction is first by default
    plots['Prediction'] = image

    if saliency:
        print(f'Getting Saliency Map...')
        output_max = logits[0][0]
        # Get saliency map
        saliency_map = get_saliency.generate(model, img_tensor, output_max)
        # Resize to input image size
        saliency_map = cv2.resize(saliency_map, (image.size[0], image.size[1]))
        # Store saliency map in dictionary
        plots['Saliency Map'] = saliency_map

    if guided_bp:
        print(f'Getting guided backprop map...')
        # Extract first convolutional layer
        layer = model.model.conv1
        GBP = get_gbp.GuidedBackprop(model, layer)
        # Generate gradient map
        guided_grads = GBP.generate(img_tensor, idx_flattened[0], device)
        # Normalize
        guided_grads = guided_grads - np.min(guided_grads)
        guided_grads = guided_grads / np.max(guided_grads)
        # Resize to input image size
        guided_grads = cv2.resize(guided_grads,(image.size[0], image.size[1]))
        # Store guided backprop map in dictionary
        plots['Guided Backpropagation'] = guided_grads
    
    if grad_cam:
        print(f'Getting Grad-CAM...')
        # Generate superimposed image of Grad-CAM and input image
        superimposed_img = get_CAM.generate(image, model, idx_flattened[0], feature_maps, device)
        # Store Grad-CAM in dictionary
        plots['Grad-CAM'] = superimposed_img

    top_pred = classes[idx_flattened[0]]
    top_prob = probs[0]
    visualize(plots, top_pred, top_prob)
