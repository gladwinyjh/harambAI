import torch
import numpy as np


def generate(model, img_tensor, output_max):
    # Do backpropagation to get the derivative of the output based on the image
    output_max.backward()
    saliency, _ = torch.max(img_tensor.grad.data.abs(), dim=1)
    saliency_np = saliency.permute(1,2,0).cpu().numpy()
    return saliency_np