import torch
import torch.nn as nn
import numpy as np


def get_children(model: torch.nn.Module):
    """ 
    Function to extract all layers of model to list 
    """
    modules = list(model.children())
    layers = []
    if not modules:
        return model
    else:
       for child in modules:
            try:
                layers.extend(get_children(child))
            except TypeError:
                layers.append(get_children(child))
    return layers


"""
Original Author:
@author: Utku Ozbulak - github.com/utkuozbulak
Copyright (c) 2017 Utku Ozbulak
Slightly modified GuidedBackprop code from https://github.com/utkuozbulak/pytorch-cnn-visualizations

Modifications:
    - Added more comments for clarity
    - Changed method to locate ReLU layers to be compatible for this model
    - register_backward_hook changed to register_full_backward_hook for torch 1.9.1
"""
class GuidedBackprop():
    """
    Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers(self.layer)
        

    def hook_layers(self, layer):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            
        # Register hook to the first layer
        layer.register_full_backward_hook(hook_function)
        

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            corresponding_forward_output = self.forward_relu_outputs[-1]
            # Backprop ReLU
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            # Set all negative gradients to 0
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            # Remove last forward output
            del self.forward_relu_outputs[-1]  
            return (modified_grad_out,)
            
        
        def relu_forward_hook_function(module, input, output):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(output)


        # Loop through layers, hook up ReLUs
        for layer in get_children(self.model):
            if isinstance(layer, nn.ReLU):
                layer.register_full_backward_hook(relu_backward_hook_function)
                layer.register_forward_hook(relu_forward_hook_function)
                

    def generate(self, input_image, target_class, device):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().to(device)
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        return gradients_as_arr.transpose(1,2,0)