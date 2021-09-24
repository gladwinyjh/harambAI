import utils

import torch
import torch.nn as nn
from torchvision import models

import argparse
import time

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        self.model = models.resnet50()
        self.model.fc = nn.Linear(2048, 5)

        # Replace ReLU layers with inplace=False ones
        # because pytorch doesnt allow inplace full backward hooks
        for name, module in self.model.named_modules():
            if hasattr(module, 'relu'):
                module.relu = nn.ReLU(inplace=False) 
    
    def forward(self, x):
        x = self.model(x)
        return x


def get_device(cuda):
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Using Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Using Device: CPU")
    return device


def load_model(path, device):
    model = NeuralNet().to(device)
    try:
        model.load_state_dict(torch.load(path))
    except:
        print(f'Failed to load weights. Check if model weights are in same directory as main.py.')
        exit()
    return model


def main():
    parser = argparse.ArgumentParser(description='harambAI ape classifier')
    parser.add_argument('-u', '--url', type=str, metavar='URL', required=True,
                        help='Url image address of ape')

    parser.add_argument('-s', '--saliency', action='store_true', default=False,
                        help='Produce saliency map, defaults to False')

    parser.add_argument('-gbp', '--guided_backprop', action='store_true', default=False,
                        help='Produce guided backpropagation image, defaults to False.')

    parser.add_argument('-gc', '--grad_cam', action='store_true', default=False,
                        help='Produce grad-cam superimposed image, defaults to False.')

    parser.add_argument('-cuda', '--cuda', action='store_true', default=False,
                        help='Uses cuda if available, defaults False.')

    args = parser.parse_args()
    device = get_device(args.cuda)
    print(f'Loading model...')
    # Load weights
    path = 'weights.pth'
    model = load_model(path, device)
    print(f'Model loaded!')
    print(f'Predicting ape...')
    utils.predict(model, args.url, args.saliency, args.guided_backprop, args.grad_cam, device)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    time_elapsed = end - start
    print('Runtime: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))