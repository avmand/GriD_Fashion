import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import tqdm
import scipy.ndimage as nd


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def deprocess(image_np):
    image_np = image_np.squeeze().transpose(1, 2, 0)
    image_np = image_np * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
    image_np = np.clip(image_np, 0.0, 255.0)
    return image_np


def clip(image_tensor):
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[0, c] = torch.clamp(image_tensor[0, c], -m / s, (1 - m) / s)
    return image_tensor

def dream(image, model, iterations, lr):
    """ Updates the image to maximize outputs for n iterations """
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor
    image = Variable(Tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        loss = out.norm()
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        image.grad.data.zero_()
    return image.cpu().data.numpy()


def deep_dream(image, model, iterations, lr, octave_scale, num_octaves):
    """ Main deep dream method """
    image = preprocess(image).unsqueeze(0).cpu().data.numpy()

    # Extract image representations for each octave
    octaves = [image]
    for _ in range(num_octaves - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
        if octave > 0:
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        input_image = octave_base + detail
        dreamed_image = dream(input_image, model, iterations, lr)
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)


if __name__ == "__main__":
  
    # Load image
    image_name='images.jpg'  #change the image name here and just upload any image at runtime and run this cell first
    image = Image.open(image_name)
  
    # Define the model
    network = models.vgg16(pretrained=True)
    layers = list(network.features.children())
    model = nn.Sequential(*layers[: (15+ 1)])
    if torch.cuda.is_available:
        model = model.cuda()
    print(network)

    # Extract deep dream image
    dreamed_image = deep_dream(
        image,
        model,
        iterations=20,
        lr=0.01,
        octave_scale=1.4,
        num_octaves=10,
    )

     # Save and plot image
    min_val = np.min(dreamed_image)
    max_val = np.max(dreamed_image)
    img_data_clamped = (dreamed_image - min_val) / (max_val - min_val) 
    os.makedirs("outputs", exist_ok=True)
    filename = image_name.split("/")[-1]
    plt.figure(figsize=(20, 20))
    plt.imshow(img_data_clamped)
    plt.imsave(f"outputs/output_{filename}", img_data_clamped)
    plt.show()
