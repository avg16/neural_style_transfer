# Neural Style Transfer

This project implements Neural Style Transfer using a pre-trained VGG-19 model. The model separates and recombines the content and style of arbitrary images to create artistic images of high perceptual quality.

## Introduction

In fine art, especially painting, humans have mastered the skill to create unique visual experiences through composing a complex interplay between the content and style of an image. This project aims to implement an artificial system based on a Deep Neural Network that creates artistic images of high perceptual quality. The system uses neural representations to separate and recombine content and style of arbitrary images, providing a neural algorithm for the creation of artistic images.

## Model Specifications

For this task, we have used the popular pretrained model of VGG Net ([Reference](https://arxiv.org/abs/1409.1556v6)).

### Architecture

- **Layers**: The network consists of 19 layers, including 16 convolutional layers, 3 fully connected layers, 5 Max Pooling layers, and the SoftMax layer.
- **Convolutional Layers**: The convolutional layers use filters with a very small receptive field: 3x3 (smallest size to capture the notion of left/right, up/down, centre).
- **Pooling Layers**: Max pooling is performed over a 2x2 pixel window, with a stride of 2.
- **Fully Connected Layers**: There are three fully connected layers. The first two have 4096 channels each, and the third has 1000 channels, one for each class in the ImageNet dataset.

### Pretrained Model

- **Training Data**: The pretrained model is trained on the ImageNet dataset, which consists of over 14 million images and 1000 classes.
- **Transfer Learning**: VGG-19 is often used for transfer learning. The pretrained weights can be fine-tuned on a new dataset, which allows the model to leverage the features learned on ImageNet.
- **Feature Extraction**: The lower layers of VGG-19 are good at detecting edges, textures, patterns, and simple shapes, while the deeper layers detect more complex structures and objects. This makes it a robust feature extractor for various image recognition tasks.

### Usage in Applications

- **Object Recognition**: The model can classify images into one of the 1000 classes in the ImageNet dataset.
- **Image Segmentation**: By using VGG-19 as a backbone in segmentation models, one can leverage its strong feature extraction capabilities.
- **Style Transfer**: VGG-19 is commonly used in style transfer applications due to its ability to extract detailed features and represent different styles.

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- PIL
- matplotlib

## Installation

Clone the repository:

```sh
git clone https://github.com/your-username/neural_style_transfer.git
cd neural_style_transfer

