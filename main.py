import os
import sys
import nltk
import torch

import numpy as np
import torch.utils.data as data

from model import EncoderCNN, DecoderRNN
from data_loader import get_loader
from torchvision import transforms
from pycocotools.coco import COCO
from collections import Counter

nltk.download('punkt')
sys.path.append('/opt/cocoapi/PythonAPI')


def coco_api_init():
    # initialize COCO API for instance annotations
    dataDir = '/opt/cocoapi'
    dataType = 'val2014'
    instances_annFile = os.path.join(dataDir, 'annotations/instances_{}.json'.format(dataType))
    coco = COCO(instances_annFile)

    # initialize COCO API for caption annotations
    captions_annFile = os.path.join(dataDir, 'annotations/captions_{}.json'.format(dataType))
    coco_caps = COCO(captions_annFile)

    # get image ids
    ids = list(coco.anns.keys())


def initialize_data_loader():
    # Define a transform to pre-process the training images.
    transform_train = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(224),  # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])

    # Set the minimum word count threshold.
    vocab_threshold = 5

    # Specify the batch size.
    batch_size = 10

    # Obtain the data loader.
    data_loader = get_loader(transform=transform_train,
                             mode='train',
                             batch_size=batch_size,
                             vocab_threshold=vocab_threshold,
                             vocab_from_file=False)

    counter = Counter(data_loader.dataset.caption_lengths)
    lengths = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
    indices = data_loader.dataset.get_train_indices()
    new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
    data_loader.batch_sampler.sampler = new_sampler

    # # Obtain the batch.
    # images, captions = next(iter(data_loader))
    # return images, captions


def experiment_with_cnn():
    # Specify the dimensionality of the image embedding.
    embed_size = 256

    # -#-#-# Do NOT modify the code below this line. #-#-#-#

    # Initialize the encoder. (Optional: Add additional arguments if necessary.)
    encoder = EncoderCNN(embed_size)

    # Move the encoder to GPU if CUDA is available.
    encoder.to(device)

    # Move last batch of images (from Step 2) to GPU if CUDA is available.
    images = images.to(device)

    # Pass the images through the encoder.
    features = encoder(images)

    print('type(features):', type(features))
    print('features.shape:', features.shape)

    # Check that your encoder satisfies some requirements of the project! :D
    assert type(features) == torch.Tensor, "Encoder output needs to be a PyTorch Tensor."
    assert (features.shape[0] == batch_size) & (
                features.shape[1] == embed_size), "The shape of the encoder output is incorrect."


if __name__ == '__main__':


    # for value, count in lengths:
    #     print('value: %2d --- count: %5d' % (value, count))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # coco_api_init()
    initialize_data_loader