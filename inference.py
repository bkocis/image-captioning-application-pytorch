import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.model import EncoderCNN, DecoderRNN

from pycocotools.coco import COCO
from utils.data_loader import get_loader
from torchvision import transforms

transform_test = transforms.Compose([
                            transforms.Resize(256),                          # smaller edge of image resized to 256
                            transforms.RandomCrop(224),                      # get 224x224 crop from random location
                            transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
                            transforms.ToTensor(),                           # convert the PIL Image to a tensor
                            transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                                 (0.229, 0.224, 0.225))])

data_loader = get_loader(transform=transform_test,
                         mode='test')
orig_image, image = next(iter(data_loader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_file = "encoder_n2-3.pkl"
decoder_file = "decoder_n2-3.pkl"

embed_size = 300
hidden_size = 300

vocab_size = len(data_loader.dataset.vocab)

encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))

encoder.to(device)
decoder.to(device)



# Move image Pytorch Tensor to GPU if CUDA is available.
image = image.to(device)

# Obtain the embedded image features.
features = encoder(image).unsqueeze(1)

# Pass the embedded image features through the model to get a predicted caption.
output = decoder.sample(features)
print('example output:', output)


def clean_sentence(output):
    words = [data_loader.dataset.vocab.idx2word[i] for i in output][1:-1]
    sentence = ' '.join(words)
    return sentence


sentence = clean_sentence(output)
print('example sentence:', sentence)


def get_prediction():
    orig_image, image = next(iter(data_loader))
    plt.imshow(np.squeeze(orig_image))
    plt.title('Sample Image')
    plt.show()
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)
    sentence = clean_sentence(output)
    print(sentence)



get_prediction()

