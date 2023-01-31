import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.model import EncoderCNN, DecoderRNN

from utils.data_loader import get_loader
from torchvision import transforms


class InferenceOnSingleImage:

    encoder_file = "encoder_n2-3.pkl"
    decoder_file = "decoder_n2-3.pkl"
    embed_size = 300
    hidden_size = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __int__(self, encoder_file=None, decoder_file=None, embed_size=None, hidden_size=None, device=None):

        self.encoder_file = encoder_file
        self.decoder_file = decoder_file
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.device = device

    # def transform_test(self):
    #     transform_test = transforms.Compose([
    #                                 transforms.Resize(256),                          # smaller edge of image resized to 256
    #                                 transforms.RandomCrop(224),                      # get 224x224 crop from random location
    #                                 transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    #                                 transforms.ToTensor(),                           # convert the PIL Image to a tensor
    #                                 transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
    #                                                      (0.229, 0.224, 0.225))])
    #     return transform_test

    def load_enc_dec(self, vocab_size):
        encoder = EncoderCNN(self.embed_size)
        encoder.eval()
        decoder = DecoderRNN(self.embed_size, self.hidden_size, vocab_size)
        decoder.eval()

        encoder.load_state_dict(torch.load(os.path.join('./models', self.encoder_file)))
        decoder.load_state_dict(torch.load(os.path.join('./models', self.decoder_file)))

        encoder.to(self.device)
        decoder.to(self.device)
        return encoder, decoder

    def load_data(self):
        transform_test = transforms.Compose([
                                    transforms.Resize(256),                          # smaller edge of image resized to 256
                                    transforms.RandomCrop(224),                      # get 224x224 crop from random location
                                    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
                                    transforms.ToTensor(),                           # convert the PIL Image to a tensor
                                    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                                         (0.229, 0.224, 0.225))])
        data_loader = get_loader(transform=transform_test, mode='test')
        vocab_size = len(data_loader.dataset.vocab)
        orig_image, image = next(iter(data_loader))
        return data_loader, orig_image, image, vocab_size

    def clean_sentence(self, data_loader, output):
        words = [data_loader.dataset.vocab.idx2word[i] for i in output][1:-1]
        sentence = ' '.join(words)
        return sentence

    def get_prediction(self, data_loader, orig_image, image, encoder, decoder):

        plt.imshow(np.squeeze(orig_image))
        plt.title('Sample Image')
        # plt.show()
        image = image.to(self.device)

        # Obtain the embedded image features.
        features = encoder(image).unsqueeze(1)

        # Pass the embedded image features through the model to get a predicted caption.
        output = decoder.sample(features)

        sentence = self.clean_sentence(data_loader, output)
        plt.savefig(f"resources/Sample_image-{sentence}.png")
        return sentence

    def caption_sentence(self):
        data_loader, orig_image, image, vocab_size = self.load_data()
        encoder, decoder = self.load_enc_dec(vocab_size)
        self.get_prediction(data_loader, orig_image, image, encoder, decoder)
