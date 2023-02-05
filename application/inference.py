import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from utils.model import EncoderCNN, DecoderRNN


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

    def load_enc_dec(self, vocab_size):
        encoder = EncoderCNN(self.embed_size)
        encoder.eval()
        decoder = DecoderRNN(self.embed_size, self.hidden_size, vocab_size)
        decoder.eval()

        encoder.load_state_dict(torch.load(os.path.join('models', self.encoder_file), map_location=torch.device('cpu')))
        decoder.load_state_dict(torch.load(os.path.join('models', self.decoder_file), map_location=torch.device('cpu')))

        encoder.to(self.device)
        decoder.to(self.device)
        return encoder, decoder

    def load_data(self):
        with open('./models/idx2word.json') as f:
            idx2word = json.load(f)
        vocab_size = idx2word.__len__()
        return idx2word, vocab_size

    def clean_sentence(self, idx2word, output):
        words = [idx2word.get(str(i)) for i in output[1:-1]]
        sentence = ' '.join(words)
        return sentence

    def get_prediction(self, idx2word, orig_image, image, encoder, decoder):
        image = image.to(self.device)
        # Obtain the embedded image features.
        features = encoder(image).unsqueeze(1)
        # Pass the embedded image features through the model to get a predicted caption.
        output = decoder.sample(features)

        sentence = self.clean_sentence(idx2word, output)

        plt.imshow(np.squeeze(orig_image))
        plt.title(f'{sentence}')
        plt.savefig(f"resources/Sample_image-{sentence}.png")
        return sentence

    def caption_sentence_from_upload(self, image_file):
        # PIL_image = Image.open(image_file).convert('RGB')
        orig_image = np.array(image_file)
        image = self.transform_image(image_file)
        image = image[None, :, :, :]
        idx2word, vocab_size = self.load_data()
        encoder, decoder = self.load_enc_dec(vocab_size)
        sentence = self.get_prediction(idx2word, orig_image, image, encoder, decoder)
        return orig_image, sentence

    def transform_image(self, image):
        transform_img = transforms.Compose([
                transforms.Resize(256),                          # smaller edge of image resized to 256
                transforms.RandomCrop(224),                      # get 224x224 crop from random location
                transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
                transforms.ToTensor(),                           # convert the PIL Image to a tensor
                transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                     (0.229, 0.224, 0.225))])
        return transform_img(image)
