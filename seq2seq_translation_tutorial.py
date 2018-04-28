# coding: utf-8

from evaluate import *
from encoder import *
from attn_decoder import *
from train import *


def train():
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    train_iters(encoder1, attn_decoder1, 75000, print_every=5000)


if __name__ == '__main__':
    encoder1 = torch.load('encoder.pt', map_location=lambda storage, loc: storage)
    attn_decoder1 = torch.load('decoder.pt', map_location=lambda storage, loc: storage)
    evaluateRandomly(encoder1, attn_decoder1)

    output_words, attentions = evaluate(encoder1, attn_decoder1, "je suis trop froid .")


    def evaluate_and_print(input_sentence):
        output_words, attentions = evaluate(
            encoder1, attn_decoder1, input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))


    evaluate_and_print("elle a cinq ans de moins que moi .")
    evaluate_and_print("elle est trop petit .")
    evaluate_and_print("je ne crains pas de mourir .")
    evaluate_and_print("c est un jeune directeur plein de talent .")
