import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
import torchtext
import argparse
from datetime import datetime

from train import train, validate
from models import Encoder, Decoder, Discriminator
from utils import tokenizer, print_sample, print_decoded
from dataset import Multi30KEminem

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RCAAE')
    parser.add_argument('--num-epochs', type=int, default=1, metavar='NE',
                        help='num epochs (default: 1)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size (default: 64)')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='DR',
                        help='dropout (default: 0.0)')
    parser.add_argument('--hidden-size', type=int, default=100, metavar='HS',
                        help='LSTM hidden size (default: 100)')
    parser.add_argument('--seed', type=int, default=42, metavar='SEED',
                        help='seed (default: 42)')
    parser.add_argument('--embeddings-size', type=int, default=50, metavar='ES',
                        help='embeddings size (default: 50)')
    parser.add_argument('--vectors', type=str, default='glove.6B.50d', metavar='V',
                        help='embeddings size (default: glove.6B.50d)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    Multi30KEminem.download('.')

    TEXT = torchtext.data.Field(eos_token='<eos>',
                                init_token='<sos>',
                                tokenize=tokenizer,
                                lower=True)

    LABEL = torchtext.data.Field(sequential=False, unk_token=None)

    train_ds, valid_ds, _ = Multi30KEminem.splits(TEXT, LABEL, '.')

    print(f"Train DS size: {len(train_ds)}, Valid DS size: {len(valid_ds)}")

    TEXT.build_vocab(train_ds)
    LABEL.build_vocab(train_ds)

    vocab_size = len(TEXT.vocab)

    label_size = len(LABEL.vocab)

    print(f"Vocab size: {vocab_size}, num classes: {label_size}")

    train_dl = torchtext.data.Iterator(train_ds, args.batch_size, repeat=False, shuffle=False)
    valid_dl = torchtext.data.Iterator(valid_ds, args.batch_size, repeat=False)
    gen_dl = torchtext.data.Iterator(train_ds, 1, repeat=False)

    if args.vectors:
        TEXT.vocab.load_vectors(args.vectors)

    prior_size = 2 * args.hidden_size

    enc = Encoder(vocab_size, args.embeddings_size, args.hidden_size, n_layers=1,
                  dropout=args.dropout, lr=args.learning_rate, vectors=TEXT.vocab.vectors)

    dec = Decoder(vocab_size, args.embeddings_size, args.hidden_size, prior_size + label_size, n_layers=1,
                  dropout=args.dropout, lr=args.learning_rate, vectors=TEXT.vocab.vectors)

    disc = Discriminator([prior_size + label_size, args.hidden_size, 1],
                         dropout=0.3, lr=args.learning_rate, activation_fn=nn.LeakyReLU(0.2))

    print("========== Encoder ==========\n{}".format(enc))

    print("========== Decoder ==========\n{}".format(dec))

    print("========== Discriminator ==========\n{}".format(disc))

    for epoch in range(1, args.num_epochs+1):
        print("========== Start epoch {} at {} ==========".format(epoch, datetime.now().strftime("%H:%M:%S")))
        train(epoch, enc, dec, disc, prior_size, train_dl, TEXT.vocab)
        validate(epoch, enc, dec, disc, prior_size, valid_dl, TEXT.vocab)
        print_decoded(enc, dec, gen_dl, TEXT.vocab)
        print_sample(dec, sample_size=prior_size, max_seq_len=41, vocab=TEXT.vocab, style_vocab=LABEL.vocab)
