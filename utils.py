import torch
import spacy
from spacy.symbols import ORTH, LEMMA, POS, TAG
import re


def to_onehot(data, n_digits, device):
    d = data.to(device)
    y = torch.zeros(d.shape[0], n_digits).to(device)
    y = y.scatter(1, d[:, None], 1).to(device)
    return y


def seq_to_str(seq, vocab):
    (seq_len, batch_size) = seq.shape
    result = []
    for i in range(batch_size):
        result.append(" ".join([vocab.itos[w_idx.item()] for w_idx in seq[:, i].view(-1)]))
    return result


def decode_z(dec, z, seq_len, label, vocab, device):
    dec.eval()

    (batch_size, hidden_size) = z.shape

    label = to_onehot(label, 2, device)

    x = torch.zeros(1, batch_size).to(device).long() + vocab.stoi['<sos>']
    h = None

    dec_seq = None

    for i in range(seq_len):
        o, h = dec(x, z, h, label)
        _, w_idxs = o.topk(1)
        x = w_idxs.view(1, -1)
        dec_seq = w_idxs if dec_seq is None else torch.cat((dec_seq, w_idxs), 0)
        if batch_size == 1 and vocab.itos[w_idxs.item()] == '<eos>':
            break

    return dec_seq


def print_decoded(enc, dec, dl, vocab, device):
    enc.eval()
    dec.eval()

    b = next(iter(dl))
    seq = b.text
    seq = seq[1:]

    label = b.label

    (seq_len, batch_size) = seq.shape

    z = enc(seq)

    dec_seq = decode_z(dec, z, seq_len, label, vocab, device)

    origin = seq_to_str(seq.detach(), vocab)[0].replace(" <nl> ", "\n\t")
    decoded = seq_to_str(dec_seq.detach(), vocab)[0].replace(" <nl> ", "\n\t")

    print("\nOrigin:\n\t{}".format(origin))
    print("\nDecoded: {}\n\t".format(decoded))


def print_sample(dec, sample_size, max_seq_len, vocab, style_vocab, device):
    dec.eval()

    z = torch.randn(1, sample_size).to(device)
    print("\nRandom sample:")

    label_0 = torch.zeros(1).long().to(device)
    dec_seq = decode_z(dec, z, max_seq_len, label_0, vocab, device)
    seq_0 = seq_to_str(dec_seq.detach(), vocab)[0].replace(" <nl> ", "\n\t")
    print("\nDecoded w. style {}:\n\t{}".format(style_vocab.itos[0], seq_0))

    label_1 = (torch.zeros(1) + 1).long().to(device)
    dec_seq = decode_z(dec, z, max_seq_len, label_1, vocab, device)
    seq_1 = seq_to_str(dec_seq.detach(), vocab)[0].replace(" <nl> ", "\n\t")
    print("\nDecoded w. style {}:\n\t{}\n".format(style_vocab.itos[1], seq_1))


NLP = spacy.load('en')
special_case = [{ORTH: '<nl>'}]
NLP.tokenizer.add_special_case('<nl>', special_case)


def tokenizer(s):
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]

    s = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(s))
    s = re.sub(r"[ ]+", " ", s)
    s = re.sub(r"\!+", "!", s)
    s = re.sub(r"\,+", ",", s)
    s = re.sub(r"\?+", "?", s)

    return [x.text for x in NLP.tokenizer(s) if x.text != " "]