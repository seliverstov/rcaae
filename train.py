from datetime import datetime
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.iterator import Iterator
from torchtext.vocab import Vocab


from utils import to_onehot, seq_to_str
from bleu import moses_multi_bleu


def _train(epoch: int, enc: nn.Module, dec: nn.Module, disc: nn.Module, prior_size: int,
           dl: Iterator, vocab: Vocab, device: str, validate: bool = False) -> Tuple[float, float, float, float]:

    if not validate:
        enc.train()
        dec.train()
        disc.train()
    else:
        enc.eval()
        dec.eval()
        disc.eval()

    epoch_g_loss = 0.0
    epoch_ae_loss = 0.0
    epoch_disc_loss = 0.0

    strs = []
    dec_strs = []

    n_batches = len(dl)

    for batch_idx, batch in enumerate(dl):

        seq = batch.text
        seq = seq[1:]

        label = batch.label
        label = to_onehot(label, 2, device)

        (seq_len, batch_size) = seq.shape

        batch_zeros = torch.zeros((batch_size, 1)).to(device)
        batch_ones = torch.ones((batch_size, 1)).to(device)

        # ======== train/validate Discriminator ========

        if not validate:
            enc.zero_grad()
            disc.zero_grad()

        z = torch.randn((batch_size, prior_size)).to(device)
        z_label = to_onehot(torch.randint(0, 2, (batch_size, )).long(), 2, device)

        latent = enc(seq)
        fake_pred = disc(latent, label)
        true_pred = disc(z, z_label)

        fake_loss = F.binary_cross_entropy_with_logits(fake_pred, batch_zeros)
        true_loss = F.binary_cross_entropy_with_logits(true_pred, batch_ones)

        disc_loss = 0.5 * (fake_loss + true_loss)

        if not validate:
            disc_loss.backward()
            disc.optim.step()

        # ======== train/validate Autoencoder ========

        if not validate:
            enc.zero_grad()
            dec.zero_grad()
            disc.zero_grad()

        latent = enc(seq)
        x = torch.zeros(1, batch_size).to(device).long() + vocab.stoi['<sos>']

        h = None

        output = None

        for i in range(seq_len):
            o, h = dec(x, latent, h, label)
            x = seq[i].view(1, -1)
            output = o if output is None else torch.cat((output, o), 0)

        ae_loss = F.nll_loss(output, seq.view(-1))

        fake_pred_z = disc(latent, label)

        enc_loss = F.binary_cross_entropy_with_logits(fake_pred_z, batch_ones)

        g_loss = ae_loss + enc_loss

        if not validate:
            g_loss.backward()
            dec.optim.step()
            enc.optim.step()

        # ----------------------------------------------------

        epoch_g_loss += g_loss.item()
        epoch_ae_loss += ae_loss.item()
        epoch_disc_loss += disc_loss.item()

        _, w_idxs = output.topk(1, dim=1)
        dec_seq = w_idxs.view(seq_len, batch_size)

        strs.extend(seq_to_str(seq.detach(), vocab))
        dec_strs.extend(seq_to_str(dec_seq.detach(), vocab))

    epoch_g_loss /= n_batches
    epoch_ae_loss /= n_batches
    epoch_disc_loss /= n_batches

    bleu = moses_multi_bleu(np.array(dec_strs), np.array(strs))

    mode = 'Valid' if validate else 'Train'

    print("Epoch {:3} {:5}: BLEU: {:.2f}, AE: {:.5f}, G: {:.5f}, D: {:.5f} at {}".format(
        epoch, mode, bleu, epoch_ae_loss, epoch_g_loss, epoch_disc_loss, datetime.now().strftime("%H:%M:%S")))

    return epoch_ae_loss, epoch_g_loss, epoch_disc_loss, bleu


def train(epoch: int, enc: nn.Module, dec: nn.Module, disc: nn.Module, prior_size: int,
          dl: Iterator, vocab: Vocab, device: str) -> Tuple[float, float, float, float]:

    return _train(epoch, enc, dec, disc, prior_size, dl, vocab, device, validate=False)


def validate(epoch: int, enc: nn.Module, dec: nn.Module, disc: nn.Module, prior_size: int,
             dl: Iterator, vocab: Vocab, device: str) -> Tuple[float, float, float, float]:

    return _train(epoch, enc, dec, disc, prior_size, dl, vocab, device, validate=True)
