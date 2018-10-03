import numpy as np
import torch
import torch.nn.functional as F

from datetime import datetime

from utils import to_onehot, seq_to_str
from bleu import moses_multi_bleu


def _train(epoch, enc, dec, disc, prior_size, dl, vocab, mode='Train'):
    if mode == 'Train':
        enc.train()
        dec.train()
        disc.train()
    else:
        enc.eval()
        dec.eval()
        disc.eval()

    g_loss = 0.0
    ae_loss = 0.0
    disc_loss = 0.0

    strs = []
    dec_strs = []

    n_batches = len(dl)

    for batch_idx, batch in enumerate(dl):

        seq = batch.text
        seq = seq[1:]

        label = batch.label
        label = to_onehot(label, 2)

        (seq_len, batch_size) = seq.shape

        batch_zeros = torch.zeros((batch_size, 1))
        batch_ones = torch.ones((batch_size, 1))

        # ======== train/validate Discriminator ========

        if mode == 'Train':
            enc.zero_grad()
            disc.zero_grad()

        z = torch.randn((batch_size, prior_size))
        z_label = to_onehot(torch.tensor(np.random.randint(0, 2, (batch_size))), 2)

        latent = enc(seq)
        fake_pred = disc(latent, label)
        true_pred = disc(z, z_label)

        fake_loss = F.binary_cross_entropy_with_logits(fake_pred, batch_zeros)
        true_loss = F.binary_cross_entropy_with_logits(true_pred, batch_ones)

        disc_loss = 0.5 * (fake_loss + true_loss)

        if mode == 'Train':
            disc_loss.backward()
            disc.optim.step()

        # ======== train/validate Autoencoder ========

        if mode == 'Train':
            enc.zero_grad()
            dec.zero_grad()
            disc.zero_grad()

        z = torch.randn((batch_size, prior_size))

        latent = enc(seq)
        x = torch.zeros(1, batch_size).long() + vocab.stoi['<sos>']

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

        if mode == 'Train':
            g_loss.backward()
            dec.optim.step()
            enc.optim.step()

        # ----------------------------------------------------

        g_loss += g_loss.item()
        ae_loss += ae_loss.item()
        disc_loss += disc_loss.item()

        _, w_idxs = output.topk(1, dim=1)
        dec_seq = torch.tensor(w_idxs.view(seq_len, batch_size))

        strs.extend(seq_to_str(seq.detach(), vocab))
        dec_strs.extend(seq_to_str(dec_seq.detach(), vocab))

    g_loss /= n_batches
    ae_loss /= n_batches
    disc_loss /= n_batches

    bleu = moses_multi_bleu(np.array(dec_strs), np.array(strs))

    print("Epoch {:3} {:5}: BLEU: {:.2f}, AE: {:.5f}, G: {:.5f}, D: {:.5f} at {}".format(
        epoch, mode, bleu, ae_loss, g_loss, disc_loss, datetime.now().strftime("%H:%M:%S")))

    return ae_loss, g_loss, disc_loss, bleu


def train(epoch, enc, dec, disc, prior_size, dl, vocab):
    _train(epoch, enc, dec, disc, prior_size, dl, vocab, mode='Train')


def validate(epoch, enc, dec, disc, prior_size, dl, vocab):
    _train(epoch, enc, dec, disc, prior_size, dl, vocab, mode='Valid')
