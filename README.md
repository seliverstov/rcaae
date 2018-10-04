# Pytorch Recurrent Conditional Adversarial Autoencoder (GAN): Generate Eminem lyrics from continuous space


Based on ideas from Samuel Bowman's [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349#) with additional changes:
* Discriminator instead KL-divergence
* Decoder conditioned on text style:  sample from continuous space can be decoded with style like `Eminem lyrics` or `Plain text`
* use simple BiLSTM for Encoder without Highways/Attention

To train models was used a [special dataset](http://files.deeppavlov.ai/datasets/multi30k_eminem.4l.zip) with text samples of two different styles: small couplets from Eminem lyrics and several small sentences from [Multi30K dataset](https://github.com/multi30k/dataset). 

## Sampling examples
```
Decoded w. style `Eminem lyrics`:
	the morning rain clouds up my window
	and i ca n't see at all
	and even if i could it 'd all be gray
	but your picture on my wall <eos>

Decoded w. style `Plain text`:
	two men are playing professional hockey .
	a man in a blue shirt is fixing a yellow and white speed train .
	a man is standing on a ladder painting bricks . <eos>
```
```
Decoded w. style `Eminem lyrics`:
	and i do n't even know you slim ,
	i 'm not a little skeptical who i hang up this
	when i 'm gone , i 'm going back on the mall
	i wanna leave the show to

Decoded w. style `Plain text`:
	two men are playing hockey , one is singing karaoke .
	a man in a blue shirt is playing a keyboard and singing into a microphone .
	a man in a black shirt is playing a trumpet . <eos>
```
  
## Usage

To train model just run

```sh
python main.py
```

### Parameters
* `--num-epochs` default: 100
* `--batch-size` default: 64
* `--learning-rate` default: 0.0001
* `--dropout` default: 0.3
* `--hidden-size` LSTM hidden size, default: 500
* `--seed'` default: 42
* `--embeddings-size` default: 300
* `--vectors` pretrained word vectors, default='fasttext.en.300d' (vectors loaded automatically by [torchtext](https://torchtext.readthedocs.io/en/latest/) library)
* `--cuda` CUDA device numer, default: 0

