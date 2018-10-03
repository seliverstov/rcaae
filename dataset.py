from typing import Tuple
from torch import Tensor
from torchtext.data.dataset import Dataset
from torchtext.data.example import Example
from torchtext.data.field import Field
from torchtext.data.iterator import Iterator, BucketIterator


class Multi30KEminem(Dataset):

    urls = ['http://files.deeppavlov.ai/datasets/multi30k_eminem.4l.zip']
    name = 'multi30k_eminem_4l'
    dirname = ''

    @staticmethod
    def sort_key(example: Example) -> int:
        return len(example.text)

    def __init__(self, path: str, text_field: Field, label_field: Field, **kwargs) -> None:
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        with open(path) as f:
            for line in f.readlines():
                line = line.strip()
                label = line[-1]
                text = line[:-2]
                examples.append(Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field: Field, label_field: Field, root: str = '.data',
               train: str = 'multi30k_eminem.4l.train.txt',
               validation: str = 'multi30k_eminem.4l.test.txt',
               test: str = 'multi30k_eminem.4l.test.txt', **kwargs) -> Tuple[Dataset, Dataset, Dataset]:

        return super().splits(
            root=root, text_field=text_field, label_field=label_field,
            train=train, validation=validation, test=test, **kwargs)

    @classmethod
    def iters(cls, batch_size: int =32, device: int = 0, root: str ='.data',
              vectors: Tensor = None, **kwargs) -> Tuple[Iterator, Iterator, Iterator]:

        text = Field()
        label = Field(sequential=False)

        train, valid, test = cls.splits(text, label, root=root, **kwargs)

        text.build_vocab(train, vectors=vectors)
        label.build_vocab(train)

        return BucketIterator.splits(
            (train, test), batch_size=batch_size, device=device)

