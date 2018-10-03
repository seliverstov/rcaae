import torchtext.data as data


class Multi30KEminem(data.Dataset):
    urls = ['http://files.deeppavlov.ai/datasets/multi30k_eminem.4l.zip']
    name = 'multi30k_eminem_4l'
    dirname = ''

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        with open(path) as f:
            for line in f.readlines():
                line = line.strip()
                label = line[-1]
                text = line[:-2]
                examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root='.data',
               train='multi30k_eminem.4l.train.txt',
               validation='multi30k_eminem.4l.test.txt',
               test='multi30k_eminem.4l.test.txt', **kwargs):
        return super().splits(
            root=root, text_field=text_field, label_field=label_field,
            train=train, validation=validation, test=test, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.data', vectors=None, **kwargs):
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, valid, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, test), batch_size=batch_size, device=device)

