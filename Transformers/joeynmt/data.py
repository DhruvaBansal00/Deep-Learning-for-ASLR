# coding: utf-8
"""
Data module
"""
import sys
import random
import os
import os.path
from typing import Optional
import tqdm

from torchtext.datasets import TranslationDataset
import torch
from torchtext import data
from torchtext.data import Dataset, Iterator, Field

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN, SRC_PAD_TOKEN
from joeynmt.vocabulary import build_vocab, Vocabulary

def label_to_int(labels, vocab):
    int_labels = []
    for label in labels:
        currLabel = [vocab.stoi[word] for word in label]
        int_labels.append(currLabel)
    return int_labels



def get_dataset(file_list: str, max_src_length: int, max_trg_length: int) -> object:
    curr_dataset_floats = []
    curr_dataset_labels = []
    fileList = open(file_list)
    for arkFile in tqdm.tqdm(fileList):
        content = []
        currFile = open(arkFile.strip("\n"))
        for frame in currFile:
            line = frame
            if "[" in frame:
                line = frame.split("[ ")[1]
            elif "]" in frame:
                line = frame.split("]")[0]
            features = []
            line = line.strip("\n").split(" ")
            for f in line:
                try:
                    features.append(float(f))
                except:
                    pass
            if len(features) != 0:
                content.append(features)
        if len(content) <= max_src_length:
            
            while len(content) < max_src_length:
                features = [SRC_PAD_TOKEN for i in range(len(content[1]))]
                content.append(features)

            curr_dataset_floats.append(content)
            curr_label = [BOS_TOKEN]
            curr_label.extend(arkFile.split("/")[-1].split(".")[1].strip("\n").split("_"))
            curr_label.append(EOS_TOKEN)

            while len(curr_label) < max_trg_length:
                curr_label.append(PAD_TOKEN)
            curr_dataset_labels.append(curr_label)
    
    return (curr_dataset_floats, curr_dataset_labels)

def load_data(data_cfg: dict, get_test: bool = True, trg_vocab: object = None) -> (object, object, Optional[object], Vocabulary):
    src_lang = data_cfg["src"]
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg.get("test", None)
    max_src_length = data_cfg["max_src_length"]
    max_trg_length = data_cfg["max_trg_length"]

    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    trg_vocab_file = data_cfg.get("trg_vocab", None)
    train_data = None
    dev_data = None
    test_data = None

    if trg_vocab is None:
        print(f'Getting train data...')
        train_data = get_dataset(train_path+"."+src_lang, max_src_length, max_trg_length)

        trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq,
                                max_size=trg_max_size,
                                dataset=train_data, vocab_file=trg_vocab_file)
    
        print(f'Getting dev data...')
        dev_data = get_dataset(dev_path+"."+src_lang, max_src_length, max_trg_length)
        
        print(f'Generating train labels')
        train_data_labels = label_to_int(train_data[1], trg_vocab)
        train_data = (train_data[0], train_data_labels)

        print(f'Generating test labels')
        dev_data_labels = label_to_int(dev_data[1], trg_vocab)
        dev_data = (dev_data[0], dev_data_labels)
    
    if test_path is not None and get_test:
        print(f'Getting test data...')
        test_data = get_dataset(test_path+"."+src_lang, max_src_length, max_trg_length)
        test_data_labels = label_to_int(test_data[1], trg_vocab)
        test_data = (test_data[0], test_data_labels)
    

    return train_data, dev_data, test_data, trg_vocab
    



def old_load_data(data_cfg: dict) -> (Dataset, Dataset, Optional[Dataset],
                                  Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """
    # load data from files
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg.get("test", None)
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)

    train_data = TranslationDataset(path=train_path,
                                    exts=("." + src_lang, "." + trg_lang),
                                    fields=(src_field, trg_field),
                                    filter_pred=
                                    lambda x: len(vars(x)['src'])
                                    <= max_sent_length
                                    and len(vars(x)['trg'])
                                    <= max_sent_length)

    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    trg_vocab_file = data_cfg.get("trg_vocab", None)

    trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq,
                            max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)

    random_train_subset = data_cfg.get("random_train_subset", -1)
    if random_train_subset > -1:
        # select this many training examples randomly and discard the rest
        keep_ratio = random_train_subset / len(train_data)
        keep, _ = train_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio],
            random_state=random.getstate())
        train_data = keep

    dev_data = TranslationDataset(path=dev_path,
                                  exts=("." + src_lang, "." + trg_lang),
                                  fields=(src_field, trg_field))
    test_data = None
    if test_path is not None:
        # check if target exists
        if os.path.isfile(test_path + "." + trg_lang):
            test_data = TranslationDataset(
                path=test_path, exts=("." + src_lang, "." + trg_lang),
                fields=(src_field, trg_field))
        else:
            # no target is given -> create dataset from src only
            test_data = MonoDataset(path=test_path, ext="." + src_lang,
                                    field=src_field)
    trg_field.vocab = trg_vocab
    return train_data, dev_data, test_data, trg_vocab


# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    # batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    # if train:
    #     # optionally shuffle and sort during training
    #     data_iter = data.BucketIterator(
    #         repeat=False, sort=False, dataset=dataset,
    #         batch_size=batch_size, batch_size_fn=batch_size_fn,
    #         train=True, sort_within_batch=True,
    #         sort_key=lambda x: len(x.src), shuffle=shuffle)
    # else:
    #     # don't sort/shuffle for validation/inference
    #     data_iter = data.BucketIterator(
    #         repeat=False, dataset=dataset,
    #         batch_size=batch_size, batch_size_fn=batch_size_fn,
    #         train=False, sort=False)

    data_iter = []
    data_labels = []

    for initIndex in range(0, len(dataset[0]), batch_size):
        curr_batch = [dataset[0][i] for i in range(initIndex, min(len(dataset[0]), initIndex + batch_size))]
        curr_labels = [dataset[1][i] for i in range(initIndex, min(len(dataset[0]), initIndex + batch_size))]
        data_iter.append(torch.tensor(curr_batch))
        data_labels.append(torch.tensor(curr_labels))

    return data_iter, data_labels


class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, ext: str, field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', field)]

        if hasattr(path, "readline"):  # special usage: stdin
            src_file = path
        else:
            src_path = os.path.expanduser(path + ext)
            src_file = open(src_path)

        examples = []
        for src_line in src_file:
            src_line = src_line.strip()
            if src_line != '':
                examples.append(data.Example.fromlist(
                    [src_line], fields))

        src_file.close()

        super(MonoDataset, self).__init__(examples, fields, **kwargs)
