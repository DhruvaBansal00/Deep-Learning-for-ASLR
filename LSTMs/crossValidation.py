import random
import torch
from sklearn import model_selection

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def split(pairs, lang, device):
    train = []
    test = []
    for label in pairs:
        label_tensor = tensorFromSentence(lang, label, device)
        iters = pairs[label]
        test_index = random.randint(0, len(iters) - 1)
        accept_prob = random.random()
        for i in range(len(iters)):
            if i == test_index and len(iters) != 1 and accept_prob > 0.5:
                test.append([iters[i], label_tensor])
            else:
                train.append([iters[i], label_tensor])
    return train, test

def kfoldSplit(pairs, lang, device, split=10):
    folds = []
    inputs = []
    outputs = []
    for label in pairs:
        for iter in pairs[label]:
            inputs.append(iter)
            outputs.append(label)
    
    skf = model_selection.StratifiedKFold(n_splits=split, shuffle=True)
    indices = skf.split(inputs, outputs)

    for train_indices, test_indices in indices:
        curr_train = []
        curr_test = []
        for indices in train_indices:
            curr_train.append([inputs[indices], tensorFromSentence(lang,  outputs[indices], device)])
        for indices in test_indices:
            curr_test.append([inputs[indices], tensorFromSentence(lang,  outputs[indices], device)])
        folds.append([curr_train, curr_test])
    
    return folds
        
