import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from readData import Lang
import glob
import random
from train import trainIters
from encoderLSTM import EncoderLSTM
from decoderLSTM import DecoderLSTM
from calculateAccuracy import calculateTrainingAccuracy
from crossValidation import split, kfoldSplit

#########    HYPERPARAMETERS   ############
random.seed(42)
users = ["Ravi"]
file_name = "Ravi"
num_features = 0
hidden_size = 1024
epochs = 60
limit_features = False
lr = 1e-4
lr_decay = 0.95
lr_drop = 15
dropout = 0.5
num_layers = 1
k_fold = False
folds = 5
bidirectional = False
expansion_factor = 2
l2_penalty = 0.001
###########################################

sil0 = 0
sil1 = 0

def expand(dataset_as_array, factor):
    expanded_array = []
    for pair in dataset_as_array:
        content = pair[0]
        label = pair[1]

        expanded_pair = [[[],label] for i in range(factor)]
        for frame in range(len(content)):
            expanded_pair[frame % factor][0].append(content[frame])
        expanded_array.extend(expanded_pair)
    return expanded_array


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eng = Lang("english")
pairs = {}
print("Reading data from files...")
for user in users:
    for file in glob.glob("data/"+user+"/*.ark"):
        label = "sil0_"+file.split(".")[1]+"_sil1"
        label = label.replace("_", " ")
        eng.addSentence(label)

        sil0 = eng.word2index["sil0"]
        sil1 = eng.word2index["sil1"]
        content = []
        f = open(file)
        for x in f:
            line = x
            if "[" in x:
                line = x.split("[ ")[1]
            elif "]" in x:
                line = x.split("]")[0]
            features = []
            line = line.strip("\n").split(" ")
            if limit_features:
                line = line[-num_features:]
            for f in line:
                try:
                    features.append(float(f)*1000)
                except:
                    pass
            if len(features) != 0:
                num_features = len(features)
                content.append(torch.tensor(features, dtype=torch.float, device=device).view(1, 1, -1))
        if label in pairs:
            temp = pairs[label]
            temp.append(content)
            pairs[label] = temp
        else:
            pairs[label] = [content]
if not k_fold:    
    print("Splitting data into train and test...")
    train_set, test_set = split(pairs, eng, device)
    train_set, test_set = expand(train_set, expansion_factor), expand(test_set, expansion_factor)
    encoder = EncoderLSTM(num_features, hidden_size, dropout, num_layers=num_layers, bidirectional=bidirectional).to(device)
    decoder = DecoderLSTM(hidden_size, eng.n_words, dropout, num_layers=num_layers, bidirectional=bidirectional).to(device)
    print("Split done. Elements in train: %d and elements in test: %d. Starting training..." % (len(train_set), len(test_set)))
    best_encoder, best_decoder = trainIters(encoder, decoder, epochs, train_set, test_set, sil0, sil1, eng, lr=lr, lr_decay=lr_decay, lr_drop_epoch=lr_drop, l2_penalty=l2_penalty)
    print("Training done. Printing stats to file....")
    calculateTrainingAccuracy(best_encoder, best_decoder, test_set, eng, sil0, sil1, 'results/'+file_name+'/results.txt')
    print("Saving Models")
    torch.save(best_encoder.state_dict(), "models/"+file_name+"/encoderLSTM.pt")
    torch.save(best_decoder.state_dict(), "models/"+file_name+"/decoderLSTM.pt")

else:
    print("Generating folds...")
    trainTestFolds = kfoldSplit(pairs, eng, device, split=folds)
    print("Fold generation done...")
    fold_num = 1
    for curr_fold in trainTestFolds:
        encoder = EncoderLSTM(num_features, hidden_size, dropout, num_layers=num_layers, bidirectional=bidirectional).to(device)
        decoder = DecoderLSTM(hidden_size, eng.n_words, dropout, num_layers=num_layers, bidirectional=bidirectional).to(device)
        print("Starting training on fold %d. %d elements in curr_fold[0] and %d in curr_fold[1]" % (fold_num, len(curr_fold[0]), len(curr_fold[1])))
        best_encoder, best_decoder = trainIters(encoder, decoder, epochs, curr_fold[0], curr_fold[1], sil0, sil1, eng, lr=lr, lr_decay=lr_decay, lr_drop_epoch=lr_drop, l2_penalty=l2_penalty)
        print("Training done. Saving predictions to file...")
        calculateTrainingAccuracy(best_encoder, best_decoder, curr_fold[1], eng, sil0, sil1, 'results/'+file_name+'/results_fold'+str(fold_num)+'.txt')
        print("Saving Models")
        torch.save(best_encoder.state_dict(), "models/"+file_name+"/encoderLSTM_fold"+str(fold_num)+".pt")
        torch.save(best_decoder.state_dict(), "models/"+file_name+"/decoderLSTM_fold"+str(fold_num)+".pt")
        fold_num += 1

