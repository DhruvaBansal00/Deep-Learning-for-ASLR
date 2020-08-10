
#CV = python3 mainDriver.py --test_type cross_val --cross_val_method stratified --n_splits 10

import sys
import glob
import argparse
import os
import shutil
import numpy as np
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneGroupOut, train_test_split)
from statistics import mean
import joeynmt
from joeynmt.training import train

def writeFiles(trainPaths, trainLabels, testPaths, testLabels):
    trainFiles = "\n".join(trainPaths)
    testFiles = "\n".join(testPaths)

    trainPathFile = open('../data/lists/train.data', 'w')
    trainLabelFile = open('../data/lists/train.en', 'w')
    devPathFile = open('../data/lists/dev.data', 'w')
    devLabelFile = open('../data/lists/dev.en', 'w')
    testPathFile = open('../data/lists/test.data', 'w')
    testLabelFile = open('../data/lists/test.en', 'w')

    trainPathFile.write(trainFiles)
    devPathFile.write(testFiles)
    testPathFile.write(testFiles)

    trainLabelFile.writelines(trainLabels)
    devLabelFile.writelines(testLabels)
    testLabelFile.writelines(testLabels)


def getLabels(ark_paths: list):
    labels = [" ".join(i.split("/")[-1].split(".")[1].split("_"))+"\n" for i in ark_paths]
    return labels

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("SignJoey")
    ######################### ARGUMENTS #############################
    parser.add_argument('--test_type', type=str, default='test_on_train',
                        choices=['test_on_train', 'cross_val', 'standard'])
    parser.add_argument('--users', nargs='*', default=[])
    parser.add_argument('--phrase_len', type=int, default=0)
    parser.add_argument('--random_state', type=int, default=24)
    parser.add_argument('--cross_val_method', required='cross_val' in sys.argv,
                        default='kfold', choices=['kfold',
                                                  'leave_one_phrase_out',
                                                  'stratified'])
    parser.add_argument('--n_splits', required='cross_val' in sys.argv,
                        type=int, default=10)
    parser.add_argument('--cv_parallel', action='store_true')
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument("--config_path", type=str, help="path to YAML config file")
    args = parser.parse_args()
    ###################################################################

    print("Version 0.0.1")

    cross_val_methods = {'kfold': (KFold, False),
                         'leave_one_phrase_out': (LeaveOneGroupOut, True),
                         'stratified': (StratifiedKFold, True)}
    cross_val_method, use_groups = cross_val_methods[args.cross_val_method]

    if len(args.users) == 0:
        ark_filepaths = glob.glob('../data/ark/*ark')
    else:
        ark_filepaths = []
        for user in args.users:
            ark_filepaths.extend(glob.glob(os.path.join("../data/ark", '*{}*.ark'.format(user))))
    
    ark_labels = getLabels(ark_filepaths)

    if args.test_type == 'test_on_train':
        
        train_paths = ark_filepaths
        train_labels = ark_labels
        test_paths = ark_filepaths
        test_labels = ark_labels

        print(f'Nmber of elements in train_paths = {str(len(train_paths))}')
        print(f'Nmber of elements in test_paths = {str(len(test_paths))}')

        writeFiles(train_paths, train_labels, test_paths, test_labels)
        train(args.config_path)

    
    if args.test_type == 'standard':

        train_paths, test_paths, train_labels, test_labels = train_test_split(
            ark_filepaths, ark_labels, test_size=args.test_size,
            random_state=args.random_state)

        print(f'Nmber of elements in train_paths = {str(len(train_paths))}')
        print(f'Nmber of elements in test_paths = {str(len(test_paths))}')

        writeFiles(train_paths, train_labels, test_paths, test_labels)
        train(args.config_path)

    if args.test_type == 'cross_val':

        unique_phrases = set(ark_labels)
        group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
        groups = [group_map[label] for label in ark_labels]
        cross_val = cross_val_method(n_splits=args.n_splits)

        if use_groups:
            splits = list(cross_val.split(ark_filepaths, ark_labels, groups))
        else:
            splits = list(cross_val.split(ark_filepaths, ark_labels))
        
        all_results = []
        
        for i, (train_index, test_index) in enumerate(splits):

            print(f'Current split = {i}')
            
            train_paths = np.array(ark_filepaths)[train_index]
            train_labels = np.array(ark_labels)[train_index]
            test_paths = np.array(ark_filepaths)[test_index]
            test_labels = np.array(ark_labels)[test_index]

            print(f'Nmber of elements in train_paths = {str(train_paths.shape)}')
            print(f'Nmber of elements in test_paths = {str(test_paths.shape)}')

            writeFiles(train_paths, train_labels, test_paths, test_labels)
            all_results.append(train(args.config_path))
        
        all_results = np.array(all_results)
        average_results = np.mean(all_results, dim=1)
        print(f'Cross validation results = {str(average_results)}')





