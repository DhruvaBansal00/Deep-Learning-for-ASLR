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
import generator
from joeynmt.training import train
from generator.generateNewFeatures import generateFeatures


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

def getUsers(ark_paths: list):
    users = [filepath.split('/')[-1].split('.')[0].split('_')[1] for filepath in ark_paths]
    return users

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
                                                  'stratified',
                                                  'leave_one_user_out'])
    parser.add_argument('--n_splits', required='cross_val' in sys.argv,
                        type=int, default=10)
    parser.add_argument('--transform_files', action='store_true')
    parser.add_argument('--create_transform_files', action='store_true')
    parser.add_argument('--cv_parallel', action='store_true')
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument("--config_path", type=str, help="path to YAML config file")
    parser.add_argument('--classifier', type=str, default='knn',
                        choices=['knn', 'adaboost'])
    parser.add_argument('--include_state', action='store_true')
    parser.add_argument('--include_index', action='store_true')
    parser.add_argument('--n_jobs', default=1, type=int)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--multiple_classifiers', action='store_true')
    parser.add_argument('--knn_neighbors', default=50)
    parser.add_argument('--pca_components', default=92)
    parser.add_argument('--no_pca', action='store_true')
    args = parser.parse_args()
    ###################################################################

    print("Version 0.0.2")

    cross_val_methods = {'kfold': (KFold, False),
                         'leave_one_phrase_out': (LeaveOneGroupOut(), True),
                         'stratified': (StratifiedKFold, True),
                         'leave_one_user_out': (LeaveOneGroupOut(), True)
    }
    cross_val_method, use_groups = cross_val_methods[args.cross_val_method]

    if len(args.users) == 0:
        ark_filepaths = glob.glob('../data/ark/*ark')
    else:
        ark_filepaths = []
        for user in args.users:
            ark_filepaths.extend(glob.glob(os.path.join("../data/ark", '*{}*.ark'.format(user))))
    
    ark_labels = getLabels(ark_filepaths)
    dataset_users = getUsers(ark_filepaths)

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
    
    if args.test_type == 'cross_val' and args.transform_files:

        unique_users = set(dataset_users)
        group_map = {user: i for i, user in enumerate(unique_users)}
        groups = [group_map[user] for user in dataset_users]            
        cross_val = cross_val_method
        splits = list(cross_val.split(ark_filepaths, ark_labels, groups))
        
        all_results = []
        user_order = []
        
        for i, (train_index, test_index) in enumerate(splits):

            print(f'Current split = {i}')
            
            train_paths = np.array(ark_filepaths)[train_index]
            train_labels = np.array(ark_labels)[train_index]
            test_paths = np.array(ark_filepaths)[test_index]
            test_labels = np.array(ark_labels)[test_index]
            curr_user = getUsers(test_paths)[0]
            user_order.append(curr_user)

            print(f'Current user = {curr_user}')

            curr_alignment_file = glob.glob(f'../data/alignment/{curr_user}/*.mlf')[-1]

            if args.create_transform_files:

                print(f'Starting feature generation')

                generator = generateFeatures(curr_alignment_file, "../data/ark/", classifier=args.classifier, include_state=args.include_state, 
                            include_index=args.include_index, n_jobs=args.n_jobs, parallel=args.parallel, trainMultipleClassifiers=args.multiple_classifiers,
                            knn_neighbors=int(args.knn_neighbors), generated_features_folder=f'../data/transformed/{curr_user}/', pca_components=args.pca_components,
                            no_pca=args.no_pca)
            
            transformedFiles = glob.glob(f'../data/transformed/{curr_user}/*.ark')
            train_paths = []
            train_labels = []
            test_paths = []
            test_labels = []
            for filePath in transformedFiles:
                if curr_user in filePath.split("/")[-1]:
                    test_paths.append(filePath)
                    test_labels.append(getLabels([filePath])[0])
                else:
                    train_paths.append(filePath)
                    train_labels.append(getLabels([filePath])[0])
            
            print(f'Number of elements in train_paths = {str(len(train_paths))}')
            print(f'Number of elements in test_paths = {str(len(test_paths))}')
            
            writeFiles(train_paths, train_labels, test_paths, test_labels)
            all_results.append(train(args.config_path))    
            

    elif args.test_type == 'cross_val':

        if args.cross_val_method == 'leave_one_user_out':
            unique_users = set(dataset_users)
            group_map = {user: i for i, user in enumerate(unique_users)}
            groups = [group_map[user] for user in dataset_users]            
            cross_val = cross_val_method
        else:
            unique_phrases = set(ark_labels)
            group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
            groups = [group_map[label] for label in ark_labels]
            cross_val = cross_val_method(n_splits=args.n_splits)
            

        if use_groups:
            splits = list(cross_val.split(ark_filepaths, ark_labels, groups))
        else:
            splits = list(cross_val.split(ark_filepaths, ark_labels))
        
        all_results = []
        user_order = []
        
        for i, (train_index, test_index) in enumerate(splits):

            print(f'Current split = {i}')
            
            train_paths = np.array(ark_filepaths)[train_index]
            train_labels = np.array(ark_labels)[train_index]
            test_paths = np.array(ark_filepaths)[test_index]
            test_labels = np.array(ark_labels)[test_index]
            curr_user = getUsers(test_paths)[0]
            user_order.append(curr_user)

            print(f'Nmber of elements in train_paths = {str(train_paths.shape)}')
            print(f'Nmber of elements in test_paths = {str(test_paths.shape)}')
            print(f'Current user = {curr_user}')

            writeFiles(train_paths, train_labels, test_paths, test_labels)
            all_results.append(train(args.config_path))
        
        all_results = np.array(all_results)
        print(f'All results = {str(all_results)}')
        average_results = np.mean(all_results, axis=0)
        print(f'Cross validation results = {str(average_results)}')
        print(f'User order = {str(user_order)}')





