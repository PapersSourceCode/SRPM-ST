import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDAsk
from sklearn.svm import LinearSVC as LSVCsk
from sklearn.linear_model import LogisticRegression as LRRsk
from joblib import Parallel, delayed

import numpy as np

from datetime import datetime

from utils.dataset import get_dataset
from utils.plot import plot_results

import sys

print('hello')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='skin',
                    help='mnist | usps | skin | har | churn | texture | wine | kdd')
parser.add_argument('--model', type=str, default='LDA',
                    help='LDA | LRR | LSVM')
parser.add_argument('--training_size', type=float,
                    default=0.01, help='training set proportion')
parser.add_argument('--test_size', type=float,
                    default=0.1, help='test set proportion')
parser.add_argument(
    '--Is', type=str, default="0,1,2,3,4,5,6",
    help='set of Is. for each i in Is: mini-batch size = (total number of unlabeled samples) / 2**i')
parser.add_argument('--output', type=str,
                    default='output.png', help='output file name')

parser.add_argument('--njobs', type=int,
                    default=1, help='number of jobs to parallelize the ST-CV procedure')
parser.add_argument('--K', type=int,
                    default=5, help='number of folds in stratified K-fold CV')
parser.add_argument('--SS', type=int,
                    default=30, help='number of random shuffle and splits')


opt = parser.parse_args()


def batch(X_unl, mini_batch_size):
    l = len(X_unl)
    for ndx in range(0, l, mini_batch_size):
        x_mini_batch = X_unl[ndx:min(ndx + mini_batch_size, l)]
        if x_mini_batch.shape[0] == mini_batch_size:
            yield x_mini_batch


def skf5fold(model_name, X_train, y_train, X_unl, train2_index, crossval_index, i):

    X_train2, X_crossval = X_train[train2_index], X_train[crossval_index]
    y_train2, y_crossval = y_train[train2_index], y_train[crossval_index]

    mini_batch_size = int(np.floor(X_unl.shape[0] / 2**i))

    train_items = np.copy(X_train2)
    train_items_labels = np.copy(y_train2)

    crossval_items = np.copy(X_crossval)
    crossval_items_labels = np.copy(y_crossval)

    scaler = StandardScaler()
    scaler.fit(train_items)
    X_train_scaled = scaler.transform(train_items)

    if model_name == 'LDA':
        model = LDAsk()
    elif model_name == 'LRR':
        model = LRRsk(random_state=42)
    elif model_name == 'LSVM':
        model = LSVCsk(random_state=42)
    model.fit(X_train_scaled, train_items_labels)

    j = 1
    for x_mini_bath in batch(X_unl, mini_batch_size):

        X_unl_scaled = scaler.transform(x_mini_bath)
        preds = model.predict(X_unl_scaled)

        train_items_labels = np.concatenate((train_items_labels, preds))
        train_items = np.concatenate(
            (train_items, x_mini_bath))

        scaler = StandardScaler()
        scaler.fit(train_items)
        X_train_scaled = scaler.transform(train_items)
        X_val_scaled = scaler.transform(crossval_items)

        if model_name == 'LDA':
            model = LDAsk()
        elif model_name == 'LRR':
            model = LRRsk(random_state=42)
        elif model_name == 'LSVM':
            model = LSVCsk(random_state=42)
        model.fit(X_train_scaled, train_items_labels)

        if j == 2**i:
            crossval_preds = model.predict(X_val_scaled)
            crossval_acc = accuracy_score(y_pred=crossval_preds, y_true=crossval_items_labels)
            break

        j += 1

    return dict(crossval_acc=crossval_acc)


def main():
    ds = ['mnist', 'usps', 'skin', 'har', 'churn', 'texture', 'wine', 'kdd']
    mls = ['LDA', 'LRR', 'LSVM']

    dataset_name = opt.dataset
    model_name = opt.model

    if dataset_name not in ds:
        raise ValueError(
            f'Given dataset is not supported. Please indicate one of the following datasets. \n {ds}')
    if model_name not in mls:
        raise ValueError(
            f'Given model is not supported. Please indicate one of the following datasets. \n {mls}')
    data, labels = get_dataset(dataset_name)

    print('Data shape: ', data.shape, 'Labels shape: ', labels.shape)
    print('Target classes with counts: ', np.unique(labels, return_counts=True))

    start_time = datetime.now()

    test_size = opt.test_size

    train_size = opt.training_size

    Is = [int(i) for i in opt.Is.split(",")]

    initials_mean_accs_all_Is = []
    final_mean_accs_all_Is = []
    crossval_mean_accs_all_Is = []

    initials_std_accs_all_Is = []
    final_std_accs_all_Is = []
    crossval_std_accs_all_Is = []

    skf = StratifiedKFold(n_splits=opt.K)
    for i in Is:
        sss = StratifiedShuffleSplit(
            n_splits=opt.SS, test_size=test_size, random_state=42)
        initial_test_acc_all_splits = []
        final_test_acc_all_splits = []
        crossval_acc_all_splits = []

        for train_unl_index, test_index in sss.split(data, labels):

            X_unl_train, X_test = data[train_unl_index], data[test_index]
            y_unl_train, y_test = labels[train_unl_index], labels[test_index]

            X_train, X_unl, y_train, y_unl = train_test_split(
                X_unl_train, y_unl_train, train_size=train_size, random_state=42,
                stratify=y_unl_train)

            st_cv = Parallel(
                n_jobs=opt.njobs, verbose=1, pre_dispatch='1.5*n_jobs')(
                delayed(skf5fold)
                (model_name, X_train, y_train, X_unl, train2_index, crossval_index, i)
                for train2_index, crossval_index in skf.split(X_train, y_train))

            crossval_acc_all_folds = [st_cv_k['crossval_acc'] for st_cv_k in st_cv]
            avg_crossval_5fold = np.mean(crossval_acc_all_folds, axis=0)
            crossval_acc_all_splits.append(avg_crossval_5fold)

            mini_batch_size = int(np.floor(X_unl.shape[0] / 2**i))

            train_items = np.copy(X_train)
            train_items_labels = np.copy(y_train)

            test_items = np.copy(X_test)
            test_items_labels = np.copy(y_test)

            scaler = StandardScaler()
            scaler.fit(train_items)
            X_train_scaled = scaler.transform(train_items)
            X_test_scaled = scaler.transform(test_items)

            if model_name == 'LDA':
                model = LDAsk()
            elif model_name == 'LRR':
                model = LRRsk(random_state=42)
            elif model_name == 'LSVM':
                model = LSVCsk(random_state=42)

            model.fit(X_train_scaled, train_items_labels)

            test_preds = model.predict(X_test_scaled)
            test_acc = accuracy_score(y_pred=test_preds, y_true=test_items_labels)
            initial_test_acc_all_splits.append(test_acc)

            j = 1
            for x_mini_batch in batch(X_unl, mini_batch_size):

                X_mini_batch_scaled = scaler.transform(x_mini_batch)
                preds = model.predict(X_mini_batch_scaled)

                train_items_labels = np.concatenate(
                    (train_items_labels, preds))

                train_items = np.concatenate(
                    (train_items, x_mini_batch))

                scaler = StandardScaler()
                scaler.fit(train_items)
                X_train_scaled = scaler.transform(train_items)
                X_test_scaled = scaler.transform(test_items)

                if model_name == 'LDA':
                    model = LDAsk()
                elif model_name == 'LRR':
                    model = LRRsk(random_state=42)
                elif model_name == 'LSVM':
                    model = LSVCsk(random_state=42)

                model.fit(X_train_scaled, train_items_labels)

                if j == 2**i:
                    test_preds = model.predict(X_test_scaled)
                    test_acc = accuracy_score(y_pred=test_preds, y_true=test_items_labels)
                    final_test_acc_all_splits.append(test_acc)
                    break

                j += 1

        crossval_acc_mean = np.mean(crossval_acc_all_splits, axis=0)
        crossval_acc_std = np.std(crossval_acc_all_splits, axis=0)

        final_acc_mean = np.mean(final_test_acc_all_splits, axis=0)
        final_acc_std = np.std(final_test_acc_all_splits, axis=0)

        initial_acc_mean = np.mean(initial_test_acc_all_splits, axis=0)
        initial_acc_std = np.std(initial_test_acc_all_splits, axis=0)

        initials_mean_accs_all_Is.append(initial_acc_mean)
        final_mean_accs_all_Is.append(final_acc_mean)
        crossval_mean_accs_all_Is.append(crossval_acc_mean)

        initials_std_accs_all_Is.append(initial_acc_std)
        final_std_accs_all_Is.append(final_acc_std)
        crossval_std_accs_all_Is.append(crossval_acc_std)

    output_file_name = opt.output
    plot_results(dataset_name, model_name, Is, train_size, initials_mean_accs_all_Is,
                 final_mean_accs_all_Is, crossval_mean_accs_all_Is, output_file_name)

    end_time = datetime.now()
    print(f'Results are saved as {output_file_name}')
    print(f'script time: {end_time-start_time}')


if __name__ == '__main__':
    main()
