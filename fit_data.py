import argparse
import random
import sys
import os
import time
import pandas as pd
import math
from collections import namedtuple
from sklearn.model_selection import train_test_split, RepeatedKFold

Instance = namedtuple('Instance', 'p t fv h a right wrong spelling'.split())
duolingo_algo = ('hlr', 'lr', 'leitner', 'pimsleur')


def load_data(input_file):
    dataset = pd.read_csv(input_file, sep='\t', index_col=None)
    # dataset = dataset[dataset['group_cnt'] > 100000]
    dataset = dataset[dataset['i'] >= 3]
    dataset['r_history'] = dataset['r_history'].map(lambda x: str(int(x)))
    dataset['t_history'] = dataset['t_history'].map(lambda x: x[2:])
    dataset['right'] = dataset['r_history'].str.count('1')
    dataset['wrong'] = dataset['r_history'].str.count('0')
    return dataset


def feature_extract(trainset, testset, method):
    instances = {'train': [], 'test': []}
    for set_id, data in (('train', trainset), ('test', testset)):
        for i, row in data.iterrows():
            p = float(row['p_recall'])
            t = max(1, int(row['delta_t']))
            h = float(row['halflife'])
            right = row['right']
            wrong = row['wrong']
            total = right + wrong
            # feature vector is a list of (feature, value) tuples
            fv = []
            # core features based on method
            fv.append((sys.intern('right'), math.sqrt(1 + right)))
            fv.append((sys.intern('wrong'), math.sqrt(1 + wrong)))
            # optional flag features
            if method == 'lr':
                fv.append((sys.intern('time'), t))
            fv.append((sys.intern('%s' % (row['difficulty'])), 1.))
            fv.append((sys.intern('bias'), 1.))
            instances[set_id].append(
                Instance(p, t, fv, h, (right + 2.) / (total + 4.), row['right'], row['wrong'], row['difficulty']))
            if i % 1000000 == 0:
                sys.stderr.write('%d...' % i)
        sys.stderr.write('done!\n')
    return instances['train'], instances['test']


argparser = argparse.ArgumentParser(description='Fit a SpacedRepetitionModel to data.')
argparser.add_argument('-test', action="store_true", default=False, help='test model')
argparser.add_argument('-m', action="store", dest="method", default='dhp', help="dhp, hlr, lr, sm2")
argparser.add_argument('input_file', action="store", help='log file for training')
if __name__ == "__main__":

    random.seed(2021)
    args = argparser.parse_args()
    sys.stderr.write('method = "%s"\n' % args.method)

    dataset = load_data(args.input_file)
    test = dataset.sample(frac=0.8, random_state=2021)
    train = dataset.drop(index=test.index)

    if not args.test:
        train_train, train_test = train_test_split(train, test_size=0.2, random_state=2021)
        sys.stderr.write('|train| = %d\n' % len(train_train))
        sys.stderr.write('|test|  = %d\n' % len(train_test))
        from model.halflife_regression import SpacedRepetitionModel

        train_fold, test_fold = feature_extract(train_train, train_test, args.method)
        model = SpacedRepetitionModel(train_fold, test_fold, method=args.method)
        model.train()
    else:
        kf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=2021)
        for idx, (train_index, test_fold) in enumerate(kf.split(test)):
            train_fold = dataset.iloc[train_index]
            test_fold = dataset.iloc[test_fold]
            repeat = idx // 2 + 1
            fold = idx % 2 + 1
            sys.stderr.write('Repeat %d, Fold %d\n' % (repeat, fold))
            sys.stderr.write('|train| = %d\n' % len(train_index))
            sys.stderr.write('|test|  = %d\n' % len(test_fold))
            if args.method in duolingo_algo:
                from model.halflife_regression import SpacedRepetitionModel

                train_fold, test_fold = feature_extract(train_fold, test_fold, args.method)
                model = SpacedRepetitionModel(train_fold, test_fold, method=args.method)
                model.train()
                model.eval()
                filebits = [args.method] + \
                           [k for k, v in sorted(vars(args).items()) if v is True] + \
                           [os.path.splitext(os.path.basename(args.input_file).replace('.gz', ''))[0]]
                filebase = '.'.join(filebits)
                if args.method in ('hlr', 'lr'):
                    model.dump_detailed_predictions(
                        f'./fit_result/{args.method}/repeat{repeat}_fold{fold}_{int(time.time())}.csv',
                        test_fold)
                else:
                    model.dump_detailed_predictions(
                        f'./fit_result/{args.method}/repeat{repeat}_fold{fold}_{idx}_{int(time.time())}.csv', test_fold)
            elif args.method == 'sm2':
                from model.sm2 import eval

                eval(test_fold, repeat, fold)
            elif args.method == 'dhp':
                from model.dhp import eval

                eval(test_fold, repeat, fold)
            else:
                break