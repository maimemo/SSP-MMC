"""
Copyright (c) 2016 Duolingo Inc. MIT Licence.

Python script that implements spaced repetition models from Settles & Meeder (2016).
Recommended to run with pypy for efficiency. See README.
"""

import argparse
import csv
import math
import os
import random
import sys

from collections import defaultdict, namedtuple

# various constraints on parameters and outputs
MIN_HALF_LIFE = 15.0 / (24 * 60)  # 15 minutes
MAX_HALF_LIFE = 274. * 100  # 9 months
LN2 = math.log(2.)

# data instance object
Instance = namedtuple('Instance', 'p t fv h a right wrong spelling'.split())


class SpacedRepetitionModel(object):
    """
    Spaced repetition model. Implements the following approaches:
      - 'hlr' (half-life regression; trainable)
      - 'lr' (logistic regression; trainable)
      - 'leitner' (fixed)
      - 'pimsleur' (fixed)
    """

    def __init__(self, trainset, testset, method='hlr', omit_h_term=False, initial_weights=None, lrate=.002, hlwt=.002,
                 l2wt=.001, sigma=1.):
        self.method = method
        self.omit_h_term = omit_h_term
        self.weights = defaultdict(float)
        if initial_weights is not None:
            self.weights.update(initial_weights)
        self.fcounts = defaultdict(int)
        self.lrate = lrate
        self.hlwt = hlwt
        self.l2wt = l2wt
        self.sigma = sigma
        self.eval_every = 5
        self.trainset = trainset
        self.testset = testset
        self.trainset_cnt = len(trainset)
        self.n_iter = 7500000
        self.n_epoch = int(self.n_iter / self.trainset_cnt + 1)
        random.seed(2021)

    def halflife(self, inst, base):
        try:
            dp = sum([self.weights[k] * x_k for (k, x_k) in inst.fv])
            """
            fv=[
                ('right', 2.6457513110645907), 
                ('wrong', 2.0), 
                ('bias', 1.0), 
                ('es:hago/hacer<vblex><pri><p1><sg>', 1.0)
            ]
            """
            return hclip(base ** dp)
        except:
            return MAX_HALF_LIFE

    def predict(self, inst, base=2.):
        if self.method == 'hlr':
            h = self.halflife(inst, base)
            p = 2. ** (-inst.t / h)
            return pclip(p), h
        elif self.method == 'leitner':
            try:
                h = hclip(2. ** inst.fv[0][1])
            except OverflowError:
                h = MAX_HALF_LIFE
            p = 2. ** (-inst.t / h)
            return pclip(p), h
        elif self.method == 'pimsleur':
            try:
                h = hclip(2. ** (2.35 * inst.fv[0][1] - 16.46))
            except OverflowError:
                h = MAX_HALF_LIFE
            p = 2. ** (-inst.t / h)
            return pclip(p), h
        elif self.method == 'lr':
            dp = sum([self.weights[k] * x_k for (k, x_k) in inst.fv])
            p = pclip(1. / (1 + math.exp(-dp)))
            h = -inst.t / math.log(p, 2)
            return p, hclip(h)
        elif self.method == 'anki':
            try:
                s = max(1.3, (2.5 - 0.15 * inst.fv[1][1])) ** inst.fv[0][1]
            except OverflowError:
                s = - MAX_HALF_LIFE * math.log(0.9) / LN2
            p = pclip(math.e ** (inst.t / s * math.log(0.9)))
            h = -inst.t / math.log(p, 2)
            return p, hclip(h)
        else:
            raise Exception

    def train_update(self, inst):
        if self.method == 'hlr':
            base = 2.
            p, h = self.predict(inst, base)
            dlp_dw = 2. * (p - inst.p) * (LN2 ** 2) * p * (inst.t / h)
            dlh_dw = 2. * (h - inst.h) * LN2 * h
            for (k, x_k) in inst.fv:
                rate = (1. / (1 + inst.p)) * self.lrate / math.sqrt(1 + self.fcounts[k])
                # rate = self.lrate / math.sqrt(1 + self.fcounts[k])
                # sl(p) update
                self.weights[k] -= rate * dlp_dw * x_k
                # sl(h) update
                if not self.omit_h_term:
                    self.weights[k] -= rate * self.hlwt * dlh_dw * x_k
                # L2 regularization update
                self.weights[k] -= rate * self.l2wt * self.weights[k] / self.sigma ** 2
                # increment feature count for learning rate
                self.fcounts[k] += 1
        elif self.method == 'leitner' or self.method == 'pimsleur' or self.method == 'anki':
            pass
        elif self.method == 'lr':
            p, _ = self.predict(inst)
            err = p - inst.p
            for (k, x_k) in inst.fv:
                # rate = (1./(1+inst.p)) * self.lrate   / math.sqrt(1 + self.fcounts[k])
                rate = self.lrate / math.sqrt(1 + self.fcounts[k])
                # error update
                self.weights[k] -= rate * err * x_k
                # L2 regularization update
                self.weights[k] -= rate * self.l2wt * self.weights[k] / self.sigma ** 2
                # increment feature count for learning rate
                self.fcounts[k] += 1

    def train(self):

        for i in range(1, self.n_epoch + 1):
            if self.method == 'leitner' or self.method == 'pimsleur' or self.method == 'anki':
                return
            random.shuffle(self.trainset)
            for inst in self.trainset:
                self.train_update(inst)
            if i % self.eval_every == 0:
                self.eval('test')

    def losses(self, inst):
        p, h = self.predict(inst)
        slp = (inst.p - p) ** 2
        slh = (inst.h - h) ** 2
        return slp, slh, p, h

    def eval(self, prefix=''):
        results = {'p': [], 'h': [], 'pp': [], 'hh': [], 'slp': [], 'slh': []}
        for inst in self.testset:
            slp, slh, p, h = self.losses(inst)
            results['p'].append(inst.p)  # ground truth
            results['h'].append(inst.h)
            results['pp'].append(p)  # predictions
            results['hh'].append(h)
            results['slp'].append(slp)  # loss function values
            results['slh'].append(slh)
        mae_p = mae(results['p'], results['pp'])
        mae_h = mae(results['h'], results['hh'])
        cor_p = spearmanr(results['p'], results['pp'])
        cor_h = spearmanr(results['h'], results['hh'])
        total_slp = sum(results['slp'])
        total_slh = sum(results['slh'])
        total_l2 = sum([x ** 2 for x in self.weights.values()])
        total_loss = total_slp + self.hlwt * total_slh + self.l2wt * total_l2
        if prefix:
            sys.stderr.write('%s\t' % prefix)
        sys.stderr.write(
            '%.1f (p=%.1f, h=%.1f, l2=%.1f)\tmae(p)=%.3f\tcor(p)=%.3f\tmae(h)=%.3f\tcor('
            'h)=%.3f\n' % \
            (total_loss, total_slp, self.hlwt * total_slh, self.l2wt * total_l2,
             mae_p, cor_p, mae_h, cor_h))

    def dump_weights(self, fname):
        with open(fname, 'w') as f:
            for (k, v) in self.weights.items():
                f.write('%s\t%.4f\n' % (k, v))

    def dump_predictions(self, fname, testset):
        with open(fname, 'w') as f:
            f.write('p\tpp\th\thh\n')
            for inst in testset:
                pp, hh = self.predict(inst)
                f.write('%.4f\t%.4f\t%.4f\t%.4f\n' % (inst.p, pp, inst.h, hh))

    def dump_detailed_predictions(self, fname, testset):
        with open(fname, 'w') as f:
            f.write('right,wrong,h,hh,p,pp\n')
            for inst in testset:
                pp, hh = self.predict(inst)
                f.write('%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (
                    inst.fv[0][1], inst.fv[1][1], inst.h, hh, inst.p, pp))


def pclip(p):
    # bound min/max model predictions (helps with loss optimization)
    return min(max(p, 0.0001), .9999)


def hclip(h):
    # bound min/max half-life
    return min(max(h, MIN_HALF_LIFE), MAX_HALF_LIFE)


def mae(l1, l2):
    # mean average error
    return mean([abs(l1[i] - l2[i]) for i in range(len(l1))])


def mean(lst):
    # the average of a list
    return float(sum(lst)) / len(lst)


def spearmanr(l1, l2):
    # spearman rank correlation
    m1 = mean(l1)
    m2 = mean(l2)
    num = 0.
    d1 = 0.
    d2 = 0.
    for i in range(len(l1)):
        num += (l1[i] - m1) * (l2[i] - m2)
        d1 += (l1[i] - m1) ** 2
        d2 += (l2[i] - m2) ** 2
    return num / math.sqrt(d1 * d2)


def read_data(input_file, method, omit_bias=False, omit_lexemes=False, max_lines=None):
    # read learning trace data in specified format, see README for details
    sys.stderr.write('reading data...')
    instances = list()
    f = open(input_file, 'r')
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if max_lines is not None and i >= max_lines:
            break
        p = pclip(float(row['p']))
        t = max(1, int(row['t']))
        h = hclip(-t / (math.log(p, 2)))
        right = row['right']
        wrong = row['wrong']
        total = right + wrong
        # feature vector is a list of (feature, value) tuples
        fv = []
        # core features based on method
        if method == 'leitner':
            fv.append((sys.intern('diff'), right - wrong))
        elif method == 'pimsleur':
            fv.append((sys.intern('total'), total))
        else:
            fv.append((sys.intern('right'), math.sqrt(1 + right)))
            fv.append((sys.intern('wrong'), math.sqrt(1 + wrong)))
        # optional flag features
        if method == 'lr':
            fv.append((sys.intern('time'), t))
        if not omit_bias:
            fv.append((sys.intern('bias'), 1.))
        if not omit_lexemes:
            fv.append((sys.intern('%s' % (row['spelling'])), 1.))
        instances.append(
            Instance(p, t, fv, h, (right + 2.) / (total + 4.)))
        if i % 1000000 == 0:
            sys.stderr.write('%d...' % i)
        if i >= 12000000:
            break
    sys.stderr.write('done!\n')
    random.shuffle(instances)
    splitpoint = int(0.8 * len(instances))
    return instances[:splitpoint], instances[splitpoint:]


argparser = argparse.ArgumentParser(description='Fit a SpacedRepetitionModel to data.')
argparser.add_argument('-b', action="store_true", default=False, help='omit bias feature')
argparser.add_argument('-l', action="store_true", default=False, help='omit lexeme features')
argparser.add_argument('-t', action="store_true", default=False, help='omit half-life term')
argparser.add_argument('-m', action="store", dest="method", default='hlr', help="hlr, lr, leitner, pimsleur, anki")
argparser.add_argument('-x', action="store", dest="max_lines", type=int, default=None,
                       help="maximum number of lines to read (for dev)")
argparser.add_argument('input_file', action="store", help='log file for training')

if __name__ == "__main__":

    args = argparser.parse_args()

    # model diagnostics
    sys.stderr.write('method = "%s"\n' % args.method)
    if args.b:
        sys.stderr.write('--> omit_bias\n')
    if args.l:
        sys.stderr.write('--> omit_lexemes\n')
    if args.t:
        sys.stderr.write('--> omit_h_term\n')
    # read data set
    trainset, testset = read_data(args.input_file, args.method, args.b, args.l, args.max_lines)
    sys.stderr.write('|train| = %d\n' % len(trainset))
    sys.stderr.write('|test|  = %d\n' % len(testset))

    # train model & print preliminary evaluation info
    model = SpacedRepetitionModel(trainset, testset, method=args.method, omit_h_term=args.t)
    model.train()

    # write out model weights and predictions
    filebits = [args.method] + \
               [k for k, v in sorted(vars(args).items()) if v is True] + \
               [os.path.splitext(os.path.basename(args.input_file).replace('.gz', ''))[0]]
    if args.max_lines is not None:
        filebits.append(str(args.max_lines))
    filebase = '.'.join(filebits)
    if not os.path.exists('results/'):
        os.makedirs('results/')
    model.dump_weights('results/' + filebase + '.weights.tsv')
    model.dump_predictions('results/' + filebase + '.preds.tsv', testset)
    # model.dump_detailed_predictions('results/'+filebase+'.detailed', testset)
