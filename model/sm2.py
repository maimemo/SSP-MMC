import numpy as np
import time
import pandas as pd
from utils import *


def sm2(line, n, ef, i):
    g = int(line[0])
    if g == 1:
        if n == 0:
            n = 1
            next_i = 1
        elif n == 1:
            n = 2
            next_i = 6
        else:
            n += 1
            next_i = i * ef
        ef += 0.1
    else:
        n = 0
        next_i = 1
        ef += -0.5
    if ef < 1.3:
        ef = 1.3
    return n, ef, next_i


def eval(testset, repeat, fold):
    record = pd.DataFrame(
        columns=['r_history', 't_history', 'p_history',
                 't',
                 'h', 'hh', 'p', 'pp', 'loss'])
    loss = 0
    count = 0
    for idx, line in testset.iterrows():
        line_tensor = lineToTensor(list(
            zip([line['r_history']], [line['t_history']],
                [line['p_history']]))[0])
        n = 0
        ef = 2.5
        i = 0
        for j in range(line_tensor.size()[0]):
            n, ef, i = sm2(line_tensor[j][0], n, ef, i)

        # print(f'model: {m}\tsample: {line}\tcorrect: {interval}\tpredict: {float(output)}')
        pp = np.exp(np.log(0.9) * line['t'] / i)
        p = line['p']
        loss += abs(p - pp)
        count += 1
        h = i
        record = pd.concat([record, pd.DataFrame(
            {'r_history': [line['r_history']],
             't_history': [line['t_history']],
             'p_history': [line['p_history']],
             't': [line['t']], 'h': [line['h']],
             'hh': [round(h, 2)], 'p': [line['p']],
             'pp': [round(pp, 3)], 'loss': [round(abs(p - pp), 3)]})],
                           ignore_index=True)
    print(f"model: sm2")
    print(f'sample num: {count}')
    print(f"avg loss: {loss / count}")
    record.to_csv(f'./result/sm2/repeat{repeat}_fold{fold}_{int(time.time())}.csv', index=False)
