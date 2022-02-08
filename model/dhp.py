import time
import pandas as pd
from model.utils import *


def dhp(line, h, d):
    recall = int(line[0])
    interval = int(line[1])
    if recall == 1:
        if interval == 0:
            h = cal_start_halflife(d)
        else:
            p_recall = np.exp2(- interval / h)
            h = cal_recall_halflife(d, h, p_recall)
    else:
        if interval == 0:
            h = cal_start_halflife(d)
        else:
            p_recall = np.exp2(- interval / h)
            h = cal_forget_halflife(d, h, p_recall)
            d = min(d + 2, 18)
    return h, d


def eval(testset, repeat, fold):
    record = pd.DataFrame(
        columns=['difficulty', 'r_history', 't_history',
                 't',
                 'h', 'hh', 'h_loss', 'p', 'pp', 'p_loss', 'total_cnt'])
    p_loss = 0
    h_loss = 0
    count = 0
    for idx, line in testset.iterrows():
        line_tensor = lineToTensor(list(
            zip([line['r_history']], [line['t_history']]))[0])
        ph = 0
        d = line['difficulty']
        for j in range(line_tensor.size()[0]):
            ph, d = dhp(line_tensor[j][0], ph, d)

        # print(f'model: {m}\tsample: {line}\tcorrect: {interval}\tpredict: {float(output)}')

        pp = np.power(2, -line['delta_t'] / ph)
        p = line['p_recall']
        p_loss += abs(p - pp) * line['total_cnt']

        h = line['halflife']
        h_loss += abs((ph - h) / h) * line['total_cnt']
        count += line['total_cnt']
        record = pd.concat([record, pd.DataFrame(
            {'difficulty': [line['difficulty']],
             'r_history': [line['r_history']],
             't_history': [line['t_history']],
             't': [line['delta_t']], 'h': [h],
             'hh': [round(ph, 2)], 'p': [p],
             'pp': [round(pp, 3)], 'p_loss': [round(abs(p - pp), 3)], 'h_loss': [round(abs((ph - h) / h), 3)],
             'total_cnt': [line['total_cnt']]})],
                           ignore_index=True)
    print(f"model: dhp")
    print(f'sample num: {count}')
    print(f"avg p loss: {p_loss / count}")
    print(f"avg h loss: {h_loss / count}")
    record.to_csv(f'./fit_result/dhp/repeat{repeat}_fold{fold}_{int(time.time())}.csv', index=False)
