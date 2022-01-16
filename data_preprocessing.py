import pandas as pd
import numpy as np
from tqdm import tqdm


def halflife_forgetting_curve(x, h):
    return np.power(2, - x / h)


def cal_halflife(group):
    if group['repeat_long'].values[0] > 1:
        r_ivl_cnt = sum(group['delta_t'] * group['p_recall'].map(np.log) * group['total_cnt'])
        ivl_ivl_cnt = sum(group['delta_t'].map(lambda x: x ** 2) * group['total_cnt'])
        group['halflife'] = round(np.log(0.5) / (r_ivl_cnt / ivl_ivl_cnt), 4)
    else:
        group['halflife'] = 0.0
    group['group_cnt'] = sum(group['total_cnt'])
    return group


if __name__ == '__main__':
    raw_data = pd.read_csv('./data/tmp_essay_dataset.tsv', sep='\t', index_col=None)
    raw_data = raw_data.rename(
        columns={'response_history_long': 'r_history', 'real_interval_history_byday': 't_history',
                 'last_real_interval_byday': 'delta_t',
                 'recall': 'p_recall'})
    raw_data['r_history'] = raw_data['r_history'].map(str)
    raw_data['r_history'] = raw_data['r_history'].map(lambda x: '0' + x if x != '0' else x)
    raw_data = raw_data.groupby(
        by=['difficulty', 'repeat_long', 'r_history', 't_history']).apply(
        cal_halflife)
    raw_data.to_csv('./data/halflife_for_fit.tsv', sep='\t', index=None)
    for idx in tqdm(raw_data.index):
        item = raw_data.loc[idx]
        delat_t = int(item['delta_t'])
        index = raw_data[(raw_data['repeat_long'] == item['repeat_long'] + 1) &
                         (raw_data['difficulty'] == item['difficulty']) &
                         (raw_data['r_history'].str.startswith(item['r_history'])) &
                         (raw_data['t_history'] == item['t_history'] + f',{delat_t}')].index
        raw_data.loc[index, 'last_halflife'] = item['halflife']
        raw_data.loc[index, 'last_p_recall'] = item['p_recall']
    raw_data['halflife_increase'] = round(raw_data['halflife'] / raw_data['last_halflife'], 4)
    raw_data = raw_data[raw_data['repeat_long'] > 2]
    raw_data['last_recall'] = raw_data['r_history'].map(lambda x: {'1': 1, '3': 0}[x[-1]])
    del raw_data['delta_t']
    del raw_data['p_recall']
    del raw_data['total_cnt']
    raw_data.drop_duplicates(inplace=True)
    raw_data.dropna(inplace=True)
    raw_data.to_csv('./data/halflife_for_visual.tsv', sep='\t', index=None)
