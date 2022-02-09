import torch
import numpy as np

responses_dict = {'1': 1, '2': 0, '3': 0}


def responseToIndex(response):
    return responses_dict[response]


def lineToTensor(line):
    tensor = torch.zeros(len(line[0]), 1, 2)
    ivl = line[1].split(',')
    for li, response in enumerate(line[0]):
        tensor[li][0][0] = responseToIndex(response)
        tensor[li][0][1] = int(ivl[li])
    return tensor


def cal_start_halflife(difficulty):
    return - 1 / np.log2(max(0.925 - 0.05 * difficulty, 0.025))


def cal_recall_halflife(difficulty, halflife, p_recall):
    return halflife * (
            1 + np.exp(3.81) * np.power(difficulty, -0.534) * np.power(halflife, -0.127) * np.power(
        1 - p_recall, 0.97))


def cal_forget_halflife(difficulty, halflife, p_recall):
    return np.exp(-0.041) * np.power(difficulty, -0.041) * np.power(halflife, 0.377) * np.power(
        1 - p_recall, -0.227)


# the following code is from https://github.com/Networks-Learning/memorize
def intensity(t, n_t, q):
    return 1.0 / np.sqrt(q) * (1 - np.exp(-n_t * t))


def sampler(n_t, q, T):
    t = 0
    while (True):
        max_int = 1.0 / np.sqrt(q)
        t_ = np.random.exponential(1 / max_int)
        if t_ + t > T:
            return None
        t = t + t_
        proposed_int = intensity(t, n_t, q)
        if np.random.uniform(0, 1, 1)[0] < proposed_int / max_int:
            return t
