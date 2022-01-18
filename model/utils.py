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
    return 5.25 * np.power(difficulty, -0.866)


def cal_recall_halflife(difficulty, halflife, p_recall):
    return np.exp(1.83) * np.power(difficulty, -0.305) * np.power(halflife, 0.765) * np.exp(1.26 * (1 - p_recall))


def cal_forget_halflife(difficulty, halflife, p_recall):
    return np.exp(0.5) * np.power(difficulty, -0.068) * np.power(halflife, 0.4) * np.exp(-0.688 * (1 - p_recall))


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
