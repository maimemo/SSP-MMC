# SSP-MMC

Copyright (c) 2022 [MaiMemo](https://www.maimemo.com/), Inc. MIT License.

Stochastic-Shortest-Path-Minimize-Memorization-Cost (SSP-MMC) is a spaced repetition scheduling algorithm used to help learners remember more words in MaiMemo, a language learning application in China.

This repository contains a public release of the data and code used for several experiments in the following paper (which introduces SSP-MMC):

> Junyao Ye, Jingyong Su, and Yilong Cao. 2022. A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. ACM, 4381–4390.

You can access this paper at: https://dl.acm.org/doi/10.1145/3534678.3539081?cid=99660547150

When using this data set and/or software, please cite this publication. A BibTeX record is:

```
@inproceedings{10.1145/3534678.3539081,
author = {Ye, Junyao and Su, Jingyong and Cao, Yilong},
title = {A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling},
year = {2022},
publisher = {ACM},
doi = {10.1145/3534678.3539081},
pages = {4381–4390},
numpages = {10}
}
```

# Software

The file `data_preprocessing.py` is used to preprocess data for the DHP model.

The file `cal_model_param.py` contains the DHP model and HLR model.

The file `model/utils.py` saves the parameters of the DHP model for training and simulation.

The file `algo/main.cpp` contains a Cpp implementation of SSP-MMC, which aims at finding the optimal policy.

The file `simulator.py` provides an environment for comparing different scheduling algorithms.

## Workflow

1. Run `data_preprocessing.py` -> `halflife_for_fit.tsv`
2. Run `cal_model_param.py` -> `intercept_` and `coef_` for the DHP model
3. Save the parameters to the function `cal_recall_halflife` and ` cal_forget_halflife` in  `model/utils.py` and the function `cal_next_recall_halflife` in `algo/main.cpp`
4. Run `algo/main.cpp` -> optimal policy in `algo/result/`
5. Run `simulator.py` to compare the SSP-MMC with several baselines.

## Data Set and Format

The dataset is available on [Dataverse](https://doi.org/10.7910/DVN/VAGUL0) (1.6 GB). This is a 7zipped TSV file containing our experiments' 220 million MaiMemo student memory behavior logs.

The columns are as follows:

- `u` - student user ID who reviewed the word (anonymized)
- `w` - spelling of the word 

- `i` - total times the user has reviewed the word
- `d` - difficulty of the word
- `t_history` - interval sequence of the historic reviews
- `r_history` - recall sequence of the historic reviews
- `delta_t` - time elapsed from the last review
- `r` - result of the review
- `p_recall` - probability of recall
- `total_cnt` - number of users who did the same memory behavior
