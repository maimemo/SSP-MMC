import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from model.utils import *

plt.style.use('seaborn-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.figsize'] = (6.0, 4.0)
plt.rcParams['figure.dpi'] = 300

policy = []
base = 1.05
min_index = - 30
difficulty_offset = 2
difficulty_limit = 20
target_halflife = 300

period_len = 7  # 滚动平均区间
learn_days = 1000  # 模拟时长
deck_size = 100000  # 新卡片总量
cost_per_day_limit = 200
new_limit = 200
review_limit = 200


def difficulty_distribution(x):
    if 0 <= x < 0.05:
        return 10
    if 0.05 <= x < 0.12:
        return 9
    if 0.12 <= x < 0.22:
        return 8
    if 0.22 <= x < 0.33:
        return 7
    if 0.33 <= x < 0.46:
        return 6
    if 0.46 <= x < 0.60:
        return 5
    if 0.60 <= x < 0.74:
        return 4
    if 0.74 <= x < 0.85:
        return 3
    if 0.85 <= x < 0.94:
        return 2
    if 0.94 <= x <= 1:
        return 1


def scheduler_init():
    for d in range(1, 21):
        dataset = pd.read_csv(f"./algo/result/ivl-{d}.csv", header=None, index_col=None)
        global policy
        policy.append(dataset.values)


def scheduler(difficulty, halflife, reps, lapses, method):
    interval = 1
    if method == 'memorize':
        interval = sampler(np.log(2) / halflife, 1, learn_days)
    elif method == 'halflife':
        interval = halflife
    elif method == 'ssp-mmc':
        index = int(round(np.log(halflife) / np.log(base) - min_index))
        interval = policy[difficulty - 1][index - 1][1]
    elif method == 'anki':
        interval = max(2.5 - 0.15 * lapses, 1.2) ** reps
    elif method == 'threshold':
        interval = - halflife * np.log2(0.9)
    return max(1, round(interval))


if __name__ == "__main__":
    scheduler_init()

    for method in ['memorize', 'ssp-mmc', 'halflife', 'anki', 'threshold']:

        random.seed(114)

        new_card_per_day = np.array([0.0] * learn_days)
        new_card_per_day_average_per_period = np.array([0.0] * learn_days)
        workload_per_day = np.array([0.0] * learn_days)
        workload_per_day_average_per_period = np.array([0.0] * learn_days)

        learned_per_day = np.array([0.0] * learn_days)
        record_per_day = np.array([0.0] * learn_days)
        meet_max_per_day = np.array([0.0] * learn_days)

        feature_list = ['difficulty', 'halflife', 'p_recall', 'delta_t', 'reps', 'lapses', 'last_date', 'due_date']

        dtypes = np.dtype([
            ('difficulty', int),
            ('halflife', float),
            ('p_recall', float),
            ('delta_t', int),
            ('reps', int),
            ('lapses', int),
            ('last_date', int),
            ('due_date', int),
        ])

        field_map = {
            'difficulty': 0, 'halflife': 1, 'p_recall': 2, 'delta_t': 3, 'reps': 4, 'lapses': 5, 'last_date': 6,
            'due_date': 7}

        df_memory = pd.DataFrame(np.full(deck_size, np.nan, dtype=dtypes), index=range(deck_size), columns=feature_list)
        df_memory['difficulty'] = df_memory['difficulty'].map(lambda x: difficulty_distribution(random.random()))
        df_memory['due_date'] = learn_days

        meet_target = 0
        flag = 0

        for day in tqdm(range(learn_days)):
            reviewed = 0
            learned = 0
            df_memory["delta_t"] = day - df_memory["last_date"]
            df_memory["p_recall"] = np.exp2(- df_memory["delta_t"] / df_memory["halflife"])
            need_review = df_memory[df_memory['due_date'] <= day]
            true_review = need_review.index[:review_limit]
            for idx in true_review:
                reviewed += 1
                df_memory.iat[idx, field_map['last_date']] = day

                halflife = df_memory.iat[idx, field_map['halflife']]
                difficulty = df_memory.iat[idx, field_map['difficulty']]
                p_recall = df_memory.iat[idx, field_map['p_recall']]
                reps = df_memory.iat[idx, field_map['reps']]
                lapses = df_memory.iat[idx, field_map['lapses']]

                if random.random() < p_recall:
                    next_halflife = cal_recall_halflife(difficulty, halflife, p_recall)
                    df_memory.iat[idx, field_map['halflife']] = next_halflife
                    df_memory.iat[idx, field_map['reps']] = reps + 1

                    if next_halflife >= target_halflife:
                        meet_target += 1
                        df_memory.iat[idx, field_map['due_date']] = learn_days

                    delta_t = scheduler(difficulty, next_halflife, reps, lapses, method)
                    df_memory.iat[idx, field_map['due_date']] = day + delta_t

                else:
                    next_halflife = cal_forget_halflife(halflife, p_recall)  # halflife
                    df_memory.iat[idx, field_map['halflife']] = next_halflife

                    reps = 0
                    lapses = lapses + 1

                    df_memory.iat[idx, field_map['reps']] = reps
                    df_memory.iat[idx, field_map['lapses']] = lapses

                    delta_t = scheduler(difficulty, next_halflife, reps, lapses, method)
                    df_memory.iat[idx, field_map['due_date']] = day + delta_t

                    difficulty = min(difficulty + difficulty_offset, difficulty_limit)
                    df_memory.iat[idx, field_map['difficulty']] = difficulty

            need_learn = df_memory[df_memory['halflife'].isna()]
            true_learn = need_learn.index[:min(new_limit, cost_per_day_limit - len(true_review))]

            for idx in true_learn:
                learned += 1

                difficulty = df_memory.iat[idx, field_map['difficulty']]
                p_recall = df_memory.iat[idx, field_map['p_recall']]
                reps = df_memory.iat[idx, field_map['reps']]
                lapses = df_memory.iat[idx, field_map['lapses']]

                halflife = cal_start_halflife(df_memory.iat[idx, field_map['difficulty']])  # halflife
                df_memory.iat[idx, field_map['halflife']] = halflife
                df_memory.iat[idx, field_map['last_date']] = day
                delta_t = scheduler(difficulty, halflife, reps, lapses, method)
                df_memory.iat[idx, field_map['due_date']] = day + delta_t

            new_card_per_day[day] = learned

            learned_per_day[day] = learned_per_day[day - 1] + learned
            workload_per_day[day] = learned + reviewed

            if day >= period_len:
                new_card_per_day_average_per_period[day] = np.true_divide(new_card_per_day[day - period_len:day].sum(),
                                                                          period_len)
                workload_per_day_average_per_period[day] = np.true_divide(workload_per_day[day - period_len:day].sum(),
                                                                          period_len)
            else:
                new_card_per_day_average_per_period[day] = np.true_divide(new_card_per_day[:day + 1].sum(), day + 1)
                workload_per_day_average_per_period[day] = np.true_divide(workload_per_day[:day + 1].sum(), day + 1)

            record_per_day[day] = df_memory['p_recall'].sum()
            meet_max_per_day[day] = meet_target
            if flag:
                break

        recall = df_memory['p_recall'].sum()
        total_learned = int(sum(new_card_per_day))
        total_reviewed = int(sum(workload_per_day)) - total_learned

        plt.figure(1)
        plt.plot(record_per_day, label=f'method={method}')

        plt.figure(2)
        plt.plot(meet_max_per_day, label=f'method={method}')

        plt.figure(3)
        plt.plot(new_card_per_day_average_per_period, label=f'learned={total_learned}, method={method}')

        plt.figure(4)
        plt.plot(workload_per_day_average_per_period, label=f'reviewed={total_reviewed}, method={method}')

        plt.figure(5)
        plt.plot(learned_per_day, label=f'method={method}')

        print('acc review', total_reviewed)
        print('acc learn', total_learned)
        print('meet target', meet_target)

        save = df_memory[df_memory['p_recall'] > 0]
        save.to_csv(f'./simulation_result/{method}.tsv', index=False, sep='\t')

    plt.figure(1)
    plt.title(f"每日学习上限:{cost_per_day_limit}-学习天数{learn_days}")
    plt.xlabel("days")
    plt.ylabel("expectation of memorization")
    plt.legend()
    plt.grid(True)
    plt.figure(2)
    plt.title(f"每日学习上限:{cost_per_day_limit}-学习天数{learn_days}")
    plt.xlabel("days")
    plt.ylabel("num of meet target halflife")
    plt.legend()
    plt.grid(True)
    plt.figure(3)
    plt.title(f"每日学习上限:{cost_per_day_limit}-学习天数{learn_days}")
    plt.xlabel("时间/天")
    plt.ylabel(f"每日新学数量({period_len}天平均)")
    plt.legend()
    plt.grid(True)
    plt.figure(4)
    plt.title(f"每日学习上限:{cost_per_day_limit}-学习天数{learn_days}")
    plt.xlabel("时间/天")
    plt.ylabel(f"每日学习数量({period_len}天平均)")
    plt.legend()
    plt.grid(True)
    plt.figure(5)
    plt.title(f"每日学习上限:{cost_per_day_limit}-学习天数{learn_days}")
    plt.xlabel("时间/天")
    plt.ylabel(f"每日累积数量({period_len}天平均)")
    plt.legend()
    plt.grid(True)
    plt.show()
