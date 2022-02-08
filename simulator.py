import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random
from model.utils import *

plt.style.use('seaborn')
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.figsize'] = (6.0, 4.0)
plt.rcParams['figure.dpi'] = 300

policy = []
base = 1.05
min_index = - 30
difficulty_offset = 2
difficulty_limit = 18
target_halflife = 360

period_len = 30  # 滚动平均区间
learn_days = 1000  # 模拟时长
deck_size = 100000  # 新卡片总量

recall_cost = 3
forget_cost = 9
new_cost = 6
day_cost_limit = 600
compare_target = 6000 


def scheduler_init():
    for d in range(1, difficulty_limit + 1):
        dataset = pd.read_csv(f"./algo/result/ivl-{d}.csv", header=None, index_col=None)
        global policy
        policy.append(dataset.values)


def scheduler(difficulty, halflife, reps, lapses, method):
    interval = 1
    if method == 'MEMORIZE':
        interval = sampler(np.log(2) / halflife, 1, learn_days * 100)
    elif method == 'HALF-LIFE':
        interval = halflife
    elif method == 'SSP-MMC':
        halflife = min(halflife, target_halflife)
        index = int(np.log(halflife) / np.log(base) - min_index)
        interval = policy[difficulty - 1][index - 1][1]
    elif method == 'ANKI':
        interval = max(2.5 - 0.15 * lapses, 1.2) ** reps
    elif method == 'THRESHOLD':
        interval = - halflife * np.log2(0.9)
    elif method == 'RANDOM':
        interval = random.randint(1, target_halflife)
    return max(1, round(interval))


if __name__ == "__main__":
    scheduler_init()

    for method in ['SSP-MMC', 'MEMORIZE', 'ANKI', 'HALF-LIFE', 'THRESHOLD', 'RANDOM']:
        print("method:", method)
        random.seed(114)

        new_item_per_day = np.array([0.0] * learn_days)
        new_item_per_day_average_per_period = np.array([0.0] * learn_days)
        cost_per_day = np.array([0.0] * learn_days)
        cost_per_day_average_per_period = np.array([0.0] * learn_days)

        learned_per_day = np.array([0.0] * learn_days)
        record_per_day = np.array([0.0] * learn_days)
        meet_target_per_day = np.array([0.0] * learn_days)

        feature_list = ['difficulty', 'halflife', 'p_recall', 'delta_t', 'reps', 'lapses', 'last_date', 'due_date',
                        'r_history', 't_history']

        dtypes = np.dtype([
            ('difficulty', int),
            ('halflife', float),
            ('p_recall', float),
            ('delta_t', int),
            ('reps', int),
            ('lapses', int),
            ('last_date', int),
            ('due_date', int),
            ('r_history', int),
            ('t_history', int),
        ])

        field_map = {
            'difficulty': 0, 'halflife': 1, 'p_recall': 2, 'delta_t': 3, 'reps': 4, 'lapses': 5, 'last_date': 6,
            'due_date': 7,
            'r_history': 8,
            't_history': 9}

        df_memory = pd.DataFrame(np.full(deck_size, np.nan, dtype=dtypes), index=range(deck_size), columns=feature_list)
        df_memory['difficulty'] = df_memory['difficulty'].map(lambda x: random.randint(1, 10))
        df_memory['due_date'] = learn_days

        meet_target = 0

        for day in tqdm(range(learn_days)):
            reviewed = 0
            learned = 0
            day_cost = 0

            df_memory["delta_t"] = day - df_memory["last_date"]
            df_memory["p_recall"] = np.exp2(- df_memory["delta_t"] / df_memory["halflife"])
            need_review = df_memory[df_memory['due_date'] <= day].index
            for idx in need_review:
                if day_cost > day_cost_limit:
                    break

                reviewed += 1
                df_memory.iat[idx, field_map['last_date']] = day
                ivl = df_memory.iat[idx, field_map['delta_t']]
                df_memory.iat[idx, field_map['t_history']] += f',{ivl}'

                halflife = df_memory.iat[idx, field_map['halflife']]
                difficulty = df_memory.iat[idx, field_map['difficulty']]
                p_recall = df_memory.iat[idx, field_map['p_recall']]
                reps = df_memory.iat[idx, field_map['reps']]
                lapses = df_memory.iat[idx, field_map['lapses']]

                if random.random() < p_recall:
                    day_cost += recall_cost

                    df_memory.iat[idx, field_map['r_history']] += ',1'

                    next_halflife = cal_recall_halflife(difficulty, halflife, p_recall)
                    df_memory.iat[idx, field_map['halflife']] = next_halflife
                    df_memory.iat[idx, field_map['reps']] = reps + 1

                    if next_halflife >= target_halflife:
                        meet_target += 1
                        df_memory.iat[idx, field_map['halflife']] = np.inf
                        df_memory.iat[idx, field_map['due_date']] = learn_days
                        continue

                    delta_t = scheduler(difficulty, next_halflife, reps, lapses, method)
                    df_memory.iat[idx, field_map['due_date']] = day + delta_t

                else:
                    day_cost += forget_cost

                    df_memory.iat[idx, field_map['r_history']] += ',0'

                    next_halflife = cal_forget_halflife(difficulty, halflife, p_recall)
                    df_memory.iat[idx, field_map['halflife']] = next_halflife

                    reps = 0
                    lapses = lapses + 1

                    df_memory.iat[idx, field_map['reps']] = reps
                    df_memory.iat[idx, field_map['lapses']] = lapses

                    delta_t = scheduler(difficulty, next_halflife, reps, lapses, method)
                    df_memory.iat[idx, field_map['due_date']] = day + delta_t

                    difficulty = min(difficulty + difficulty_offset, difficulty_limit)
                    df_memory.iat[idx, field_map['difficulty']] = difficulty

            need_learn = df_memory[df_memory['halflife'].isna()].index

            for idx in need_learn:
                if day_cost > day_cost_limit:
                    break
                learned += 1
                day_cost += new_cost
                df_memory.iat[idx, field_map['last_date']] = day

                difficulty = df_memory.iat[idx, field_map['difficulty']]
                p_recall = df_memory.iat[idx, field_map['p_recall']]
                reps = df_memory.iat[idx, field_map['reps']]
                lapses = df_memory.iat[idx, field_map['lapses']]

                halflife = cal_start_halflife(df_memory.iat[idx, field_map['difficulty']])  # halflife
                df_memory.iat[idx, field_map['halflife']] = halflife
                delta_t = scheduler(difficulty, halflife, reps, lapses, method)
                df_memory.iat[idx, field_map['due_date']] = day + delta_t
                df_memory.iat[idx, field_map['r_history']] = '0'
                df_memory.iat[idx, field_map['t_history']] = '0'

            new_item_per_day[day] = learned
            learned_per_day[day] = learned_per_day[day - 1] + learned
            cost_per_day[day] = day_cost

            if day >= period_len:
                new_item_per_day_average_per_period[day] = np.true_divide(new_item_per_day[day - period_len:day].sum(),
                                                                          period_len)
                cost_per_day_average_per_period[day] = np.true_divide(cost_per_day[day - period_len:day].sum(),
                                                                      period_len)
            else:
                new_item_per_day_average_per_period[day] = np.true_divide(new_item_per_day[:day + 1].sum(), day + 1)
                cost_per_day_average_per_period[day] = np.true_divide(cost_per_day[:day + 1].sum(), day + 1)

            record_per_day[day] = df_memory['p_recall'].sum()
            meet_target_per_day[day] = meet_target

        total_learned = int(sum(new_item_per_day))
        total_cost = int(sum(cost_per_day))

        plt.figure(1)
        plt.plot(record_per_day, label=f'{method}', linewidth=0.8)

        plt.figure(2)
        plt.plot(meet_target_per_day, label=f'{method}', linewidth=0.8)
        cost_day = np.argmax(meet_target_per_day >= compare_target)
        if cost_day > 0:
            print(f'cost day: {cost_day}')
            plt.plot(cost_day, compare_target, 'k*', linewidth=2)

        plt.figure(3)
        plt.plot(new_item_per_day_average_per_period, label=f'{method}', linewidth=0.8)

        plt.figure(4)
        plt.plot(cost_per_day_average_per_period, label=f'{method}', linewidth=0.8)

        plt.figure(5)
        plt.plot(learned_per_day, label=f'{method}', linewidth=0.8)

        print('acc learn', total_learned)
        print('meet target', meet_target)

        save = df_memory[df_memory['p_recall'] > 0].copy()
        save['halflife'] = round(save['halflife'], 4)
        save['p_recall'] = round(save['p_recall'], 4)
        save.to_csv(f'./simulation_result/{method}.tsv', index=False, sep='\t')

    plt.figure(1)
    plt.title(f"day cost limit:{day_cost_limit}-learn days:{learn_days}")
    plt.xlabel("days")
    plt.ylabel("SRP")
    # plt.legend()
    plt.grid(True)
    with PdfPages('./simulation_result/SRP.pdf') as pdf:
        # plt.plot(....)
        pdf.savefig()
    plt.figure(2)
    plt.plot((0, learn_days), (compare_target, compare_target), color='black', linestyle='dotted', linewidth=0.8)
    plt.title(f"day cost limit:{day_cost_limit}-learn days:{learn_days}")
    plt.xlabel("days")
    plt.ylabel("THR")
    plt.legend()
    plt.grid(True)
    with PdfPages('./simulation_result/THR.pdf') as pdf:
        # plt.plot(....)
        pdf.savefig()
    plt.figure(3)
    plt.title(f"day cost limit:{day_cost_limit}-learn days:{learn_days}")
    plt.xlabel("days")
    plt.ylabel(f"new item per day({period_len} days average)")
    plt.legend()
    plt.grid(True)
    with PdfPages('./simulation_result/new.pdf') as pdf:
        # plt.plot(....)
        pdf.savefig()
    plt.figure(4)
    plt.title(f"day cost limit:{day_cost_limit}-learn days:{learn_days}")
    plt.xlabel("days")
    plt.ylabel(f"cost per day({period_len} days average)")
    # plt.legend()
    plt.grid(True)
    with PdfPages('./simulation_result/cost.pdf') as pdf:
        # plt.plot(....)
        pdf.savefig()
    plt.figure(5)
    plt.title(f"day cost limit:{day_cost_limit}-learn days:{learn_days}")
    plt.xlabel("days")
    plt.ylabel(f"WTL")
    # plt.legend()
    plt.grid(True)
    # plt.show()
    with PdfPages('./simulation_result/WTL.pdf') as pdf:
        # plt.plot(....)
        pdf.savefig()
