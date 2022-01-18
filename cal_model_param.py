import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics

plt.style.use('seaborn-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.rcParams['figure.dpi'] = 300

if __name__ == "__main__":
    raw = pd.read_csv('./data/halflife_for_visual.tsv', sep='\t')
    raw.drop_duplicates(inplace=True)
    raw = raw[raw['group_cnt'] > 1000]
    raw['r_history'] = raw['r_history'].map(str)
    raw_copy = raw.copy()
    raw.dropna(inplace=True)
    raw = raw[(raw['halflife_increase'] > 1) & (raw['last_recall'] == 1)]
    raw = raw[raw['r_history'].str.count('3') == 1]

    raw['log_hinc'] = raw['halflife_increase'].map(lambda x: np.log(x - 1))
    raw['log_h'] = raw['last_halflife'].map(lambda x: np.log(x))
    raw['log_halflife'] = raw['halflife'].map(lambda x: np.log(x))
    raw['fi'] = raw['last_p_recall'].map(lambda x: 1 - x)
    raw['log_fi'] = raw['last_p_recall'].map(lambda x: np.log(1 - x))
    raw['log_d'] = raw['difficulty'].map(lambda x: np.log(x))
    raw['log_delta_h'] = np.log(raw['halflife'] - raw['last_halflife'])
    corr = raw.corr()
    print(corr)

    X = raw[['log_d', 'log_h', 'log_fi']]
    Y = raw[['log_hinc']]

    lr = LinearRegression()
    lr.fit(X, Y, sample_weight=raw['group_cnt'])
    print('Intercept: ', lr.intercept_)
    print('Slope: ', lr.coef_)

    y_pred = lr.predict(X)

    print("Explained Variance Score: ",
          metrics.explained_variance_score(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Mean Absolute Error:',
          metrics.mean_absolute_error(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Mean Squared Error:',
          metrics.mean_squared_error(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(Y, y_pred, sample_weight=raw['group_cnt'])))

    raw['predict_log_hinc'] = y_pred
    outlier = raw[raw['log_hinc'] < -1]

    plt.scatter(Y, y_pred)
    plt.xlabel('true log(halflife)')
    plt.ylabel('predict log(halflife)')
    plt.show()

    a = lr.intercept_[0]
    b, c, d = lr.coef_[0]

    raw_copy.dropna(inplace=True)

    raw_copy = raw_copy[raw_copy['last_recall'] == 0]
    raw_copy['log_h'] = raw_copy['last_halflife'].map(lambda x: np.log(x))
    raw_copy['fi'] = raw_copy['last_p_recall'].map(lambda x: 1 - x)
    raw_copy['log_fi'] = raw_copy['last_p_recall'].map(lambda x: np.log(1 - x))
    raw_copy['log_d'] = raw_copy['difficulty'].map(lambda x: np.log(x))
    raw_copy['log_halflife'] = raw_copy['halflife'].map(lambda x: np.log(x))

    print(raw_copy.corr())

    X = raw_copy[['log_d', 'log_h', 'log_fi']]
    Y = raw_copy[['log_halflife']]

    lr = LinearRegression()
    lr.fit(X, Y, sample_weight=raw_copy['group_cnt'])
    print('Intercept: ', lr.intercept_)
    print('Slope: ', lr.coef_)

    y_pred = lr.predict(X)

    print("Explained Variance Score: ",
          metrics.explained_variance_score(Y, y_pred, sample_weight=raw_copy['group_cnt']))
    print('Mean Absolute Error:',
          metrics.mean_absolute_error(Y, y_pred, sample_weight=raw_copy['group_cnt']))
    print('Mean Squared Error:',
          metrics.mean_squared_error(Y, y_pred, sample_weight=raw_copy['group_cnt']))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(Y, y_pred, sample_weight=raw_copy['group_cnt'])))

    plt.scatter(Y, y_pred)
    plt.xlabel('true log(halflife)')
    plt.ylabel('predict log(halflife)')
    plt.show()
