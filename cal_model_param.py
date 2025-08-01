import time

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import plotly.express as px
import joblib

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.rcParams['figure.dpi'] = 300


def fit_recall_halflife(raw):
    print('fit_recall_halflife_dhp')
    raw['log_hinc'] = raw['halflife_increase'].map(lambda x: np.log(x - 1))
    raw['log_h'] = raw['last_halflife'].map(lambda x: np.log(x))
    raw['fi'] = raw['last_p_recall'].map(lambda x: 1 - x)
    raw['log_fi'] = raw['last_p_recall'].map(lambda x: np.log(1 - x))
    raw['log_d'] = raw['d'].map(lambda x: np.log(x))
    raw['log_delta_h'] = np.log(raw['halflife'] - raw['last_halflife'])
    corr = raw.select_dtypes(exclude=['object']).corr()
    print(corr)

    X = raw[['log_d', 'log_h', 'log_fi']]
    Y = raw[['log_hinc']]

    lr = LinearRegression()
    lr.fit(X, Y, sample_weight=raw['group_cnt'])
    print('Intercept: ', lr.intercept_)
    print('Slope: ', lr.coef_)

    y_pred = (np.exp(lr.predict(X)) + 1) * raw[['last_halflife']].values
    Y = (np.exp(Y) + 1) * raw[['last_halflife']].values

    print("Explained Variance Score: ",
          metrics.explained_variance_score(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Mean Absolute Error:',
          metrics.mean_absolute_error(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Mean Squared Error:',
          metrics.mean_squared_error(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(Y, y_pred, sample_weight=raw['group_cnt'])))
    print('Mean Absolute Percentage Error:',
          np.sqrt(metrics.mean_absolute_percentage_error(Y, y_pred, sample_weight=raw['group_cnt'])))

    raw['predict_halflife_dhp'] = y_pred

    print('fit_recall_halflife_hlr')
    lr = joblib.load('fit_result/clf.pkl')
    X = raw[['right', 'wrong', 'd']]
    Y = raw[['log_halflife']]
    print('Intercept: ', lr.intercept_)
    print('Slope: ', lr.coef_)

    y_pred = np.exp(lr.predict(X))
    Y = np.exp(Y)

    print("Explained Variance Score: ",
          metrics.explained_variance_score(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Mean Absolute Error:',
          metrics.mean_absolute_error(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Mean Squared Error:',
          metrics.mean_squared_error(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(Y, y_pred, sample_weight=raw['group_cnt'])))
    print('Mean Absolute Percentage Error:',
          np.sqrt(metrics.mean_absolute_percentage_error(Y, y_pred, sample_weight=raw['group_cnt'])))

    raw['predict_halflife_hlr'] = y_pred
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=raw['halflife'], y=raw['predict_halflife_dhp'], marker_size=np.log(raw['group_cnt']),
                   mode='markers',
                   name='DHP'))
    fig.add_trace(
        go.Scatter(x=raw['halflife'], y=raw['predict_halflife_hlr'], marker_size=np.log(raw['group_cnt']),
                   mode='markers',
                   name='HLR', opacity=0.7))
    fig.update_xaxes(title_text='observed half-life after recall', title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_yaxes(title_text='predicted half-life after recall', title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_layout(legend_font_size=14, margin_t=10)
    fig.write_image(f"plot/fit_recall_halflife.pdf")
    time.sleep(3)
    fig.write_image(f"plot/fit_recall_halflife.pdf")
    # fig.show()


def fit_forget_halflife(raw):
    print('fit_forget_halflife_dhp')
    raw['log_h'] = raw['last_halflife'].map(lambda x: np.log(x))
    raw['fi'] = raw['last_p_recall'].map(lambda x: 1 - x)
    raw['log_fi'] = raw['last_p_recall'].map(lambda x: np.log(1 - x))
    raw['log_d'] = raw['d'].map(lambda x: np.log(x))

    print(raw.select_dtypes(exclude=['object']).corr())

    X = raw[['log_d', 'log_h', 'log_fi']]
    Y = raw[['log_halflife']]

    lr = LinearRegression()
    lr.fit(X, Y, sample_weight=raw['group_cnt'])
    print('Intercept: ', lr.intercept_)
    print('Slope: ', lr.coef_)

    y_pred = np.exp(lr.predict(X))
    Y = np.exp(Y)

    print("Explained Variance Score: ",
          metrics.explained_variance_score(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Mean Absolute Error:',
          metrics.mean_absolute_error(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Mean Squared Error:',
          metrics.mean_squared_error(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(Y, y_pred, sample_weight=raw['group_cnt'])))
    print('Mean Absolute Percentage Error:',
          np.sqrt(metrics.mean_absolute_percentage_error(Y, y_pred, sample_weight=raw['group_cnt'])))

    raw['predict_halflife_dhp'] = y_pred

    print('fit_forget_halflife_hlr')
    lr = joblib.load('fit_result/clf.pkl')
    X = raw[['right', 'wrong', 'd']]
    Y = raw[['log_halflife']]
    print('Intercept: ', lr.intercept_)
    print('Slope: ', lr.coef_)

    y_pred = np.exp(lr.predict(X))
    Y = np.exp(Y)

    print("Explained Variance Score: ",
          metrics.explained_variance_score(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Mean Absolute Error:',
          metrics.mean_absolute_error(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Mean Squared Error:',
          metrics.mean_squared_error(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(Y, y_pred, sample_weight=raw['group_cnt'])))
    print('Mean Absolute Percentage Error:',
          np.sqrt(metrics.mean_absolute_percentage_error(Y, y_pred, sample_weight=raw['group_cnt'])))

    raw['predict_halflife_hlr'] = y_pred
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=raw['halflife'], y=raw['predict_halflife_dhp'], marker_size=np.log(raw['group_cnt']),
                   mode='markers',
                   name='DHP'))
    fig.add_trace(
        go.Scatter(x=raw['halflife'], y=raw['predict_halflife_hlr'], marker_size=np.log(raw['group_cnt']),
                   mode='markers',
                   name='HLR', opacity=0.7))
    fig.update_xaxes(title_text='observed half-life after forget', title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_yaxes(title_text='predicted half-life after forget', title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_layout(legend_font_size=14, margin_t=10)
    fig.write_image(f"plot/fit_forget_halflife.pdf")
    time.sleep(3)
    fig.write_image(f"plot/fit_forget_halflife.pdf")
    # fig.show()


def fit_hlr_model(raw):
    print('hlr_model_fitting')
    corr = raw.select_dtypes(exclude=['object']).corr()
    print(corr)

    X = raw[['right', 'wrong', 'd']]
    Y = raw[['log_halflife']]

    lr = LinearRegression()
    lr.fit(X, Y, sample_weight=raw['group_cnt'])
    print('Intercept: ', lr.intercept_)
    print('Slope: ', lr.coef_)
    joblib.dump(lr, 'fit_result/clf.pkl')

    y_pred = np.exp(lr.predict(X))
    Y = np.exp(Y)

    print("Explained Variance Score: ",
          metrics.explained_variance_score(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Mean Absolute Error:',
          metrics.mean_absolute_error(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Mean Squared Error:',
          metrics.mean_squared_error(Y, y_pred, sample_weight=raw['group_cnt']))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(Y, y_pred, sample_weight=raw['group_cnt'])))
    print('Mean Absolute Percentage Error:',
          np.sqrt(metrics.mean_absolute_percentage_error(Y, y_pred, sample_weight=raw['group_cnt'])))

    raw['predict_halflife'] = y_pred
    fig = px.scatter(raw, x='halflife', y='predict_halflife', size='group_cnt')
    fig.update_xaxes(title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=14))
    fig.write_image(f"plot/fit_hlr_model.pdf", width=600, height=360)
    time.sleep(3)
    fig.write_image(f"plot/fit_hlr_model.pdf", width=600, height=360)
    # fig.show()


if __name__ == "__main__":
    raw = pd.read_csv('./data/halflife_for_visual.tsv', sep='\t')

    raw['right'] = raw['r_history'].str.count('1')
    raw['wrong'] = raw['r_history'].str.count('0')
    raw['right'] = raw['right'].map(lambda x: np.sqrt(x + 1))
    raw['wrong'] = raw['wrong'].map(lambda x: np.sqrt(x + 1))
    raw['log_halflife'] = raw['halflife'].map(lambda x: np.log(x))

    raw.drop_duplicates(inplace=True)
    raw = raw[raw['group_cnt'] > 1000]
    fit_hlr_model(raw[(raw['last_recall'] == 0) | (raw['halflife_increase'] > 1) & (raw['last_recall'] == 1) & (
                raw['r_history'].str.count('0') == 1)].copy())
    fit_recall_halflife(
        raw[(raw['halflife_increase'] > 1) & (raw['last_recall'] == 1) & (raw['r_history'].str.count('0') == 1)].copy())
    fit_forget_halflife(raw[raw['last_recall'] == 0].copy())
