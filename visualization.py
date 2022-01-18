import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from model.utils import *

plt.style.use('seaborn-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.rcParams['figure.dpi'] = 300


def raw_data_visualize():
    raw = pd.read_csv('./data/halflife_for_visual.tsv', sep='\t')
    raw['r_history'] = raw['r_history'].map(str)
    raw['label'] = raw['r_history'] + '/' + raw['t_history']
    raw = raw[raw['r_history'].str.endswith('1')]
    raw = raw[raw['r_history'].str.count('3') == 1]
    raw['log(last_halflife)'] = np.log(raw['last_halflife'])
    raw['log(halflife)'] = np.log(raw['halflife'])
    raw['log(difficulty)'] = np.log(raw['difficulty'])

    fig = px.scatter_3d(raw, x='log(last_halflife)', y='last_p_recall',
                        z='log(difficulty)', color='log(halflife)',
                        hover_name='label')

    d_array, h_array, p_array = np.mgrid[1:10:10j, raw['last_halflife'].min():raw['last_halflife'].max():500j,
                                raw['last_p_recall'].min():raw['last_p_recall'].max():100j]
    value = np.log(cal_recall_halflife(d_array, h_array, p_array))
    fig.add_isosurface(
        x=np.log(h_array.flatten()),
        y=p_array.flatten(),
        z=np.log(d_array.flatten()),
        value=value.flatten(),
        isomin=raw['log(halflife)'].min(),
        isomax=raw['log(halflife)'].max(),
        surface_count=10,
        colorbar_nticks=10, showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False))
    fig.update_traces(marker_size=2, selector=dict(type='scatter3d'))
    fig.write_html("./plot/DHP_model.html")
    fig.show()


def dhp_model_visualize():
    h_array = np.arange(0.5, 500.5, 1)  # 03
    p_array = np.arange(0.97, 0.3, -0.01)  # 03
    h_array, p_array = np.meshgrid(h_array, p_array)
    surface = [
        go.Surface(x=h_array, y=p_array, z=cal_recall_halflife(diff, h_array, p_array),
                   surfacecolor=np.full_like(h_array, diff), cmin=0.5, cmax=10.5) for diff in
        range(1, 11)]
    fig = go.Figure(data=surface)
    fig.update_layout(scene=dict(
        xaxis_title='last_halflife',
        yaxis_title='last_p_recall',
        zaxis_title='halflife'))
    fig.write_html(f"./plot/DHP_recall_model.html")
    fig.show()
    surface = [
        go.Surface(x=h_array, y=p_array, z=cal_recall_halflife(diff, h_array, p_array) / h_array,
                   surfacecolor=np.full_like(h_array, diff), cmin=0.5, cmax=10.5) for diff in
        range(1, 11)]
    fig = go.Figure(data=surface)
    fig.update_layout(scene=dict(
        xaxis_title='last_halflife',
        yaxis_title='last_p_recall',
        zaxis_title='halflife/last_halflife'))
    fig.write_html(f"./plot/DHP_recall_inc_model.html")
    fig.show()
    surface = [
        go.Surface(x=h_array, y=p_array, z=cal_forget_halflife(diff, h_array, p_array),
                   surfacecolor=np.full_like(h_array, diff), cmin=0.5, cmax=10.5) for diff in
        range(1, 11)]
    fig = go.Figure(data=surface)
    fig.update_layout(scene=dict(
        xaxis_title='last_halflife',
        yaxis_title='last_p_recall',
        zaxis_title='halflife'))
    fig.write_html(f"./plot/DHP_forget_model.html")
    fig.show()


def policy_action_visualize():
    df = pd.DataFrame()
    for d in range(1, 21):
        dataset = pd.read_csv(f"./algo/result/ivl-{d}.csv", header=None, index_col=None)
        dataset.columns = ['halflife', f'{d}']
        df = pd.concat([df, dataset[f'{d}']], axis=1)
    halflife = dataset['halflife'].values[30:-1]
    difficulty = np.arange(1, 21, 1)
    difficulty, halflife = np.meshgrid(difficulty, halflife)
    delta_t = df.values[30:-1, :]
    fig = go.Figure(data=go.Surface(x=halflife, y=difficulty, z=delta_t))
    fig.update_layout(scene=dict(
        xaxis_title='halflife',
        yaxis_title='difficulty',
        zaxis_title='delta_t'))
    fig.write_html(f"./plot/policy_delta_t.html")
    fig.show()

    df = pd.DataFrame()
    for d in range(1, 21):
        dataset = pd.read_csv(f"./algo/result/cost-{d}.csv", header=None, index_col=None)
        dataset.columns = ['halflife', f'{d}']
        df = pd.concat([df, dataset[f'{d}']], axis=1)
    halflife = dataset['halflife'].values[30:-1]
    difficulty = np.arange(1, 21, 1)
    difficulty, halflife = np.meshgrid(difficulty, halflife)
    cost = df.values[30:-1, :]
    fig = go.Figure(data=go.Surface(x=halflife, y=difficulty, z=cost))
    fig.update_layout(scene=dict(
        xaxis_title='halflife',
        yaxis_title='difficulty',
        zaxis_title='cost'))
    fig.write_html(f"./plot/policy_cost.html")
    fig.show()

    df = pd.DataFrame()
    for d in range(1, 21):
        dataset = pd.read_csv(f"./algo/result/recall-{d}.csv", header=None, index_col=None)
        dataset.columns = ['halflife', f'{d}']
        df = pd.concat([df, dataset[f'{d}']], axis=1)
    halflife = dataset['halflife'].values[30:-1]
    difficulty = np.arange(1, 21, 1)
    difficulty, halflife = np.meshgrid(difficulty, halflife)
    p_recall = df.values[30:-1, :]
    fig = go.Figure(data=go.Surface(x=halflife, y=difficulty, z=p_recall))
    fig.update_layout(scene=dict(
        xaxis_title='halflife',
        yaxis_title='difficulty',
        zaxis_title='p_recall'))
    fig.write_html(f"./plot/policy_p_recall.html")
    fig.show()


if __name__ == "__main__":
    raw_data_visualize()
    dhp_model_visualize()
    policy_action_visualize()
