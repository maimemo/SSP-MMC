import time

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from model.utils import *

plt.style.use('seaborn-whitegrid')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.5, y=1.5, z=1.25)
)


# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.rcParams['figure.figsize'] = (8.0, 6.0)
# plt.rcParams['figure.dpi'] = 300


def difficulty_visualize():
    raw = pd.read_csv('./data/opensource_dataset_difficulty.tsv', sep='\t')
    u = raw['p_recall'].mean()
    std = raw['p_recall'].std()
    print(u, std)
    # plt.hist(raw['p_recall'], bins=20, rwidth=0.8)
    # plt.xlabel('probability of recall')
    # plt.ylabel('samples')
    # plt.savefig("plot/distribution_p.eps")
    # plt.cla()
    # plt.hist(raw['d'], rwidth=0.8)
    # plt.xlabel('difficulty')
    # plt.ylabel('samples')
    # plt.savefig("plot/distribution_d.eps")
    # plt.cla()
    fig = px.histogram(raw, x="p_recall", nbins=20)
    fig.update_xaxes(title_text='probability of P(recall)', title_font=dict(size=18), tickfont=dict(size=14),
                     range=[0, 1])
    fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_layout(bargap=0.2, margin_t=10)
    # fig.show()
    fig.write_image("plot/distribution_p.pdf", width=600, height=360)
    time.sleep(3)
    fig.write_image("plot/distribution_p.pdf", width=600, height=360)
    fig = px.histogram(raw, x="d", text_auto=True)
    fig.update_xaxes(title_text='difficulty', title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_layout(bargap=0.2, margin_t=10)
    # fig.show()
    fig.write_image("plot/distribution_d.pdf", width=600, height=360)
    time.sleep(3)
    fig.write_image("plot/distribution_d.pdf", width=600, height=360)


def forgetting_curve_visualize():
    raw = pd.read_csv('./data/halflife_for_fit.tsv', sep='\t')
    filters = [(4, '0,1', '0,1'), (4, '0,1,1', '0,1,3'), (4, '0,1,1', '0,1,4'), (4, '0,1,1', '0,1,5')]
    fig = go.Figure()
    color = ['blue', 'red', 'green', 'orange']
    for i, f in enumerate(filters):
        d = f[0]
        r_history = f[1]
        t_history = f[2]
        tmp = raw[(raw['d'] == d) & (raw['r_history'] == r_history) & (raw['t_history'] == t_history)].copy()
        tmp.sort_values(by=['delta_t'], inplace=True)
        tmp['size'] = np.log(tmp['total_cnt'])
        halflife = tmp['halflife'].values[0]
        tmp['fit_p_recall'] = np.power(2, -tmp['delta_t'] / halflife)
        fig.add_trace(
            go.Scatter(x=tmp['delta_t'], y=tmp['fit_p_recall'], mode='lines', name=f'halflife={halflife:.2f}'))
        fig.add_trace(go.Scatter(x=tmp['delta_t'], y=tmp['p_recall'],
                                 mode='markers', marker_size=tmp['size'],
                                 name=r'$d=%d|\boldsymbol r_{1:i-1}=%s|\boldsymbol{\Delta t}_{1:i-1}=%s$' % (
                                     d, r_history, t_history)))
        fig.update_traces(marker_color=color[i], selector=dict(name=f'halflife={halflife:.2f}'))
        fig.update_traces(marker_color=color[i],
                          selector=dict(name=r'$d=%d|\boldsymbol r_{1:i-1}=%s|\boldsymbol{\Delta t}_{1:i-1}=%s$' % (
                              d, r_history, t_history)))
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    ))
    fig.update_xaxes(title_text='delta_t', title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_yaxes(title_text='p_recall', title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_layout(margin_t=10)
    # fig.show()
    fig.write_image(f"plot/forgetting_curve.pdf", width=600, height=360)
    time.sleep(3)
    fig.write_image(f"plot/forgetting_curve.pdf", width=600, height=360)


def raw_data_visualize():
    raw = pd.read_csv('./data/halflife_for_visual.tsv', sep='\t')
    raw = raw[raw['group_cnt'] > 1000]
    raw['label'] = raw['r_history'] + '/' + raw['t_history']
    raw['log(last_halflife)'] = np.log(raw['last_halflife'])
    raw['log(halflife)'] = np.log(raw['halflife'])
    raw['log(d)'] = np.log(raw['d'])

    fig = px.scatter_3d(raw, x='last_p_recall', y='log(last_halflife)',
                        z='log(halflife)', color='d',
                        hover_name='label')
    fig.update_traces(marker_size=2, selector=dict(type='scatter3d'))
    fig.update_scenes(xaxis_autorange="reversed")
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_coloraxes(colorbar_tickfont_size=24)
    fig.write_image(f"plot/DHP_model_raw.pdf", width=1000, height=1000)
    fig.show()

    raw = raw[raw['r_history'].str.endswith('1')]
    raw = raw[raw['r_history'].str.count('0') == 1]

    fig = px.scatter_3d(raw, x='last_p_recall', y='log(last_halflife)',
                        z='log(d)', color='log(halflife)',
                        hover_name='label')

    d_array, h_array, p_array = np.mgrid[1:10:10j, raw['last_halflife'].min():raw['last_halflife'].max():500j,
                                raw['last_p_recall'].min():raw['last_p_recall'].max():100j]
    value = np.log(cal_recall_halflife(d_array, h_array, p_array))
    fig.add_isosurface(
        x=p_array.flatten(),
        y=np.log(h_array.flatten()),
        z=np.log(d_array.flatten()),
        value=value.flatten(),
        isomin=raw['log(halflife)'].min(),
        isomax=raw['log(halflife)'].max(),
        surface_count=10,
        colorbar_nticks=10, showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False))
    fig.update_traces(marker_size=2, selector=dict(type='scatter3d'))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_tickfont_size=18, selector=dict(type='isosurface'))
    fig.write_image(f"plot/DHP_model.pdf", width=1000, height=1000)
    fig.show()


def dhp_model_visualize():
    h_array = np.arange(0.5, 900.5, 1)  # 03
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
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=5),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_tickfont_size=18, selector=dict(type='surface'))
    # fig.write_html(f"./plot/DHP_recall_model.html")
    fig.write_image(f"plot/DHP_recall_model.pdf", width=1000, height=1000)
    # fig.show()
    surface = [
        go.Surface(x=h_array, y=p_array, z=cal_recall_halflife(diff, h_array, p_array) / h_array,
                   surfacecolor=np.full_like(h_array, diff), cmin=0.5, cmax=10.5) for diff in
        range(1, 11)]
    fig = go.Figure(data=surface)
    fig.update_layout(scene=dict(
        xaxis_title='last_halflife',
        yaxis_title='last_p_recall',
        zaxis_title='halflife/last_halflife'))
    # fig.write_html(f"./plot/DHP_recall_inc_model.html")
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=5),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_tickfont_size=18, selector=dict(type='surface'))
    fig.write_image(f"plot/DHP_recall_inc_model.pdf", width=1000, height=1000)
    # fig.show()
    surface = [
        go.Surface(x=h_array, y=p_array, z=cal_forget_halflife(diff, h_array, p_array),
                   surfacecolor=np.full_like(h_array, diff), cmin=0.5, cmax=10.5) for diff in
        range(1, 11)]
    fig = go.Figure(data=surface)
    fig.update_layout(scene=dict(
        xaxis_title='last_halflife',
        yaxis_title='last_p_recall',
        zaxis_title='halflife'))
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=5),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_tickfont_size=18, selector=dict(type='surface'))
    # fig.write_html(f"./plot/DHP_forget_model.html")
    fig.write_image(f"plot/DHP_forget_model.pdf", width=1000, height=1000)
    # fig.show()


def policy_action_visualize():
    df = pd.DataFrame()
    for d in range(1, 19):
        dataset = pd.read_csv(f"./algo/result/ivl-{d}.csv", header=None, index_col=None)
        dataset.columns = ['halflife', f'{d}']
        df = pd.concat([df, dataset[f'{d}']], axis=1)
    halflife = dataset['halflife'].values[30:-1]
    d = np.arange(1, 19, 1)
    d, halflife = np.meshgrid(d, halflife)
    delta_t = df.values[30:-1, :]
    fig = go.Figure(data=go.Surface(x=d, y=halflife, z=delta_t))
    fig.update_scenes(xaxis_autorange="reversed")
    fig.update_scenes(yaxis_autorange="reversed")
    fig.update_layout(scene=dict(
        xaxis_title='d',
        yaxis_title='halflife',
        zaxis_title='delta_t'))
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=4),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_tickfont_size=18, selector=dict(type='surface'))
    fig.write_image(f"plot/policy_delta_t.pdf", width=1000, height=1000)
    # fig.show()

    df = pd.DataFrame()
    for d in range(1, 19):
        dataset = pd.read_csv(f"./algo/result/cost-{d}.csv", header=None, index_col=None)
        dataset.columns = ['halflife', f'{d}']
        df = pd.concat([df, dataset[f'{d}']], axis=1)
    halflife = dataset['halflife'].values[30:-1]
    d = np.arange(1, 19, 1)
    d, halflife = np.meshgrid(d, halflife)
    cost = df.values[30:-1, :]
    fig = go.Figure(data=go.Surface(x=d, y=halflife, z=cost))
    fig.update_scenes(xaxis_autorange="reversed")
    fig.update_layout(scene=dict(
        xaxis_title='d',
        yaxis_title='halflife',
        zaxis_title='cost'))
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=4),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_tickfont_size=18, selector=dict(type='surface'))
    fig.write_image(f"plot/policy_cost.pdf", width=1000, height=1000)
    # fig.show()

    df = pd.DataFrame()
    for d in range(1, 19):
        dataset = pd.read_csv(f"./algo/result/recall-{d}.csv", header=None, index_col=None)
        dataset.columns = ['halflife', f'{d}']
        df = pd.concat([df, dataset[f'{d}']], axis=1)
    halflife = dataset['halflife'].values[30:-1]
    d = np.arange(1, 19, 1)
    d, halflife = np.meshgrid(d, halflife)
    p_recall = df.values[30:-1, :]
    fig = go.Figure(data=go.Surface(x=d, y=halflife, z=p_recall))
    fig.update_scenes(xaxis_autorange="reversed")
    fig.update_scenes(yaxis_autorange="reversed")
    fig.update_layout(scene=dict(
        xaxis_title='d',
        yaxis_title='halflife',
        zaxis_title='p_recall'))
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=4),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_tickfont_size=18, selector=dict(type='surface'))
    fig.write_image(f"plot/policy_p_recall.pdf", width=1000, height=1000)
    # fig.show()


if __name__ == "__main__":
    difficulty_visualize()
    forgetting_curve_visualize()
    raw_data_visualize()
    dhp_model_visualize()
    policy_action_visualize()
