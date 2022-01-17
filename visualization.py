import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

plt.style.use('seaborn-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.rcParams['figure.dpi'] = 300

if __name__ == "__main__":
    raw = pd.read_csv('./data/halflife_for_visual.tsv', sep='\t')
    raw = raw[raw['group_cnt'] > 100000]
    raw['r_history'] = raw['r_history'].map(str)
    raw['label'] = raw['r_history'] + '/' + raw['t_history']
    raw = raw[raw['r_history'].str.endswith('1')]
    fig = px.scatter_3d(raw, x='last_halflife', y='last_p_recall',
                        z='halflife', color='difficulty', log_x=True, log_z=True,
                        hover_name='label')
    fig.update_traces(marker_size=2, selector=dict(type='scatter3d'))
    fig.write_html("./plot/DHP_model.html")
    fig.show()
