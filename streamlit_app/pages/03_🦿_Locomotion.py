import streamlit as st
from PIL import Image
import codecs
import streamlit.components.v1 as components
from utils import inject_custom_css
import streamlit as st
from streamlit_plotly_events import plotly_events
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import typing as tp

plt.style.use('default')

shapes=[
    dict(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(
            color="Black",
            width=2,
        ),
    )
]

import colorsys

def interpolate_color(color1, color2, factor):
    """Interpolates between two RGB colors. Factor is between 0 and 1."""
    color1 = colorsys.rgb_to_hls(int(color1[1:3], 16)/255.0, int(color1[3:5], 16)/255.0, int(color1[5:], 16)/255.0)
    color2 = colorsys.rgb_to_hls(int(color2[1:3], 16)/255.0, int(color2[3:5], 16)/255.0, int(color2[5:], 16)/255.0)
    new_color = [color1[i] * (1 - factor) + color2[i] * factor for i in range(3)]
    new_color = colorsys.hls_to_rgb(*new_color)
    return '#{:02x}{:02x}{:02x}'.format(int(new_color[0]*255), int(new_color[1]*255), int(new_color[2]*255))


color1 = "#fa7659"
color2 = "#6dafd7"

def plot_pareto(dict_results: tp.Dict):
    keys = list(dict_results["wa"][0].keys())
    lambda_key, reward2_key, reward1_key = keys

    # Series for "wa"
    dict_results["wa"] = [x for i,x in enumerate(dict_results["wa"]) if i%2==0]
    lambda_values_wa = [item[lambda_key] for item in dict_results["wa"]][::-1]
    reward1_values_wa = [item[reward1_key] for item in dict_results["wa"]][::-1]
    reward2_values_wa = [item[reward2_key] for item in dict_results["wa"]][::-1]

    # Series for "init"
    reward1_values_init = [item[reward1_key] for item in dict_results["init"]]
    reward2_values_init = [item[reward2_key] for item in dict_results["init"]]

    layout = go.Layout(autosize=False,width=1000,height=1000)
    fig = go.Figure(layout=layout)

    for i in range(len(reward1_values_wa) - 1):
        fig.add_trace(go.Scatter(
            x=reward1_values_wa[i:i+2],
            y=reward2_values_wa[i:i+2],
            mode='lines',
            hoverinfo='skip',
            line=dict(
                color=interpolate_color(color1, color2, i/(len(reward1_values_wa)-1)),
                width=2
            ),
            showlegend=False
        ))

    # Plot for "wa"
    fig.add_trace(
        go.Scatter(
            x=reward1_values_wa,
            y=reward2_values_wa,
            mode='markers',
            name='Rewarded soups: 0≤λ≤1',
            hoverinfo='text',
            hovertext=[f'λ={lmbda}' for lmbda in lambda_values_wa],
            marker=dict(
                color=[
                    interpolate_color(color1, color2, i / len(lambda_values_wa))
                    for i in range(len(lambda_values_wa))
                ],
                size=10
            )
        )
    )

    # Plot for "morl"
    fig.add_trace(
        go.Scatter(
            x=[6400.],
            y=[3300.],
            mode='markers',
            name='MORL: μ=0.5',
            hoverinfo='skip',
            marker=dict(color='#A45EE9', size=15, symbol="star"),
        )
    )
    # Plot for "init"
    fig.add_trace(
        go.Scatter(
            x=reward1_values_init,
            y=reward2_values_init,
            mode='markers',
            name='Pre-trained init',
            hoverinfo='skip',
            marker=dict(color='#9f9bc8', size=15, symbol="star"),
        )
    )

    fig.update_layout(
        xaxis=dict(
            range=[3000, 7000],
            nticks=6,
            showticklabels=True,
            ticks='outside',
            tickfont=dict(size=18,),
            title=dict(text="Risky reward", font=dict(size=18), standoff=10),
            showgrid=False,
            zeroline=False,
            hoverformat='.2f'
        ),
        yaxis=dict(
            range=[-1000, 4500],
            nticks=7,
            showticklabels=True,
            ticks='outside',
            tickfont=dict(size=18,),
            title=dict(text="Cautious reward", font=dict(size=18), standoff=10),
            showgrid=False,
            zeroline=False,
            hoverformat='.2f'
        ),
        font=dict(family="Roboto", size=12, color="Black"),
        hovermode='x unified',
        autosize=False,
        width=500,
        height=500,
        margin=dict(l=100, r=50, b=150, t=20, pad=0),
        paper_bgcolor="White",
        plot_bgcolor="White",
        shapes=shapes,
        legend=dict(
            x=0.5,
            y=0.03,
            traceorder="normal",
            font=dict(family="Roboto", size=12, color="black"),
            bgcolor="White",
            bordercolor="Black",
            borderwidth=1
        )
    )

    return fig

def run():

    st.write(
        f"""
        <link href='http://fonts.googleapis.com/css?family=Roboto' rel='stylesheet' type='text/css'>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML">
        </script>
        <h3 style='text-align: left;';>Making humanoid run more naturally with diverse engineered rewards</h3>""",unsafe_allow_html=True)

    st.markdown(
        r"""
Teaching humanoids to walk in a human-like manner serves as a benchmark to evaluate RL strategies for continuous control. One of the key challenges is shaping a suitable proxy reward, given the intricate coordination and balance involved in human locomotion. It is standard to consider the dense reward at each timestep: ${r(t)=velocity-\alpha \times \sum_t a^{2}_{t}}$, controlling the agent's velocity while penalizing wide actions. Yet, the penalty coefficient $\alpha$ is challenging to set. To tackle this, we devised two rewards in the Brax physics engine: a *risky* one with $\alpha=0$, and a *cautious* one $\alpha=1$.

Below in the interactive animation, you will see the humanoids trained with these two rewards: the humanoid for $\alpha=0$ is the fastest but the most chaotic, while the one for $\alpha=1$ is more cautious but slower. For intermediate values of $\lambda$, the policy is obtained by linear interpolation of those extreme weights, arguably resulting in smoother motion patterns.
""", unsafe_allow_html=True
    )
    st.markdown("""<h3 style='text-align: left;';>Click on a rewarded soup point!</h3>""",unsafe_allow_html=True)

    files = []
    for i in range(21):
        filename = f'streamlit_app/data/locomotion/trajectories/{i}.html'
        files.append(codecs.open(filename, "r", "utf-8").read())
    files = [x for i,x in enumerate(files) if i%2==0]

    row_0_1,row_0_2,row_0_3,row_0_4  = st.columns([3,1,1,1])
    with row_0_1:
        with open("streamlit_app/data/locomotion/pareto/humanoid_averse_taker_with_morl.pkl","rb") as f:
            dict_results = pickle.load(f)
        fig = plot_pareto(dict_results)
        onclick = plotly_events(fig, click_event=True)
    with row_0_4:
        st.markdown(f"""<div style='text-align: left; color: {color1}; font-size: 30px; padding-right: 40px; padding-top: 20px;'>λ=1.0</div>""",unsafe_allow_html=True)
        components.html(files[-1],width=150,height=300)
    with row_0_3:
        if len(onclick) > 0:
            idx = onclick[-1]['pointIndex']
        else:
            idx = 5
        st.markdown(
            f"""<div style='text-align: left; color: {interpolate_color(color1, color2, round(1- idx/(len(files)-1),2))}; font-size: 30px; padding-right: 40px; padding-top: 20px;'>λ={round(1-idx/(len(files)-1),2)}</div>""",
            unsafe_allow_html=True
        )
        components.html(files[idx], width=150, height=300)
    with row_0_2:
        st.markdown(f"""<div style='text-align: left; color: {color2}; font-size: 30px; padding-right: 40px; padding-top: 20px;'>λ=0.0</div>""",unsafe_allow_html=True)
        components.html(files[0],width=150,height=300)


if __name__ == "__main__":
    img = Image.open("streamlit_app/assets/images/icon.png")
    st.set_page_config(page_title="Rewarded soups",page_icon=img,layout="wide")
    inject_custom_css("streamlit_app/assets/styles.css")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    run()
