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
import colorsys

plt.style.use('default')
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'


def interpolate_color(color1, color2, factor):
    """Interpolates between two RGB colors. Factor is between 0 and 1."""
    color1 = colorsys.rgb_to_hls(
        int(color1[1:3], 16) / 255.0,
        int(color1[3:5], 16) / 255.0,
        int(color1[5:], 16) / 255.0
    )
    color2 = colorsys.rgb_to_hls(
        int(color2[1:3], 16) / 255.0,
        int(color2[3:5], 16) / 255.0,
        int(color2[5:], 16) / 255.0
    )
    new_color = [color1[i] * (1 - factor) + color2[i] * factor for i in range(3)]
    new_color = colorsys.hls_to_rgb(*new_color)
    return '#{:02x}{:02x}{:02x}'.format(
        int(new_color[0] * 255), int(new_color[1] * 255), int(new_color[2] * 255)
    )


color1 = "#fa7659"
color2 = "#6dafd7"

shapes = [
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

shapes = [
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


def plot_pareto(dict_results: tp.Dict):

    reward1_key = "R1"
    reward2_key = "R2"

    # Series for "wa"
    dict_results["wa_d"] = [x for i, x in enumerate(dict_results["wa_d"]) if i % 2 == 0]
    lambda_values_wa = [
        round(i / (len(dict_results["wa_d"]) - 1), 2) for i in range(len(dict_results["wa_d"]))
    ][::-1]
    reward1_values_wa = [item[reward1_key] for item in dict_results["wa_d"]]
    reward2_values_wa = [item[reward2_key] for item in dict_results["wa_d"]]

    # Series for "morl"
    # Series for "init"
    reward1_values_morl = [dict_results["morl"][reward1_key]]
    reward2_values_morl = [dict_results["morl"][reward2_key]]

    # Series for "init"
    reward1_values_init = [dict_results["init"][reward1_key]]
    reward2_values_init = [dict_results["init"][reward2_key]]

    layout = go.Layout(autosize=False, width=1000, height=1000)
    fig = go.Figure(layout=layout)

    for i in range(len(reward1_values_wa) - 1):
        fig.add_trace(
            go.Scatter(
                x=reward1_values_wa[i:i + 2],
                y=reward2_values_wa[i:i + 2],
                mode='lines',
                hoverinfo='skip',
                line=dict(
                    color=interpolate_color(color1, color2, i / (len(reward1_values_wa) - 1)),
                    width=2
                ),
                showlegend=False
            )
        )

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
            x=reward1_values_morl,
            y=reward2_values_morl,
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
            #range = [5.21,5.31],
            #nticks=6,
            showticklabels=True,
            ticks='outside',
            tickfont=dict(size=18,),
            title=dict(text="R1", font=dict(size=18), standoff=10),
            showgrid=False,
            zeroline=False,
            hoverformat='.2f'
        ),
        yaxis=dict(
            #range = [0.78,0.825],
            #nticks=7,
            showticklabels=True,
            ticks='outside',
            tickfont=dict(size=18,),
            title=dict(text="R2", font=dict(size=18), standoff=10),
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
        <h3 style='text-align: left;';>RLHF of LLaMA for diverse news summarization</h3>""",
        unsafe_allow_html=True
    )

    st.markdown(
        r"""
Given the importance of RLHF to train LLMs, we begin our experiments with text-to-text generation.
Our pre-trained network is LLaMA-7b, instruction fine-tuned on Alpaca.
For RL training with PPO, we employ the trl package and the setup from with low-rank adapters (LoRA) for efficiency.
Here we consider summarization on Reuter news.
To evaluate the summary in the absence of supervision, we utilized two different reward models, available on HuggingFace: [$R_1$](https://huggingface.co/Tristan/gpt2_reward_summarization) follows the Summarize from Human Feedback paper while [$R_2$](https://huggingface.co/CogComp/bart-faithful-summary-detector) leverages contrast candidate generation.

Our results below reveal the following insights. The front defined by rewarded soups between the two weights specialized on $R_1$ (i.e., $\lambda=0.0$) and $R_2$ (i.e., $\lambda=1.0$) is above the straight line connecting those two points; this validates what we call in the paper *the linear mode connectivity hypothesis*. Moreover, the front intersects the point obtained by multi-objective RL (MORL) fine-tuning on $(1-\mu) \times R_1 + \mu \times R_2$ for $\mu=0.5$ (i.e., the average of the two rewards). Interestingly, when we compare both full fronts in the paper, they exhibit qualitatively the same shape. The qualitative visual inspections of the generations show that increasing $\lambda$ leads to shorter but more factual summaries; this is because $R_2$ evaluates faithfulness in priority.""",
        unsafe_allow_html=True
    )
    st.markdown(
        """<h3 style='text-align: center;';>Click on a rewarded soup point on the left and select a subject on the right!</h3>""",
        unsafe_allow_html=True
    )

    files = []

    with open("streamlit_app/data/textgen/data.pkl", "rb") as f:
        data = pickle.load(f)
    with open("streamlit_app/data/textgen/data_prompt.pkl", "rb") as f:
        data_prompt = pickle.load(f)
    with open("streamlit_app/data/textgen/data_title.pkl", "rb") as f:
        data_title = pickle.load(f)

    left, right = st.columns((2, 2))
    with left:
        fig = plot_pareto(data)
        onclick = plotly_events(fig, click_event=True)
    with right:
        option = st.selectbox('', data_title.keys())

        subject = data_title[option]
        st.markdown(
            f"""
        <div class="promptTextbox">
            <div class="promptHeader">
                Text to summarize:
            </div>
            <div class="promptContent">
                {data_prompt[subject]['query']}
            </div>
        </div>
        """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        summary1 = data_prompt[subject]['outs'][0]["out"]
        summary3 = data_prompt[subject]['outs'][-1]["out"]
        nb_summaries = len(data_prompt[subject]['outs'])
        if len(onclick) > 0:
            idx = onclick[0]["pointIndex"]
        else:
            idx = 5
        lambda2 = round(1 - idx / (len(data["wa_d"]) - 1), 2)
        summary2 = data_prompt[subject]['outs'][idx]["out"]
        bgcolor = interpolate_color(color2, color1, lambda2)

        st.markdown(
            f"""
        <div class="promptTextbox">
            <div class="promptHeader">
                Generated summaries:
            </div>
            <div class="promptContent">
                <div class="lambda-header" style='background-color: {color2};'>λ=0.0</div><div class="lambdas" style='background-color: {color2};'>{summary3}</div>
                <div class="lambda-header" style='background-color: {bgcolor};'>λ={lambda2} </div><div class="lambdas" style='background-color: {bgcolor};'>{summary2}</div><br>
                <div class="lambda-header" style='background-color: {color1};'>λ=1.0</div><div class="lambdas" style='background-color: {color1};'>{summary1}</div><br>
            </div>
        </div>
        """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    img = Image.open("streamlit_app/assets/images/icon.png")
    st.set_page_config(page_title="Rewarded soups", page_icon=img, layout="wide")
    inject_custom_css("streamlit_app/assets/styles.css")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    run()
