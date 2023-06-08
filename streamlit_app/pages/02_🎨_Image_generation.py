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

def interpolate_color(color1, color2, factor):
    """Interpolates between two RGB colors. Factor is between 0 and 1."""
    color1 = colorsys.rgb_to_hls(int(color1[1:3], 16)/255.0, int(color1[3:5], 16)/255.0, int(color1[5:], 16)/255.0)
    color2 = colorsys.rgb_to_hls(int(color2[1:3], 16)/255.0, int(color2[3:5], 16)/255.0, int(color2[5:], 16)/255.0)
    new_color = [color1[i] * (1 - factor) + color2[i] * factor for i in range(3)]
    new_color = colorsys.hls_to_rgb(*new_color)
    return '#{:02x}{:02x}{:02x}'.format(int(new_color[0]*255), int(new_color[1]*255), int(new_color[2]*255))


color1 = "#fa7659"
color2 = "#6dafd7"

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

def plot_pareto(dict_results: tp.Dict):

    reward1_key = "ava"
    reward2_key = "cafe"

    # Series for "wa"
    lambda_values_wa = [round(1 - i/(len(dict_results["wa_d"])-1),2) for i in range(len(dict_results["wa_d"]))]
    reward1_values_wa = [item[reward1_key] for item in dict_results["wa_d"]]
    reward2_values_wa = [item[reward2_key] for item in dict_results["wa_d"]]

    # Series for "morl"
    mu_values_morl = [round(1 - i/(len(dict_results["morl_d"])-1),2) for i in range(len(dict_results["morl_d"]))]
    reward1_values_morl = [item[reward1_key] for item in dict_results["morl_d"]][3]
    reward2_values_morl = [item[reward2_key] for item in dict_results["morl_d"]][3]

    # Series for "init"
    reward1_values_init = [dict_results["init"][reward1_key]]
    reward2_values_init = [dict_results["init"][reward2_key]]


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
            x=[reward1_values_morl],
            y=[reward2_values_morl],
            mode='markers',
            name='MORL: μ=0.5',
            hoverinfo='skip',
            marker=dict(color='#A45EE9', size=15, symbol="star")
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
            showticklabels=True,
            ticks='outside',
            tickfont=dict(size=18,),
            title=dict(text="Ava reward", font=dict(size=18), standoff=10),
            showgrid=False,
            zeroline=False,
            hoverformat='.2f'
        ),
        yaxis=dict(
            showticklabels=True,
            ticks='outside',
            tickfont=dict(size=18,),
            title=dict(text="Cafe reward", font=dict(size=18), standoff=10),
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
        <h3 style='text-align: left;';>RLHF of diffusion model for diverse human aesthetics</h3>""",unsafe_allow_html=True)

    st.markdown(
        r"""
Beyond text generation, we now apply RS to align text-to-image generation with human feedbacks.
Here, we demonstrate how rewarded soups allows to interpolate between models fine-tuned for different aesthetic metrics.
Our network is a diffusion model with 2.2B parameters, pre-trained on an internal dataset of 300M images; it reaches similar quality as Stable Diffusion, which was not used for copyright reasons.
To represent the subjectivity of human aesthetics, we employ $N=2$ open-source reward models: [*ava*](https://github.com/christophschuhmann/improved-aesthetic-predictor/), trained on the AVA dataset, and [*cafe*](https://huggingface.co/cafeai/cafe_aesthetic), trained on a mix of real-life and manga images.
We first generate 10000 images; then, for each reward, we remove half of the images with the lowest reward's score and fine-tune 10\% of the parameters on the reward-weighted negative log-likelihood.

Our results below show that interpolating between the expert models unveils a Pareto-optimal front, enabling alignment with a variety of aesthetic preferences.
Specifically, all interpolated models produce images of similar quality compared to fine-tuned models, demonstrating linear mode connectivity between the two fine-tuned models.
This ability to adapt at test time paves the way for a new form of user interaction with text-to-image models, beyond prompt engineering.
""",
        unsafe_allow_html=True
    )
    st.markdown("""<h3 style='text-align: left;';>Click on a rewarded soup point on the left and select a prompt on the right!</h3>""",unsafe_allow_html=True)

    files = []

    with open("streamlit_app/data/imgen/data.pkl","rb") as f:
        data = pickle.load(f)
    with open("streamlit_app/data/imgen/data_images.pkl","rb") as f:
        data_images = pickle.load(f)

    row_0_1,row_0_2  = st.columns([2,3])
    with row_0_1:
        fig = plot_pareto(data)
        onclick = plotly_events(fig, click_event=True)
    with row_0_2:
        option = st.selectbox('',data_images.keys())
        for i in range(11):
            filename = f'https://github.com/continual-subspace/hidden_soup/blob/main/{data_images[option]["filename"]}_{i}.png?raw=true'
            files.append(filename)
        row_1_1,row_1_2,row_1_3  = st.columns([1,1,1])
        if len(onclick) > 0:
            idx = onclick[-1]['pointIndex']
        else:
            idx = 5
        img = files[idx]
        bgcolor = interpolate_color(color2,color1,round(1 - idx/(len(files)-1),2))
        lambda2 = round(1 - idx/(len(files)-1),2)

        img1 = files[0]
        img0 = files[-1]

        st.markdown(
            f"""
            <div class="promptTextbox">
                <div class="promptHeader">
                    Generated images:
                </div>
                <div class="imgContent">
                    <div class="imgContainer">
                        <div class="imglambda-header" >λ=0.0</div>
                        <div class="imglambdas" style='background-color: {color2};'><img src='{img0}' alt='{img0}'></div>
                    </div>
                    <div class="imgContainer">
                        <div class="imglambda-header" >λ={lambda2}</div>
                        <div class="imglambdas" style='background-color: {bgcolor};'><img src='{img}' alt='{img}'></div>
                    </div>
                    <div class="imgContainer">
                        <div class="imglambda-header" >λ=1.0</div>
                        <div class="imglambdas" style='background-color: {color1};'><img src='{img1}' alt='{img1}'></div>
                    </div>
                </div>
            </div>
        """,
            unsafe_allow_html=True
        )



if __name__ == "__main__":
    img = Image.open("streamlit_app/assets/images/icon.png")
    st.set_page_config(page_title="Rewarded soups",page_icon=img,layout="wide")
    inject_custom_css("streamlit_app/assets/styles.css")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    run()
