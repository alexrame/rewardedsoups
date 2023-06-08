import streamlit as st
from utils import inject_custom_css
from PIL import Image
from matplotlib import rcParams

color_title = "#000000"
color_text = "#000000"

text = f"""
<p style="text-align: center; font-size: 40px; font-weight: bold;">🍲 Rewarded soups 🍲</p>
<p style="text-align: center;">Welcome to our interactive streamlit app showcasing the key concepts and experiments presented in our paper <br><a href="http://arxiv.org/abs/2306.04488" target="_blank" style="color: #007BFF;">Rewarded soups: towards Pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards</a></p>
<h3 > Abstract </h3>
<p style="text-align: justify;"> Foundation models are first pre-trained on vast unsupervised datasets and then fine-tuned on labeled data. Reinforcement learning, notably from human feedback (RLHF), can further align the network with the intended usage. Yet the imperfections in the proxy reward may hinder the training and lead to suboptimal results; the diversity of objectives in real-world tasks and human opinions exacerbate the issue. This paper proposes embracing the heterogeneity of diverse rewards by following a multi-policy strategy. Rather than focusing on a single a priori reward, we aim for Pareto-optimal generalization across the entire space of preferences. To this end, we propose rewarded soup, first specializing multiple networks independently (one for each proxy reward) and then interpolating their weights linearly. This succeeds empirically because we show that the weights remain linearly connected when fine-tuned on diverse rewards from a shared pre-trained initialization. We demonstrate the effectiveness of our approach for text-to-text (summarization, Q&A, helpful assistant, review), text-image (image captioning, text-to-image generation, visual grounding, VQA), and control (locomotion) tasks. We hope to enhance the alignment of deep models, and how they interact with the world in all its diversity.</p>

<h3 >What will I find here ?</h3>

<p style="text-align: justify;">In this app, you will find interactive figures and qualitative examples demonstratating the effectiveness of our approach. Specifically, we detail the following tasks: RLHF of LLaMA for news summarization, RLHF of a diffusion model for text-to-image generation, and the locomotion task. To help the reproduction of these results, we also provide <a href="https://github.com/alexrame/rewardedsoups" target="_blank" style="color: #007BFF;">our code here</a>.</p>
"""


def run_UI():
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Tahoma']
    inject_custom_css("streamlit_app/assets/styles.css")

    st.markdown(
        f"""
        <link href='http://fonts.googleapis.com/css?family=Roboto' rel='stylesheet' type='text/css'>
        {text}
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    img = Image.open("streamlit_app/assets/images/icon.png")
    st.set_page_config(
        page_title="Rewarded soups",
        page_icon=img,
        layout="wide",
    )
    st.set_option("deprecation.showPyplotGlobalUse", False)
    run_UI()
