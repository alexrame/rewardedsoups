# Text-to-text experiments: fine-tuning LLaMA with diverse RLHFs

Given the significance of RLHF to train LLMs, we begin our experiments with text-to-text generation tasks. We fine-tune LLaMA-7b (instruction fine-tuned on Alpaca), with PPO, employing the [trl](https://github.com/lvwerra/trl/) package and the setup from [trl-peft](https://huggingface.co/blog/trl-peft) with low-rank adapters (LoRA) for efficiency.

We consider the following tasks: [summary](tasks/summary.py) on two datasets (news and reddit), answering [stack](tasks/stack.py) exchange questions, movie [review](tasks/review.py) generation, and helpfulness as a conversational [assistant](tasks/assistant.py). To evaluate the generation in the absence of supervision, we utilized $N=2$ different reward models for each task, except for the assistant task where $N=4$. These RMs were trained on human preferences datasets and all open-sourced on HuggingFace. For example in summarization, $R_1$ follows the "Summarize from Human Feedback" paper while $R_2$ leverages "contrast candidate generation". For other tasks, we rely on diverse RMs from OpenAssistant; though they all assess if the answer is adequate, they differ by their architectures and procedures. All reward models can be found [here](utils/args_utils.py).

## Setup

### Requiremeents

The needed requirements are [here](llama/requirements.txt).

## Rewarded soups

Considering a `task` in `["summary", "stack", "review", "assistant"]`, the strategy is in two steps:

### Reward optimizations

First, call twice the [PPO](train_ppo.py) with two different reward models,
All required args are defined [here](utils/args_utils.py).

```
python3 train_ppo.py --task summary --dataset_name news --reward_models Tristan/gpt2_reward_summarization --output_folder ${folder_r1}
python3 train_ppo.py --task summary --dataset_name news --reward_models CogComp/bart-faithful-summary-detector --dataset_name news --reward_formats '1-0' --output_folder ${folder_r2}
```


### Inference of interpolated weights

Second, interpolate the fine-tuned weights at [inference](inference_rewardedsoups.py).

```
python3 inference_rewardedsoups.py --task summary --dataset_name news --peft_names ${folder_r1} ${folder_r2}
```

This should provide a Pareto-front of solutions. We obtain the following result:

```
results[ 0.0 ] = [{'LABEL_0': 0.9837322938069701, 'n': 'grs'}, {'HALLUCINATED': 0.34354008454363794, 'FAITHFUL': -0.4440372304804623, 'n': 'bfsd'}]
results[ 0.1 ] = [{'LABEL_0': 1.108713237568736, 'n': 'grs'}, {'HALLUCINATED': 0.34228092256031234, 'FAITHFUL': -0.44228213937953115, 'n': 'bfsd'}]
results[ 0.2 ] = [{'LABEL_0': 1.1973440561071038, 'n': 'grs'}, {'HALLUCINATED': 0.3337066343799233, 'FAITHFUL': -0.44304117301478985, 'n': 'bfsd'}]
results[ 0.3 ] = [{'LABEL_0': 1.285310146510601, 'n': 'grs'}, {'HALLUCINATED': 0.41363132309168577, 'FAITHFUL': -0.44826980523765086, 'n': 'bfsd'}]
results[ 0.4 ] = [{'LABEL_0': 1.3942182363569737, 'n': 'grs'}, {'HALLUCINATED': 0.6186571175232529, 'FAITHFUL': -0.4634812906384468, 'n': 'bfsd'}]
results[ 0.5 ] = [{'LABEL_0': 1.4592856185883283, 'n': 'grs'}, {'HALLUCINATED': 0.617289823628962, 'FAITHFUL': -0.4630217505991459, 'n': 'bfsd'}]
results[ 0.6 ] = [{'LABEL_0': 1.5127975235134363, 'n': 'grs'}, {'HALLUCINATED': 0.6535065068677067, 'FAITHFUL': -0.4648162759467959, 'n': 'bfsd'}]
results[ 0.7 ] = [{'LABEL_0': 1.5688072920590639, 'n': 'grs'}, {'HALLUCINATED': 0.7415622024610639, 'FAITHFUL': -0.4705177211761475, 'n': 'bfsd'}]
results[ 0.8 ] = [{'LABEL_0': 1.653215588144958, 'n': 'grs'}, {'HALLUCINATED': 0.8457040116935969, 'FAITHFUL': -0.47928548775613306, 'n': 'bfsd'}]
results[ 0.9 ] = [{'LABEL_0': 1.728214347474277, 'n': 'grs'}, {'HALLUCINATED': 0.8377302279276774, 'FAITHFUL': -0.47746868235990403, 'n': 'bfsd'}]
results[ 1.0 ] = [{'LABEL_0': 1.7814147435873746, 'n': 'grs'}, {'HALLUCINATED': 1.0361331396130844, 'FAITHFUL': -0.4901799873448908, 'n': 'bfsd'}]
```


### Multi objective RL

To reproduce the MORL baseline, simply provide the different reward models, and then update the reward formats argument. For example, `0x0.4 '1-0'x0.6`, then the reward is $0.4 \times R_1 + 0.6 \times R_2$, where in $R_2$ we optimize the difference between the score given to the second class wrt the first.

```
python3 train_ppo.py --task summary --dataset_name news --reward_models Tristan/gpt2_reward_summarization CogComp/bart-faithful-summary --dataset_name news-detector --reward_formats '0x0.4' '1-0x0.6' --output_folder ${folder_morl}
```
