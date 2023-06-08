import os
import argparse
import re
from datetime import datetime

from trl import set_seed

# WANDB__SERVICE_WAIT = 300
LOCAL_FILES_ONLY = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"
FOLDER_EXPE = os.path.join(os.environ.get("DATA", "/"), "experimentsllama")
LOAD_IN_8BIT = True


class DefaultArgs:
    seed = 0
    output_max_length = 32
    base_model_name = "decapoda-research/llama-7b-hf"
    peft_name = "tloen/alpaca-lora-7b"

    reward_models_summary = [
        "Tristan/gpt2_reward_summarization",
        "CogComp/bart-faithful-summary-detector"  # format {'0': 'HALLUCINATED', '1': 'FAITHFUL'}, thus requiring --reward_formats '1-0'
    ]

    reward_models_stack = [
        "OpenAssistant/reward-model-deberta-v3-base",
        "OpenAssistant/reward-model-electra-large-discriminator",
    ]

    reward_models_review = [
        "OpenAssistant/reward-model-deberta-v3-base",
        "OpenAssistant/reward-model-electra-large-discriminator",
    ]

    reward_models_assistant = [
        "OpenAssistant/reward-model-deberta-v3-large-v2",
        "OpenAssistant/reward-model-electra-large-discriminator",
        "theblackcat102/reward-model-deberta-v3-base-v2",
        "OpenAssistant/reward-model-deberta-v3-base",
    ]


class Naming:
    @staticmethod
    def str_dict(dict_args):
        """
        function that print args dictionaries in a beautiful way
        """
        str_out = "\n" + "#" * 40
        col_width = max(len(str(word)) for word in dict_args) + 2
        for arg in sorted(list(dict_args.keys())):
            if arg.startswith("__"):
                continue
            else:
                str_print = str(dict_args[arg])
                str_out += "\n" + "".join([str(arg).ljust(col_width), str_print])
        str_out += "\n" + "#" * 40 + "\n"
        return str_out

    @staticmethod
    def get_name(script_args):
        name = script_args.base_model_name.split("/")[-1] + "-ppo-" + script_args.task

        if script_args.dataset_name != "default":
            dataset_name = script_args.dataset_name.split("/")[-1]
            name += f"-d{dataset_name}"

        for reward_model in script_args.reward_models:
            short_sent = Naming.get_name_model(reward_model)
            name += f"-{short_sent}"

        reward_format = "_".join(
            [reward_format.replace("x", "") for reward_format in script_args.reward_formats]
        )
        name += f"-r{reward_format}"

        if script_args.learning_rate != DefaultArgsPPO.learning_rate:
            name += f"-lr{script_args.learning_rate}"

        if script_args.init_kl_coef != DefaultArgsPPO.init_kl_coef:
            init_kl_coef = str(script_args.init_kl_coef).split(".")[-1]
            name += f"-kl{init_kl_coef}"

        if script_args.peft_name not in [None, "None", "none", DefaultArgs.peft_name]:
            short_peft_name = script_args.peft_name.split("/")[-1]
            name += f"-p{short_peft_name}"

        name += "-" + datetime.now().strftime("%m-%d-%s")
        return name[:92]

    @staticmethod
    def get_name_model(name):
        list_reward_suffix = re.split("-|_", name.split('/')[-1])
        list_reward_suffix = [t for t in list_reward_suffix if t]
        short_sent = "".join([t[0] for t in list_reward_suffix])
        return short_sent




class DefaultArgsPPO:
    seed = DefaultArgs.seed
    base_model_name = DefaultArgs.base_model_name
    peft_name = DefaultArgs.peft_name
    dataset_name = "default"

    output_min_length = DefaultArgs.output_max_length//2
    output_max_length = DefaultArgs.output_max_length
    log_with = "wandb"
    mini_batch_size = 4
    batch_size = 128
    learning_rate = 1.41e-5
    init_kl_coef = 0.2
    num_epochs = 1
    gradient_accumulation_steps = 1


def get_args_ppo(default_args):
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--task', type=str)
    parser.add_argument('--reward_models', type=str, nargs='+')
    parser.add_argument('--reward_formats', type=str, nargs='+', default=None)
    # reward_formats handles the potentially various formats of the (potentially) multiple reward models, including their weightings in morl

    parser.add_argument('--seed', type=int, default=default_args.seed)
    parser.add_argument('--dataset_name', type=str, default=default_args.dataset_name)
    parser.add_argument('--base_model_name', type=str, default=default_args.base_model_name)
    parser.add_argument('--peft_name', type=str, default=default_args.peft_name)

    parser.add_argument('--output_min_length', type=int, default=default_args.output_min_length)
    parser.add_argument('--output_max_length', type=int, default=default_args.output_max_length)
    parser.add_argument('--log_with', type=str, default=default_args.log_with)
    parser.add_argument('--output_folder', type=str, default="auto")

    parser.add_argument(
        '--mini_batch_size',
        type=int,
        default=default_args.mini_batch_size,
    )
    parser.add_argument('--batch_size', type=int, default=default_args.batch_size)
    parser.add_argument('--learning_rate', type=float, default=default_args.learning_rate)
    parser.add_argument('--init_kl_coef', type=float, default=default_args.init_kl_coef)
    parser.add_argument('--num_epochs', type=int, default=default_args.num_epochs)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=default_args.gradient_accumulation_steps)

    args = parser.parse_args()
    args.myhost = os.uname()[1]
    if args.log_with == "None":
        args.log_with = None
    if args.reward_formats is None:
        args.reward_formats = ["0" for _ in args.reward_models]
    print(len(args.reward_formats), len(args.reward_models))
    assert len(args.reward_formats) == len(args.reward_models)

    return args


class DefaultArgsInference:
    seed = DefaultArgs.seed
    base_model_name = DefaultArgs.base_model_name
    dataset_name = "default"
    peft_names = [DefaultArgs.peft_name]
    num_samples = 200
    num_lambdas = 11
    verbose = 0
    output_max_length = DefaultArgs.output_max_length
    task = None

def get_args_inference(default_args):
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--task', type=str)
    parser.add_argument('--base_model_name', type=str, default=default_args.base_model_name)
    parser.add_argument('--dataset_name', type=str, default=default_args.dataset_name)
    parser.add_argument('--peft_names', type=str, nargs='+', default=default_args.peft_names)
    parser.add_argument('--num_samples', type=int, default=default_args.num_samples)
    parser.add_argument('--num_lambdas', type=int, default=default_args.num_lambdas)
    parser.add_argument('--verbose', type=int, default=default_args.verbose)
    parser.add_argument('--seed', type=int, default=default_args.seed)
    parser.add_argument('--output_max_length', type=int, default=default_args.output_max_length)

    args = parser.parse_args()
    args.myhost = os.uname()[1]
    return args
