import torch
import os

import numpy as np
from utils import llama_utils, inference_utils, args_utils

device = 0 if torch.cuda.is_available() else "cpu"

from tasks import assistant, summary, review, stack

if __name__ == "__main__":
    default_args = args_utils.DefaultArgsInference()
    script_args = args_utils.get_args_inference(default_args=default_args)
    args_utils.set_seed(script_args.seed)
    print(args_utils.Naming.str_dict(script_args.__dict__))

    # 1. load dataset and tokenizers
    tokenizer = llama_utils.Tokenizer.load_tokenizer(script_args.base_model_name)

    if script_args.task == "summary":
        reward_models = args_utils.DefaultArgs.reward_models_summary
        predictor_class = summary.PredictorSummary
        samples_class = summary.Samples
    elif script_args.task == "stack":
        reward_models = args_utils.DefaultArgs.reward_models_stack
        predictor_class = stack.PredictorStack
        samples_class = stack.Samples
    elif script_args.task == "review":
        reward_models = args_utils.DefaultArgs.reward_models_review
        predictor_class = review.PredictorReview
        samples_class = review.Samples
    elif script_args.task == "assistant":
        reward_models = args_utils.DefaultArgs.reward_models_assistant
        predictor_class = assistant.PredictorAssistant
        samples_class = assistant.Samples
    else:
        raise ValueError(f"Unknown task {script_args.task}")

    if script_args.dataset_name == "samples":
        # fake samples for debug
        query_tensors = samples_class.get_fake_samples(bs=script_args.num_samples, tokenizer=tokenizer)
    else:
        query_tensors = samples_class.get_samples(
            dataset_name=script_args.dataset_name, bs=script_args.num_samples, tokenizer=tokenizer
        )

    print("First query:", query_tensors[0])
    print("First decoded query:", tokenizer.decode(query_tensors[0]))

    # 2. load models
    base_model = inference_utils.Loader.load_base_model(script_args.base_model_name)
    reward_pipes = llama_utils.Pipelines.load_pipes(reward_models, device=device)

    # 3. inference for wa
    predictor = predictor_class(
        reward_pipes=reward_pipes,
        tokenizer=tokenizer,
        output_max_length=script_args.output_max_length,
        device=device,
    )
    resultscomputer = inference_utils.ResultsComputer(
        predictor=predictor,
        base_model=base_model,
        query_tensors=query_tensors,
        verbose=script_args.verbose
    )
    inference_utils.get_results_rewards(
        resultscomputer, peft_names=script_args.peft_names, num_lambdas=script_args.num_lambdas
    )
