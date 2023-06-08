import os
import numpy as np
import torch

from trl import PPOConfig, PPOTrainer

from tasks import assistant, summary, review, stack

from utils import llama_utils, ppo_utils, args_utils

if __name__ == "__main__":
    default_args = args_utils.DefaultArgsPPO()
    script_args = args_utils.get_args_ppo(default_args=default_args)

    os.environ["WANDB_DIR"] = os.path.join(args_utils.FOLDER_EXPE, "wandb")
    script_args.script_args_name = args_utils.Naming.get_name(script_args)
    os.environ["WANDB_NAME"] = script_args.script_args_name
    print(args_utils.Naming.str_dict(script_args.__dict__))

    base_model = ppo_utils.Loader.load_base_model(script_args.base_model_name)
    model = ppo_utils.Loader.load_peft_model(base_model, peft_name=script_args.peft_name)

    config = PPOConfig(
        model_name=script_args.base_model_name,
        init_kl_coef=script_args.init_kl_coef,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        batch_size=script_args.batch_size,
        mini_batch_size=script_args.mini_batch_size,
        optimize_cuda_cache=True,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        seed=script_args.seed
    )
    # set seed before initializing value head for deterministic eval
    args_utils.set_seed(script_args.seed)

    tokenizer = llama_utils.Tokenizer.load_tokenizer(script_args.base_model_name)
    ppo_utils.Loader.print_trainable_parameters(model)

    # We retrieve the dataloader by calling the `build_dataset` function.

    if script_args.task == "summary":
        dataset_method = summary.build_dataset
        transform_text_method = summary.transform_text_summary
    elif script_args.task == "stack":
        dataset_method = stack.build_dataset
        transform_text_method = stack.transform_text_stack
    elif script_args.task == "review":
        dataset_method = review.build_dataset
        transform_text_method = review.transform_text_review
    elif script_args.task == "assistant":
        dataset_method = assistant.build_dataset
        transform_text_method = assistant.transform_text_assistant
    else:
        raise ValueError("Unknown task", script_args.task)

    dataset = dataset_method(
        dataset_name=script_args.dataset_name, tokenizer=tokenizer, split="train"
    )
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate
    )
    ppo_utils.Loader.load_optimizer(
        optimizer,
        peft_name=script_args.peft_name,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    ref_model = ppo_utils.Loader.load_ref_model(script_args)
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer
    )

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"

    runner = ppo_utils.Runner(
        ppo_trainer=ppo_trainer,
        tokenizer=tokenizer,
        device=device,
        script_args=script_args,
        do_review=(script_args.task == "review"),
        transform_text_method=transform_text_method,
    )

    if script_args.task == "summary":
        for reward_pipe in runner.reward_pipes:
            reward_pipe.tokenizer.pad_token_id = model.config.eos_token_id

    runner.train_ppo(model, num_epochs=script_args.num_epochs)
