import os
from pathlib import Path
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm

from transformers import LlamaForCausalLM
from trl import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_int8_training

from utils import llama_utils, args_utils


def get_scheduler(optimizer, warmup_steps):
    if warmup_steps == 0:
        return None
    def warmup_schedule(step,):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    return LambdaLR(optimizer, warmup_schedule)


class Loader:

    @staticmethod
    def load_base_model(base_model_name):
        base_model = LlamaForCausalLM.from_pretrained(base_model_name, load_in_8bit=args_utils.LOAD_IN_8BIT, device_map="auto")
        base_model = prepare_model_for_int8_training(base_model)
        return base_model

    @staticmethod
    def load_peft_model(base_model, peft_name):
        """### Apply LoRA
        Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.
        """
        if peft_name.startswith("lora"):
            if "-" in peft_name:
                r = int(peft_name.split("-")[-1])
            else:
                r = 16
            lora_config = LoraConfig(
                r=r,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(base_model, lora_config)
        else:
            model = PeftModel.from_pretrained(
                base_model, peft_name, local_files_only=args_utils.LOCAL_FILES_ONLY)
        modelvaluehead = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        return modelvaluehead

    def load_optimizer(optimizer, peft_name, device):
        if not (peft_name.startswith("/") and os.path.exists(peft_name)):
            return
        optimizer_path = os.path.join(peft_name, "optimizer.pth")
        if not os.path.exists(optimizer_path):
            return
        print("Load optimizer from", optimizer_path)
        optimizer_state_dict = torch.load(optimizer_path, map_location=device)
        optimizer.load_state_dict(optimizer_state_dict)

    @staticmethod
    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    @staticmethod
    def load_ref_model(script_args):
        # Copy reference model does not behave like creating "from scratch" the same reference model
        ref_base_model = Loader.load_base_model(script_args.base_model_name)
        ref_model = Loader.load_peft_model(
            ref_base_model, peft_name=args_utils.DefaultArgs.peft_name
        )
        return ref_model


class Runner():

    def __init__(self, ppo_trainer, tokenizer, device, script_args, transform_text_method, do_review=False):
        self.ppo_trainer = ppo_trainer
        self.tokenizer = tokenizer
        self.do_review = do_review
        self.generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": -1,
        }
        self.transform_text_method = transform_text_method

        self.script_args = script_args
        self.output_length_sampler = LengthSampler(script_args.output_min_length, script_args.output_max_length)

        self.reward_pipes = llama_utils.Pipelines.load_pipes(reward_models=script_args.reward_models, device=device)

        self.sent_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": script_args.mini_batch_size
        }

    def apply_reward(self, batch):

        sum_coeff = 0
        list_rewards = []
        for i in range(len(self.reward_pipes)):
            reward_pipe = self.reward_pipes[i]
            reward_format = self.script_args.reward_formats[i].split("x")[0]
            rewards = self._apply_reward_per_pipe(
                reward_pipe, batch, reward_format=reward_format
            )
            if "x" in self.script_args.reward_formats[i]:
                coeff = float(self.script_args.reward_formats[i].split("x")[1])
                rewards = [coeff * reward for reward in rewards]
            else:
                coeff = 1
            sum_coeff += coeff
            list_rewards.append(rewards)

        return np.sum(np.stack(list_rewards), 0) / sum_coeff

    def _apply_reward_per_pipe(self, reward_pipe, batch, reward_format):
        if self.do_review:
            texts = [
                self.transform_text_method(
                    reward_pipe=reward_pipe,
                    response=llama_utils.Instructions.get_response(query) + response)
                for query, response in zip(batch["query"], batch["response"])
            ]
        else:
            texts = [
                self.transform_text_method(
                    reward_pipe=reward_pipe,
                    post=llama_utils.Instructions.get_input(query),
                    response=llama_utils.Instructions.get_response(query) + response
                ) for query, response in zip(batch["query"], batch["response"])
            ]
        pipe_outputs = reward_pipe(texts, **self.sent_kwargs)

        def get_score_from_output(output, reward_format):
            if reward_format == "":
                return 0.
            if "-" in reward_format:
                # special case, when optimizing the difference between two classes
                return (get_score_from_output(
                    output, reward_format.split("-")[0]) -
                        get_score_from_output(output, reward_format.split("-")[1]))
            return output[int(reward_format)]["score"]

        rewards = [get_score_from_output(output, reward_format) for output in pipe_outputs]
        return rewards

    def train_ppo(self, model, num_steps=None, num_epochs=1):
        step = 0
        for epoch in range(num_epochs):
            print(f"Begin epoch: {epoch}")
            for _, batch in tqdm(enumerate(self.ppo_trainer.dataloader)):
                self.step_ppo(model=model, batch=batch)
                if step % 10 == 1 :
                    self.push(model, step=step, optimizer=self.ppo_trainer.optimizer)
                step += 1
                if num_steps is not None and step > num_steps:
                    break
        self.push(model, step=None)

    def step_ppo(self, model, batch):
        query_tensors = batch["input_ids"]

        model.gradient_checkpointing_disable()
        model.pretrained_model.config.use_cache = True

        # Get response from Causal LM
        response_tensors = []
        for query in query_tensors:
            gen_len = self.output_length_sampler()
            self.generation_kwargs["max_new_tokens"] = gen_len
            response = self.ppo_trainer.generate(query, **self.generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [self.tokenizer.decode(r.squeeze()) for r in response_tensors]

        # Compute reward score
        rewards = self.apply_reward(batch)
        rewards = [torch.tensor(reward) for reward in rewards]

        # Run PPO step
        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False

        stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
        try:
            self.ppo_trainer.log_stats(stats, batch, rewards)
        except Exception as exc:
            print("error at step ppo")
            print(exc)

    def push(self, model, optimizer=None, step=None):
        str_args = args_utils.Naming.str_dict(self.script_args.__dict__)
        if step is not None:
            str_args += "-e" + str(step)

        import wandb
        if self.script_args.log_with == "wandb":
            wandb_dir = wandb.run.dir
            if not args_utils.LOCAL_FILES_ONLY:
                wandb_url = wandb.run.get_url()
                assert wandb_url is not None
            else:
                wandb_url = ""
        else:
            wandb_dir = ""
            wandb_url = ""
        commit_message = "llama-7b-" + self.script_args.task + ".py:" + str_args + "At wandb dir: " + wandb_dir + "\nAt wandb url: " + wandb_url + "\n"
        print("Commit message:", commit_message)
        print("Saving model to:", self.script_args.script_args_name)

        if self.script_args.log_with == "":
            print("Not pushing anything because debug")
        else:
            if self.script_args.output_folder == "auto":
                output_folder = os.path.join(args_utils.FOLDER_EXPE, self.script_args.script_args_name)
            else:
                output_folder = self.script_args.output_folder
            save_folder = os.path.join(output_folder, "epoch" + str(step))

            Path(save_folder).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_folder)
            with open(os.path.join(save_folder, 'commit.md'), 'w') as f:
                f.write(commit_message)
            if optimizer is not None:
                torch.save(optimizer.state_dict(), os.path.join(save_folder, 'optimizer.pth'))
