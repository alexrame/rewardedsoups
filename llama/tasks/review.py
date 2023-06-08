import os
import torch
import numpy as np

from datasets import load_dataset
from trl.core import LengthSampler

from utils import llama_utils, inference_utils

device = 0 if torch.cuda.is_available() else "cpu"

def build_dataset(
    tokenizer, dataset_name="imdb", split="train"
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    ds = load_dataset(dataset_name, split=split)
    ds = ds.filter(lambda x: len(x["text"]) > 200, batched=False)
    size_prompt_review = llama_utils.Instructions.get_size_prompt_review(tokenizer)

    input_size_sampler = LengthSampler(
        size_prompt_review + 2,
        size_prompt_review + 8
    )

    def tokenize(sample):
        input_size = input_size_sampler()
        sample["input_ids"] = tokenizer.encode(
            llama_utils.Instructions.get_prompt_review() + sample["text"])[:input_size]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False, load_from_cache_file=False)
    ds.set_format(type="torch")
    return ds

def transform_text_review(reward_pipe, response):
    if reward_pipe.model.name_or_path.startswith("OpenAssistant/"):
        # the first reward_pipe.tokenizer.cls_token is here for historical reasons, the first experiments were launched with it
        # though it does seem to change anything
        return reward_pipe.tokenizer.cls_token + llama_utils.Instructions.instruction_review + reward_pipe.tokenizer.sep_token + response
    raise ValueError(reward_pipe)


class PredictorReview(inference_utils.Predictor):

    def get_rewards(self, texts):

        responses = [llama_utils.Instructions.get_response(q) for q in texts]
        rewards = [
            [
                reward_pipe(
                    transform_text_review(
                        reward_pipe=reward_pipe,
                        response=response,
                    ), **self.sent_kwargs
                ) for reward_pipe in self.reward_pipes
            ] for response in responses
        ]

        rewards = [self.transform_reward(reward) for reward in rewards]
        return rewards


class Samples:

    @staticmethod
    def get_fake_samples(bs, tokenizer):
        prompt_review = llama_utils.Instructions.get_prompt_review()
        list_texts = [
            prompt_review + "We really hated the horrible hint towards this badaboum.",
            prompt_review + "We really enjoyed the slight hint towards this badaboum."
        ]
        size_prompt_review = llama_utils.Instructions.get_size_prompt_review(tokenizer)
        batch = [
            np.array(tokenizer.encode(text), dtype=np.int32)[:size_prompt_review + 8]
            for text in list_texts
        ][:bs]
        return batch

    @staticmethod
    def get_samples(dataset_name, tokenizer, bs=16):
        ds = build_dataset(
            tokenizer=tokenizer, dataset_name=dataset_name, split="test"
        )

        ds.set_format("pandas")
        df_batch = ds[:bs]
        query_tensors = df_batch['input_ids'].tolist()
        return query_tensors
