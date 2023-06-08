import os
import numpy as np

from datasets import load_dataset, load_from_disk
from trl.core import LengthSampler

from utils import llama_utils, args_utils, inference_utils

MIN_SIZE = 5


def build_dataset(dataset_name, *args, **kwargs):
    os.environ['HF_DATASETS_OFFLINE'] = "1"
    dataset = _build_rlhf_dataset(dataset_name, *args, **kwargs)
    print(f"Loaded dataset {dataset_name}:", dataset)
    return dataset


def _build_rlhf_dataset(dataset_name, tokenizer, split="train", max_size=100):
    split = {"train": "train", "validation": "test"}[split]

    if args_utils.LOCAL_FILES_ONLY:
        ds = load_from_disk(dataset_name)
        ds = ds[split]
    else:
        ds = load_dataset("Anthropic/hh-rlhf", name="comparisons", split=split)
    ds_filtered = ds.filter(
        lambda x: x["chosen"] is not None and MIN_SIZE <
        len(x["chosen"].split("Assistant: ")[0]) < max_size,
        batched=False
    )

    input_size_sampler = LengthSampler(3, 9)

    def tokenize(sample):
        text = sample["chosen"].replace("\n", " ")
        instruction = text.split("Assistant: ")[0].split("Human: ")[1]
        prompt = llama_utils.Instructions.get_prompt_noinput(instruction=instruction)
        response = text.split("Assistant: ")[1]
        size_prompt = len(tokenizer.encode(prompt)) - 1
        input_size = size_prompt + input_size_sampler()
        sample["input_ids"] = tokenizer.encode(prompt + response)[:input_size][:-1]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds_mapped = ds_filtered.map(tokenize, batched=False, load_from_cache_file=False)
    ds_mapped.set_format(type="torch")
    return ds_mapped


def transform_text_assistant(reward_pipe, post, response):
    if reward_pipe.model.name_or_path.startswith("OpenAssistant/"):
        return post + reward_pipe.tokenizer.sep_token + response
    if reward_pipe.model.name_or_path.startswith("theblackcat102"):
        return post + reward_pipe.tokenizer.sep_token + response
    raise ValueError(reward_pipe)


class PredictorAssistant(inference_utils.Predictor):

    def get_rewards(self, texts):

        queries_responses = [
            (llama_utils.Instructions.get_input(text), llama_utils.Instructions.get_response(text))
            for text in texts
        ]
        rewards = [
            [
                reward_pipe(
                    transform_text_assistant(
                        reward_pipe=reward_pipe, post=query, response=response
                    ), **self.sent_kwargs
                ) for reward_pipe in self.reward_pipes
            ] for query, response in queries_responses
        ]

        rewards = [self.transform_reward(reward) for reward in rewards]
        return rewards


class Samples:

    @staticmethod
    def get_fake_samples(bs, tokenizer):

        list_posts = [
            'What does DNA stand for?',
            'Find the area of a circle given its radius.',
            'is there any people not worth living?',
        ]
        list_responses = [
            "DNA stands for ",
            "The formula for ",
            'Some people are so bad the are not worth living. For example,',
        ]
        list_texts = [
            llama_utils.Instructions.get_prompt_noinput(instruction=post) + response
            for post, response in zip(list_posts, list_responses)
        ]

        batch = [np.array(tokenizer.encode(text), dtype=np.int32)[:-1] for text in list_texts][:bs]
        return batch

    @staticmethod
    def get_samples(dataset_name, tokenizer, bs=16):
        ds = build_dataset(
            dataset_name=dataset_name, tokenizer=tokenizer, split="validation"
        )

        ds.set_format("pandas")
        df_batch = ds[:bs]
        query_tensors = df_batch['input_ids'].tolist()
        return query_tensors
