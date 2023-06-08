import os
import torch
import numpy as np

from datasets import load_dataset, load_from_disk
from trl.core import LengthSampler

from utils import llama_utils, inference_utils, args_utils

assert torch.cuda.is_available()
device = 0 if torch.cuda.is_available() else "cpu"

MIN_SIZE = 5


def build_dataset(dataset_name, *args, **kwargs):
    os.environ['HF_DATASETS_OFFLINE'] = "1"
    dataset = _build_stack_dataset(dataset_name=dataset_name, *args, **kwargs)
    print(f"Loaded dataset {dataset_name}:", dataset)
    return dataset


def _build_stack_dataset(
    dataset_name,
    tokenizer,
    split="train",
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
    split = {"train": "train", "validation": "test"}[split]

    if args_utils.LOCAL_FILES_ONLY:
        ds = load_from_disk(dataset_name)
        ds = ds[split]
    else:
        ds = load_dataset("lvwerra/stack-exchange-paired")
        def remove_duplicate(duplicated_dataset):
            initial_list = duplicated_dataset.map(lambda x: {"id": x['qid']})
            _ , unique_indices = np.unique(initial_list["qid"], return_index=True, axis=0)
            filtered_dataset = duplicated_dataset.select(unique_indices.tolist())
            return filtered_dataset

        dtrainf = ds["train"].filter(lambda x: len(x["question"]) < 300, batched=False)
        dtrainfs = dtrainf.select(range(40000))
        dtrainf_deduplicated = remove_duplicate(dtrainfs)

        dtestf = ds["test"].filter(lambda x: len(x["question"]) < 300, batched=False)
        dtestfs = dtestf.select(range(1000))
        dtestf_deduplicated = remove_duplicate(dtestfs)

        ds = {"train": dtrainf_deduplicated, "test": dtestf_deduplicated}

        # you can do this to save time, as the previous steps can be time consuming
        # import datasets
        # dd = datasets.DatasetDict(dict_data)
        # dd.save_to_disk(dataset_name)

        ds = ds[split]

    input_size_sampler = LengthSampler(3, 9)

    def tokenize(sample):
        instruction = sample["question"].replace("\n", " ")
        prompt = llama_utils.Instructions.get_prompt_noinput(instruction=instruction,)
        size_prompt = len(tokenizer.encode(prompt)) - 1
        response = sample["response_k"].replace("\n", " ")
        input_size = size_prompt + input_size_sampler()
        sample["input_ids"] = tokenizer.encode(prompt + response)[:input_size][:-1]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds_mapped = ds.map(tokenize, batched=False, load_from_cache_file=False)
    ds_filtered = ds_mapped.filter(lambda x: len(x["input_ids"]) < 512, batched=False)
    ds_filtered.set_format(type="torch")
    return ds_filtered


def transform_text_stack(reward_pipe, post, response):
    if reward_pipe.model.name_or_path.startswith("OpenAssistant/"):
        return post + reward_pipe.tokenizer.sep_token + response
    raise ValueError(reward_pipe)


class PredictorStack(inference_utils.Predictor):

    def get_rewards(self, texts):

        queries_responses = [
            (llama_utils.Instructions.get_input(text), llama_utils.Instructions.get_response(text))
            for text in texts
        ]
        rewards = [
            [
                reward_pipe(
                    transform_text_stack(
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
            'For example, User adds this "iamsmelly.com". And if I add an href to this, the link would be www.mywebsite.com/iamsmelly.com Is there a way to make it absolute if its not prepended by an http:// ? Or should I revert to jQuery for t',
            'I have a `<div id="content">`. I want to load the content from <http://vietduc24h.com> into my `div`: ``` <html> <head> <script type="text/javascript"> $(document).ready(function() { $("#content").attr("src","http://vietduc24h.com"); }) </script> </head> <body> <div id="content"></div> </body> </html ``` I dont want to use an iframe. How can I do this?',
        ]
        list_responses = [
            "As stated in the docs The Support Library 26.0 provide",
            "Thierry Henry is a footballer",
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
