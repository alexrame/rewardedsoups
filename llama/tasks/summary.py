import os
import torch
import numpy as np

from datasets import load_dataset
from trl.core import LengthSampler

from utils import llama_utils, inference_utils

device = 0 if torch.cuda.is_available() else "cpu"

MIN_SIZE = 100
MAX_SIZE_NEWS = 1500

def build_dataset(dataset_name, *args, **kwargs):
    os.environ['HF_DATASETS_OFFLINE'] = "1"
    if dataset_name == "news":
        dataset = _build_news_dataset(*args, **kwargs)
    else:
        dataset = _build_openai_dataset(*args, **kwargs)
    print(f"Loaded dataset {dataset_name}:", dataset)
    return dataset


def _build_news_dataset(
    tokenizer,
    split="train",
):
    # As the train set is very small and the validation set is larger, we swap them.
    split = {"train": "test", "validation": "train"}[split]
    ds = load_dataset("argilla/news-summary", name="comparisons", split=split)
    ds_filtered = ds.filter(
        lambda x: x["text"] is not None and MIN_SIZE < len(x["text"]) < MAX_SIZE_NEWS and x["id"] is
        not None,
        batched=False
    )
    def remove_duplicate(duplicated_dataset):
        initial_list = duplicated_dataset.map(lambda x: {"id": x['id']})
        _ , unique_indices = np.unique(initial_list["id"], return_index=True, axis=0)
        filtered_dataset = duplicated_dataset.select(unique_indices.tolist())
        return filtered_dataset

    ds_deduplicated = remove_duplicate(ds_filtered)
    input_size_sampler = LengthSampler(
        2,
        8
    )

    def tokenize(sample):
        info_post = "-".join(sample["text"].replace("\n", " ").split("(Reuters) -")[1:]).strip()
        prompt_summary = llama_utils.Instructions.get_prompt_summary(post=info_post)
        size_prompt_summary = len(tokenizer.encode(prompt_summary)) - 1
        input_size = size_prompt_summary + input_size_sampler()
        choice = 0 # select the best summary
        response = sample["prediction"][choice]["text"].replace("\n", " ").replace(".", ",")
        sample["input_ids"] = tokenizer.encode(prompt_summary + response)[:input_size]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds_mapped = ds_deduplicated.map(tokenize, batched=False, load_from_cache_file=False)
    ds_mapped.set_format(type="torch")
    return ds_mapped


def _build_openai_dataset(tokenizer, split="train", max_size=1200):
    ds = load_dataset("openai/summarize_from_feedback", name="comparisons", split=split)
    ds = ds.filter(
        lambda x: x["info"]["post"] is not None and MIN_SIZE < len(x["info"]["post"]) < max_size and
        x['info']["id"] is not None,
        batched=False
    )

    def remove_duplicate(duplicated_dataset):
        initial_list = duplicated_dataset.map(lambda x: {"id": x['info']["id"]})
        _ , unique_indices = np.unique(initial_list["id"], return_index=True, axis=0)
        filtered_dataset = duplicated_dataset.select(unique_indices.tolist())
        return filtered_dataset

    ds = remove_duplicate(ds)

    input_size_sampler = LengthSampler(
        2,
        8
    )

    def tokenize(sample):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = llama_utils.Instructions.get_prompt_summary(post=info_post)
        size_prompt_summary = len(tokenizer.encode(prompt_summary)) - 1
        input_size = size_prompt_summary + input_size_sampler()
        choice = sample["choice"] # select the best summary
        response = sample["summaries"][choice]["text"].replace("\n", " ").replace(".", ",")
        sample["input_ids"] = tokenizer.encode(prompt_summary + response)[:input_size]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False, load_from_cache_file=False)
    ds.set_format(type="torch")
    return ds


def transform_text_summary(reward_pipe, post, response):
    response = response.split(".")[0] + "." # selecting only first sentence
    if reward_pipe.model.name_or_path.startswith("CogComp/bart-faithful-summary-detector"):
        return response + reward_pipe.tokenizer.eos_token + reward_pipe.tokenizer.eos_token + post
    if reward_pipe.model.name_or_path.startswith("Tristan/gpt2_reward_summarization"):
        return response + " " + reward_pipe.tokenizer.bos_token + " " + post
    raise ValueError(reward_pipe)


class PredictorSummary(inference_utils.Predictor):

    def get_rewards(self, texts):

        queries_responses = [
            (llama_utils.Instructions.get_input(text), llama_utils.Instructions.get_response(text))
            for text in texts
        ]
        rewards = [
            [
                reward_pipe(
                    transform_text_summary(
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
            "Zinedine Yazid Zidane popularly known as Zizou, is a French professional football manager and former player who played as an attacking midfielder. He most recently coached Spanish club Real Madrid and is one of the most successful coaches in the world. Widely regarded as one of the greatest players of all time, Zidane was a playmaker renowned for his elegance, vision, passing, ball control, and technique. He received many individual accolades as a player, including being named FIFA World Player of the Year in 1998, 2000 and 2003, and winning the 1998 Ballon d'Or.",
            "Thierry Daniel Henry is a French professional football coach, pundit, and former player. Considered one of the best strikers of all time, one of the best players to play in the Premier League and Arsenal's greatest player, Henry was runner-up for both the Ballon d'Or in 2003 and the FIFA World Player of the Year in 2003 and 2004.",
            "Pablo Escobar was named the FWA Footballer of the Year a record three times, the PFA Players Player of the Year a joint-record two times, and was named in the PFA Team of the Year six consecutive times. He was also included in the FIFA FIFPro World XI once and the UEFA Team of the Year five times."
        ]
        list_responses = [
            "Zinedine Zidane is a footballer",
            "Thierry Henry is a footballer",
            "The mafia is"
        ]
        list_texts = [
            llama_utils.Instructions.get_prompt_summary(post=post) + response
            for post, response in zip(list_posts, list_responses)
        ]

        batch = [np.array(tokenizer.encode(text), dtype=np.int32)[:-1] for text in list_texts][:bs]
        return batch

    @staticmethod
    def get_samples(dataset_name, tokenizer, bs=16):
        ds = build_dataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            split="validation"
        )

        ds.set_format("pandas")
        df_batch = ds[:bs]
        query_tensors = df_batch['input_ids'].tolist()
        return query_tensors
