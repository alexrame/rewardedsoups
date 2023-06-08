import os
from transformers import LlamaTokenizer
from transformers import pipeline
from utils import args_utils


class Pipelines:

    @staticmethod
    def load_pipes(reward_models, device):
        return [
            Pipelines.load_pipe(reward_model, device) for reward_model in reward_models
        ]

    @staticmethod
    def load_pipe(reward_model, device):
        print(f"Load reward model: {reward_model}")
        pipe = pipeline("text-classification", model=reward_model, device=device,
                        tokenizer=Tokenizer.load_tokenizer_name(reward_model))
        return pipe

class Tokenizer:

    @staticmethod
    def load_tokenizer(base_model_name):
        tokenizer_name = Tokenizer.load_tokenizer_name(base_model_name)
        tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_name,
            add_eos_token=True,
            padding_side="left",
            local_files_only=args_utils.LOCAL_FILES_ONLY
        )
        tokenizer.pad_token_id = 0
        return tokenizer

    @staticmethod
    def load_tokenizer_name(model_name):
        if "llama-7b" in model_name or "lora-7b" in model_name:
            return "decapoda-research/llama-7b-hf"
        return model_name


class Instructions:
    instruction_review = "Generate a movie review."
    instruction_summary = "Generate a one-sentence summary of this post."

    response_split = "### Response:"
    input_split = "### Input:"
    instruction_split = "### Instruction:"

    @classmethod
    def get_prompt_review(cls):
        return cls.get_prompt_noinput(cls.instruction_review)

    @classmethod
    def get_size_prompt_review(cls, tokenizer):
        prompt_review = cls.get_prompt_review()
        return len(tokenizer.encode(prompt_review)) - 1

    @classmethod
    def get_prompt_summary(cls, post):
        return cls.get_prompt_input(cls.instruction_summary, post)

    @classmethod
    def get_prompt_input(cls, instruction, input):
        return f"### Instruction: {instruction} ### Input: {input} ### Response: "

    @staticmethod
    def get_prompt_noinput(instruction):
        return f"### Instruction: {instruction} ### Response: "

    @staticmethod
    def get_input(query):
        after_input = ". ".join(query.split(Instructions.input_split)[1:]).replace("\n", " ").strip()
        before_response = after_input.split(Instructions.response_split)[0]
        return before_response

    @staticmethod
    def get_response(response):
        return ". ".join(response.split(Instructions.response_split)[1:]).replace("\n", " ").strip()
