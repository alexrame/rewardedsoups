import copy
from collections import OrderedDict
import torch
import glob
import os

from transformers import LlamaForCausalLM
from peft import PeftModel
from peft.utils.save_and_load import get_peft_model_state_dict

from utils import args_utils


class Predictor:
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1}

    def __init__(self, reward_pipes, tokenizer, output_max_length, device):
        self.reward_pipes = reward_pipes
        self.tokenizer = tokenizer
        self.output_max_length = output_max_length
        self.device = device

    def get_rewards(self, texts):
        raise ValueError()

    @staticmethod
    def transform_reward(reward):
        d_reward = []
        for rew in reward:
            d = {}
            assert len(rew) == 1
            for r in rew[0]:
                d[r["label"]] = r["score"]
            d_reward.append(d)
        return d_reward

    def average_rewards(self, rewards):
        avg_reward = None
        for reward in rewards:
            if avg_reward is None:
                avg_reward = copy.deepcopy(reward)
            else:
                for a_dict_reward, r_dict_reward in zip(avg_reward, reward):
                    for label in a_dict_reward:
                        a_dict_reward[label] = a_dict_reward[label] + r_dict_reward[label]
        assert avg_reward is not None
        for i in range(len(avg_reward)):
            a_dict_reward = avg_reward[i]
            for label in a_dict_reward:
                a_dict_reward[label] = a_dict_reward[label] / len(rewards)
            a_dict_reward["n"] = args_utils.Naming.get_name_model(
                self.reward_pipes[i].model.name_or_path
            )
        return avg_reward

    def get_prediction_rewards(self, model, query_tensors):

        texts = []
        for i in range(len(query_tensors)):
            query_tensor = torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(self.device)
            output = model.generate(
                input_ids=query_tensor,
                max_new_tokens=self.output_max_length,
                pad_token_id=self.tokenizer.eos_token_id
            ).squeeze()
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            texts.append(text)

        rewards = self.get_rewards(texts)
        avg_reward = self.average_rewards(rewards)
        avg_reward.append({"length": len(query_tensors)})
        return texts, rewards, avg_reward

    def predict(self, model, query_tensors, verbose=False):
        texts, rewards, avg_reward = self.get_prediction_rewards(model, query_tensors)
        for text, reward in zip(texts, rewards):
            print("=== text:", text.replace("\n", "[NEWLINE] "), reward)
            if not verbose:
                break
        return avg_reward


class ResultsComputer:

    def __init__(self, predictor, base_model, query_tensors, verbose):
        self.predictor = predictor
        self.base_model = base_model
        self.query_tensors = query_tensors
        self.verbose = verbose

    def singlenolora(self):
        return self.predictor.predict(
            self.base_model, self.query_tensors, verbose=self.verbose
        )

    def single(self, peft_name):
        print("Single")
        return self.create_and_call_wa(
            [peft_name], coefficients=[1.], name=peft_name.split("/")[-1]
        )

    def interpolation(self, peft_names, num_lambdas):
        print("Interpolation")
        dict_coeff_to_reward = {}
        list_coeffs = [x / (num_lambdas-1) for x in range(0, num_lambdas, 1)]
        for i, coeff in enumerate(list_coeffs):
            dict_coeff_to_reward[coeff] = self.create_and_call_wa(peft_names, [1 - coeff, coeff])
        return dict_coeff_to_reward

    def average(self, peft_names):
        len_peft_names = len(peft_names)
        print("Uniform average of", len_peft_names, "models")
        coefficients = [1 / len_peft_names for _ in range(len_peft_names)]
        avg_reward = self.create_and_call_wa(
            peft_names, coefficients, name="_".join(name[0] for name in peft_names)
        )
        return avg_reward

    def create_and_call_wa(self, peft_names, coefficients, name=None):
        # 4.1 load wa
        wa = WeightAverager.build_wa(
            base_model=self.base_model, peft_names=peft_names, coefficients=coefficients
        )
        torch.cuda.empty_cache()

        # 4.2 predict with wa
        if name is None:
            coeff = 1 - coefficients[0]
            name = "wa coeff coefficient[1] " + str(coeff)
        avg_reward = self.predictor.predict(
            wa, self.query_tensors, verbose=self.verbose
        )
        print("==", name, avg_reward, "\n")

        # 4.3 del wa
        del wa
        torch.cuda.empty_cache()
        wa = None
        return avg_reward


LOAD_ONLY_LORA = True


class WeightAverager:

    @staticmethod
    def average_weights(base_model, peft_names, coefficients):
        weights_averaged = OrderedDict()
        i = 0
        for peft_name, coefficient in zip(peft_names, coefficients):
            if coefficient == 0.:
                continue
            if peft_name is None:
                print("Skipping none peft_name")
                continue
            current_model = Loader.load_peft_model(base_model, peft_name)
            assert LOAD_ONLY_LORA
            current_weights = get_peft_model_state_dict(current_model, state_dict=None)
            for key in list(current_weights.keys()):
                if i == 0:
                    weights_averaged[key] = coefficient * current_weights[key]
                else:
                    weights_averaged[key] += coefficient * current_weights[key]
                del current_weights[key]
            del current_model
            torch.cuda.empty_cache()
            i += 1
        return weights_averaged

    @staticmethod
    def build_wa(base_model, peft_names, coefficients):
        weights_averaged = WeightAverager.average_weights(
            base_model=base_model, peft_names=peft_names, coefficients=coefficients
        )

        torch.cuda.empty_cache()
        wa = Loader.load_peft_model(base_model, peft_names[0])
        wa.load_state_dict(weights_averaged, strict=not LOAD_ONLY_LORA)
        return wa


class Loader:

    @staticmethod
    def load_base_model(base_model_name):
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=args_utils.LOAD_IN_8BIT,
            device_map="auto",
            local_files_only=args_utils.LOCAL_FILES_ONLY
        )
        return base_model

    @staticmethod
    def load_peft_model(base_model, peft_name):
        peft_model = PeftModel.from_pretrained(
            base_model, peft_name, local_files_only=args_utils.LOCAL_FILES_ONLY
        )
        peft_model.eval()
        return peft_model


def get_dict_peft_names(folder):
    list_folder = sorted(glob.glob(folder))
    print("list_folder:", list_folder)
    dict_peft_names = {int(os.path.split(path)[-1].split("epoch")[1]): path for path in list_folder}
    dict_peft_names = OrderedDict(
        {key: dict_peft_names[key] for key in sorted(dict_peft_names.keys(), reverse=True)}
    )
    if os.environ.get("MINEPOCH", "0") != "0":
        dict_peft_names = {
            key: value
            for key, value in dict_peft_names.items()
            if key > int(os.environ.get("MINEPOCH"))
        }
    return dict_peft_names

def get_last_epoch(peft_name):
    if os.path.split(peft_name)[-1].startswith("epoch"):
        return peft_name
    list_folder = os.listdir(peft_name)
    dict_epochs = {int(path.split("epoch")[1]): path for path in list_folder}
    last_epoch = os.path.join(peft_name, dict_epochs[max(dict_epochs.keys())])
    print("detected", last_epoch)
    return last_epoch


def get_results_rewards(resultscomputer, peft_names, num_lambdas):
    if len(peft_names) == 1:
        if peft_names[0] == "nolora":
            dict_coeff_to_reward = {0: resultscomputer.singlenolora()}
        elif "*" in peft_names[0]:
            # detect all files that match the given regex
            dict_peft_names = get_dict_peft_names(peft_names[0])
            print("The following files have been detected", dict_peft_names)
            max_step = max(dict_peft_names.keys())
            dict_coeff_to_reward = {}
            for step, peft_name in dict_peft_names.items():
                dict_coeff_to_reward[step / max_step] = resultscomputer.single(peft_name)
                dict_coeff_to_reward[step / max_step].append({"step": step, "peft_name": peft_name})
        else:
            dict_coeff_to_reward = {0: resultscomputer.single(peft_names[0])}
    elif len(peft_names) == 2:
        peft_names = [get_last_epoch(peft_name) for peft_name in peft_names]
        dict_coeff_to_reward = resultscomputer.interpolation(peft_names, num_lambdas)
    else:
        dict_coeff_to_reward = {}
        dict_coeff_to_reward["uniform"] = resultscomputer.average(peft_names=peft_names)
        for i, peftname in enumerate(peft_names):
            dict_coeff_to_reward[i] = resultscomputer.single(peftname)

    # 4. print results
    for coeff in sorted(dict_coeff_to_reward.keys()):
        print("d[", coeff, "] =", dict_coeff_to_reward[coeff])
