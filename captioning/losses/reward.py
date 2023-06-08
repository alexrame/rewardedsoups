import torch
from eval.cider.reinforce_cider import ReinforceCider
from eval.meteor.reinforce_meteor import Meteor
from eval.bleu.bleu import Bleu
from eval.rouge.rouge import Rouge
import itertools
from utils import language_utils


class ReinforceReward:

    def __init__(
        self, training_references, eos_token, num_sampled_captions, rank, reinforce_metric="cider"
    ):
        super(ReinforceReward).__init__()
        self.rank = rank
        self.num_sampled_captions = num_sampled_captions
        preprocessed_training_references = []
        for i in range(len(training_references)):
            preprocessed_captions = []
            for caption in training_references[i]:
                # it's faster than invoking the tokenizer
                caption = language_utils.lowercase_and_clean_trailing_spaces([caption])
                caption = language_utils.add_space_between_non_alphanumeric_symbols(caption)
                caption = language_utils.remove_punctuations(caption)
                caption = " ".join(caption[0].split() + [eos_token])
                preprocessed_captions.append(caption)
            preprocessed_training_references.append(preprocessed_captions)
        self.training_references = preprocessed_training_references

        self.list_dict_reward = []
        for _reinforce_reward in reinforce_metric.split(","):
            dict_reward = {"name": _reinforce_reward}
            if "-" in _reinforce_reward:
                dict_reward["coeff"] = float(_reinforce_reward.split("-")[1])
                _reinforce_reward = _reinforce_reward.split("-")[0]
            else:
                dict_reward["coeff"] = 1.0
            if _reinforce_reward == "cider":
                dict_reward["callable"] = ReinforceCider(self.training_references)
            elif _reinforce_reward.startswith("bleu"):
                n = int(_reinforce_reward[-1])
                # n between 1 and 4, dependin on thenumber of grams considered
                dict_reward["callable"] = Bleu(n=n)
            elif _reinforce_reward == "rouge":
                dict_reward["callable"] = Rouge()
            elif _reinforce_reward == "meteor":
                dict_reward["callable"] = Meteor()
            else:
                raise ValueError(reinforce_metric)
            self.list_dict_reward.append(dict_reward)

    def call_reward(self, hypo, refs):
        list_rewards = []
        sum_coeff = 0.
        for dict_reward in self.list_dict_reward:
            reward = dict_reward["callable"].compute_score_rl(
                hypo=hypo,
                refs=refs,
            )[1]
            reward = torch.tensor(reward).to(self.rank)
            list_rewards.append(dict_reward["coeff"] * reward)
            sum_coeff += dict_reward["coeff"]
        reward = torch.sum(torch.stack(list_rewards), dim=0)/sum_coeff
        return reward

    def compute_reward(
        self,
        all_images_pred_caption,
        all_images_logprob,
        all_images_idx,
        all_images_base_caption=None
    ):

        batch_size = len(all_images_pred_caption)
        num_sampled_captions = len(all_images_pred_caption[0])

        # Important for Correct and Fair results: keep EOS in the Log loss computation
        all_images_pred_caption = [
            ' '.join(caption[1:])
            for pred_one_image in all_images_pred_caption
            for caption in pred_one_image
        ]

        # repeat the references for the number of outputs
        all_images_ref_caption = [self.training_references[idx] for idx in all_images_idx]
        all_images_ref_caption = list(
            itertools.chain.from_iterable(
                itertools.repeat(ref, self.num_sampled_captions) for ref in all_images_ref_caption
            )
        )

        cider_result = self.call_reward(hypo=all_images_pred_caption, refs=all_images_ref_caption)
        reward = torch.tensor(cider_result).to(self.rank).view(batch_size, num_sampled_captions)

        if all_images_base_caption is None:  # Mean base
            reward_base = (reward.sum(dim=-1, keepdim=True) - reward) / (num_sampled_captions - 1)
        else:  # anything else like Greedy
            all_images_base_caption = [
                ' '.join(caption[1:])
                for pred_one_image in all_images_base_caption
                for caption in pred_one_image
            ]
            base_cider_result = self.call_reward(hypo=all_images_base_caption, refs=all_images_ref_caption)
            reward_base = torch.tensor(base_cider_result).to(
                self.rank).view(batch_size, len(all_images_base_caption[0]))

        reward_loss = (reward - reward_base) * torch.sum(-all_images_logprob, dim=-1)
        reward_loss = reward_loss.mean()
        return reward_loss, reward, reward_base
