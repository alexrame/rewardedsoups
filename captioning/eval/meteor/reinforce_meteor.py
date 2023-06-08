#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

# modified according to: https://github.com/tylin/coco-caption/issues/27
# to support python3.5

import numpy as np
import nltk


class Meteor:
    # we use this METEOR implementation in REINFORCE to remove the dependancy on JAVA

    def __init__(self):
        pass

    def calc_score(self, hypo, refs):
        return nltk.translate.meteor_score.meteor_score(
            [ref.split(" ") for ref in refs],
            hypo.split(" ")
        )
    def calc_score_mean(self, hypo, refs):
        scores = []
        for ref in refs:
            score = nltk.translate.meteor_score.meteor_score(
                [ref.split(" ")],
                hypo.split(" "))
            scores.append(score)
        return np.mean(scores)

    def compute_score(self, gts, res):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()

        score = []
        for id in imgIds:
            hypos = res[id]
            refs = gts[id]

            score.append(self.calc_score(hypos[0], refs))

            # Sanity check.
            assert (type(hypos) is list)
            assert (len(hypos) == 1)
            assert (type(refs) is list)
            assert (len(refs) > 0)

        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def compute_score_rl(self, hypo, refs):
        score = []
        for i in range(len(refs)):
            score.append(self.calc_score(hypo=hypo[i], refs=refs[i]))
        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "METEOR python"
