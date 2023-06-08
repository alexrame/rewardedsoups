import os
from eval.tokenizer.ptbtokenizer import PUNCTUATIONS
from nltk.tokenize.stanford import StanfordTokenizer

class PythonTokenizer():

    def __init__(self):
        print("Using NLTK tokenizer")
        folder = os.path.dirname(os.path.abspath(__file__))
        self.t = StanfordTokenizer(
            path_to_jar=os.path.join(folder, "stanford-corenlp-3.4.1.jar")
        )
        #options={"americanize": True})

    def tokenize(self, gts):
        gts_final_tokenized_captions_for_image = {}
        for k, vs in gts.items():
            if not k in gts_final_tokenized_captions_for_image:
                gts_final_tokenized_captions_for_image[k] = []
            for v in vs:
                tokenized_caption = self.t.tokenize(v['caption'])
                tokenized_caption = ' '.join(
                    [w for w in tokenized_caption if w not in PUNCTUATIONS]
                ).lower()
                gts_final_tokenized_captions_for_image[k].append(tokenized_caption)
        return gts_final_tokenized_captions_for_image
