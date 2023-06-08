# Image-to-text experiments: captioning with diverse statistical rewards

RL training is especially effective in image captioning where the task is to generate textual descriptions of images. We rely on various hand-engineered, non-differentiable metrics: the precision-focused BLEU, the recall-focused ROUGE METEOR handling synonyms and CIDEr using TF-IDF. We borrow the implementation of [ExpansionNet v2: Block Static Expansion in fast end to end training for Image Captioning](https://arxiv.org/abs/2208.06551v3), available [here](https://github.com/jchenghu/ExpansionNet_v2).

If you find this repository useful, please consider citing:

```
@article{hu2022expansionnet,
  title={ExpansionNet v2: Block Static Expansion in fast end to end training for Image Captioning},
  author={Hu, Jia Cheng and Cavicchioli, Roberto and Capotondi, Alessandro},
  journal={arXiv preprint arXiv:2208.06551},
  year={2022}
}
```

## Setup

### Requirements

* python >= 3.7
* numpy
* Java 1.8.0
* pytorch 1.9.0
* h5py

### Data preparation

MS-COCO 2014 images can be downloaded [here](https://cocodataset.org/#download), the respective captions are uploaded in our online [drive](https://drive.google.com/drive/folders/1bBMH4-Fw1LcQZmSzkMCqpEl0piIP88Y3?usp=sharing) and the backbone can be found [here](https://github.com/microsoft/Swin-Transformer). All files, in particular the 3 json files and the backbone are suggested to be moved in `${DATA_DIR}/raw_data/` since commands provided in the following steps assume these files are placed in that directory.

## Initialization

We conduct our experiments on COCO with an ExpansionNetv2 network and a Swin Transformer visual encoder, initialized from the state-of-the-art weights of optimized on CIDEr. We download the `rf_model.pth` from [here](https://drive.google.com/drive/folders/1bBMH4-Fw1LcQZmSzkMCqpEl0piIP88Y3?usp=sharing) into `${DATA_DIR}/saves/`. This model has sequentially been trained with Cross-Entropy (partial then end-to-end training), and then with CIDEr optimization (partial then end-to-end training) using the self-critical learning. We refer to the seminal [code](https://github.com/jchenghu/ExpansionNet_v2) for more information.

### 1. Features generation

We first generate the features, as the backbone remains frozen in most of our experiments (except in Figure 10(d)):

```
python -m scripts.data_generator\
    --save_path ${DATA_DIR}/saves/rf_model.pth\
    --features_path ${DATA_DIR}/raw_data/features_rf.hdf5\
    --image_path ${DATA_DIR}/raw_data/MS_COCO_2014/\
```

This procedure should create a pretty big file (102GB) at `${DATA_DIR}/raw_data/features_rf.hdf5`.

### 2. Evaluation

Then, you can try the evaluation setup with:

```
python -m scripts.test\
    --save_model_path ${DATA_DIR}/saves/rf_model.pth\
    --features_path ${DATA_DIR}/raw_data/features_rf.hdf5
```

It should achieve `[('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408), ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036)]`.

## Rewarded soups

The strategy is in two steps:

### Reward optimizations

We then utilize the code of and their self-critical procedure to reward the network on BLEU1, BLEU4, ROUGE or METEOR.
We use the default hyperparameters.
The following command performs the partial training using the self-critical learning for two rewards: `bleu1` and `rouge`

```
python3 train.py\
    --body_save_path ${DATA_DIR}/saves/model_rf.pth\
    --features_path ${DATA_DIR}/raw_data/features_rf.hdf5\
    --reinforce bleu1\
    --save_path ${DATA_DIR}/saves/folder_bleu1
```

It should achieve something similar to `[('CIDEr', 1.3707), ('Bleu_1', 0.8497), ('Bleu_2', 0.6931), ('Bleu_3', 0.538), ('Bleu_4', 0.4086), ('ROUGE_L', 0.6038), ('SPICE', 0.2413), ('METEOR', 0.299)]`

```
python3 train.py\
    --body_save_path ${DATA_DIR}/saves/model_rf.pth\
    --features_path ${DATA_DIR}/raw_data/features_rf.hdf5\
    --reinforce rouge\
    --save_path ${DATA_DIR}/saves/folder_rouge
```

It should achieve something similar to `[('CIDEr', 1.3701), ('Bleu_1', 0.8234), ('Bleu_2', 0.682), ('Bleu_3', 0.54), ('Bleu_4', 0.4187), ('ROUGE_L', 0.61), ('SPICE', 0.2388), ('METEOR', 0.3011)]`
We will release the fine-tuned weights.

### Inference of interpolated weights

Then given the obtained weights, you can perform the weight interpolation of their weights while sliding the interpolating coefficient.

```
for lambda in 0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python -m scripts.test\
        --ensemble wa\
        --coeffs [$lambda]\
        --save_model_path ${DATA_DIR}/saves/folder_bleu1/last_model.pth ${DATA_DIR}/saves/folder_rouge/last_model.pth\
        --features_path ${DATA_DIR}/raw_data/features_rf.hdf5
done
```

This should provide a Pareto-front of solutions. We obtain the following results:

```
results[0.0] = [('CIDEr', 1.3701), ('Bleu_1', 0.8234), ('Bleu_2', 0.682), ('Bleu_3', 0.54), ('Bleu_4', 0.4187), ('ROUGE_L', 0.61), ('SPICE', 0.2388), ('METEOR', 0.3011)]
results[0.1] = [('CIDEr', 1.3732), ('Bleu_1', 0.8265), ('Bleu_2', 0.6842), ('Bleu_3', 0.5411), ('Bleu_4', 0.419), ('ROUGE_L', 0.6095), ('SPICE', 0.2398), ('METEOR', 0.3013)]
results[0.2] = [('CIDEr', 1.3781), ('Bleu_1', 0.8309), ('Bleu_2', 0.6871), ('Bleu_3', 0.5429), ('Bleu_4', 0.42), ('ROUGE_L', 0.6093), ('SPICE', 0.2407), ('METEOR', 0.3016)]
results[0.3] = [('CIDEr', 1.381), ('Bleu_1', 0.8342), ('Bleu_2', 0.6895), ('Bleu_3', 0.5443), ('Bleu_4', 0.4205), ('ROUGE_L', 0.6096), ('SPICE', 0.2412), ('METEOR', 0.3019)]
results[0.4] = [('CIDEr', 1.384), ('Bleu_1', 0.8369), ('Bleu_2', 0.6911), ('Bleu_3', 0.5448), ('Bleu_4', 0.42), ('ROUGE_L', 0.6095), ('SPICE', 0.2414), ('METEOR', 0.3018)]
results[0.5] = [('CIDEr', 1.3868), ('Bleu_1', 0.8395), ('Bleu_2', 0.6922), ('Bleu_3', 0.5448), ('Bleu_4', 0.4194), ('ROUGE_L', 0.6095), ('SPICE', 0.2422), ('METEOR', 0.3022)]
results[0.6] = [('CIDEr', 1.3842), ('Bleu_1', 0.8418), ('Bleu_2', 0.693), ('Bleu_3', 0.5442), ('Bleu_4', 0.4182), ('ROUGE_L', 0.6082), ('SPICE', 0.2424), ('METEOR', 0.3017)]
results[0.7] = [('CIDEr', 1.382), ('Bleu_1', 0.844), ('Bleu_2', 0.6932), ('Bleu_3', 0.5429), ('Bleu_4', 0.4161), ('ROUGE_L', 0.6069), ('SPICE', 0.242), ('METEOR', 0.3015)]
results[0.8] = [('CIDEr', 1.3789), ('Bleu_1', 0.846), ('Bleu_2', 0.6932), ('Bleu_3', 0.5412), ('Bleu_4', 0.4132), ('ROUGE_L', 0.6054), ('SPICE', 0.2418), ('METEOR', 0.3006)]
results[0.9] = [('CIDEr', 1.3756), ('Bleu_1', 0.8486), ('Bleu_2', 0.6939), ('Bleu_3', 0.5399), ('Bleu_4', 0.4107), ('ROUGE_L', 0.6047), ('SPICE', 0.2413), ('METEOR', 0.3002)]
results[1.0] = [('CIDEr', 1.3707), ('Bleu_1', 0.8497), ('Bleu_2', 0.6931), ('Bleu_3', 0.538), ('Bleu_4', 0.4086), ('ROUGE_L', 0.6038), ('SPICE', 0.2413), ('METEOR', 0.299)]
```

### Multi objective RL

To reproduce the MORL baseline, simply link different rewards with commas, for example `bleu1-0.2,rouge-0.8`; then the reward is $0.2 \times bleu1 + 0.8 \times rouge$.

```
python3 train.py\
    --body_save_path ${DATA_DIR}/saves/model_rf.pth\
    --features_path ${DATA_DIR}/raw_data/features_rf.hdf5\
    --reinforce bleu1-0.2,rouge-0.8\
    --save_path ${DATA_DIR}/saves/folder_morl_bleu1-0.2_rouge-0.8
```

The trained point should be somewhere on the Pareto-front of solutions. The issue being that for each new user's preference, you would need an additional training with adapted reward weightings.
