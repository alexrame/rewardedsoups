import os
import random
import math
import torch
import argparse
from argparse import Namespace
from utils import args_utils
import copy
from time import time

from utils.saving_utils import partially_load_state_dict
from models.ensemble_captioning_model import EnsembleCaptioningModel
from data.coco_dataloader import CocoDataLoader
from data.coco_dataset import CocoDatasetKarpathy
from utils import language_utils
from utils.language_utils import compute_num_pads as compute_num_pads
from eval.eval import COCOEvalCap

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import functools

print = functools.partial(print, flush=True)


def convert_time_as_hhmmss(ticks):
    return str(int(ticks / 60)) + " m " + \
           str(int(ticks) % 60) + " s"


def compute_evaluation_loss(
    loss_function,
    model,
    data_set,
    data_loader,
    num_samples,
    sub_batch_size,
    dataset_split,
    rank=0,
    verbose=False
):
    model.eval()

    sb_size = sub_batch_size

    tot_loss = 0
    num_sub_batch = int(num_samples / sb_size)
    tot_num_tokens = 0
    for sb_it in range(num_sub_batch):
        from_idx = sb_it * sb_size
        to_idx = min((sb_it + 1) * sb_size, num_samples)

        sub_batch_input_x, sub_batch_target_y, sub_batch_input_x_num_pads, sub_batch_target_y_num_pads, \
          = data_loader.get_batch_samples(img_idx_batch_list=list(range(from_idx, to_idx)),
                                          dataset_split=dataset_split)
        sub_batch_input_x = sub_batch_input_x.to(rank)
        sub_batch_target_y = sub_batch_target_y.to(rank)

        sub_batch_input_x = sub_batch_input_x
        sub_batch_target_y = sub_batch_target_y
        tot_num_tokens += sub_batch_target_y.size(1)*sub_batch_target_y.size(0) - \
                          sum(sub_batch_target_y_num_pads)
        pred = model(
            enc_x=sub_batch_input_x,
            dec_x=sub_batch_target_y[:, :-1],
            enc_x_num_pads=sub_batch_input_x_num_pads,
            dec_x_num_pads=sub_batch_target_y_num_pads,
            apply_softmax=False
        )
        tot_loss += loss_function(
            pred,
            sub_batch_target_y[:, 1:],
            data_set.get_pad_token_idx(),
            divide_by_non_zeros=False
        ).item()
        del sub_batch_input_x, sub_batch_target_y, pred
        torch.cuda.empty_cache()
    tot_loss /= tot_num_tokens
    if verbose and rank == 0:
        print("Validation Loss on " + str(num_samples) + " samples: " + str(tot_loss))

    return tot_loss


def evaluate_model(
    ddp_model,
    y_idx2word_list,
    beam_size,
    max_seq_len,
    sos_idx,
    eos_idx,
    rank,
    parallel_batches=16,
    indexes=[0],
    data_loader=None,
    dataset_split=CocoDatasetKarpathy.TrainSet_ID,
    use_images_instead_of_features=False,
    verbose=True,
    stanford_model_path="./eval/get_stanford_models.sh",
):

    start_time = time()

    sub_list_predictions = []
    validate_y = []
    num_samples = len(indexes)

    ddp_model.eval()
    with torch.no_grad():
        sb_size = parallel_batches
        num_iter_sub_batches = math.ceil(len(indexes) / sb_size)
        for sb_it in range(num_iter_sub_batches):
            start_time = time()
            last_iter = sb_it == num_iter_sub_batches - 1
            if last_iter:
                from_idx = sb_it * sb_size
                to_idx = num_samples
            else:
                from_idx = sb_it * sb_size
                to_idx = (sb_it + 1) * sb_size

            if use_images_instead_of_features:
                sub_batch_x = [
                    data_loader.get_images_by_idx(
                        i, dataset_split=dataset_split, transf_mode='test'
                    ).unsqueeze(0) for i in list(range(from_idx, to_idx))
                ]
                sub_batch_x = torch.cat(sub_batch_x).to(rank)
                sub_batch_x_num_pads = [0] * sub_batch_x.size(0)
            else:
                sub_batch_x = [
                    data_loader.get_bboxes_by_idx(i, dataset_split=dataset_split)
                    for i in list(range(from_idx, to_idx))
                ]
                sub_batch_x = torch.nn.utils.rnn.pad_sequence(sub_batch_x,
                                                              batch_first=True).to(rank)
                sub_batch_x_num_pads = compute_num_pads(sub_batch_x)

            validate_y += [data_loader.get_all_image_captions_by_idx(i, dataset_split=dataset_split) \
                           for i in list(range(from_idx, to_idx))]

            beam_search_kwargs = {
                'beam_size': beam_size,
                'beam_max_seq_len': max_seq_len,
                'sample_or_max': 'max',
                'how_many_outputs': 1,
                'sos_idx': sos_idx,
                'eos_idx': eos_idx
            }
            output_words, _ = ddp_model(
                enc_x=sub_batch_x,
                enc_x_num_pads=sub_batch_x_num_pads,
                mode='beam_search',
                **beam_search_kwargs
            )
            output_words = [output_words[i][0] for i in range(len(output_words))]

            pred_sentence = language_utils.convert_allsentences_idx2word(
                output_words, y_idx2word_list
            )
            for sentence in pred_sentence:
                sub_list_predictions.append(' '.join(sentence[1:-1]))  # remove EOS and SOS

            del sub_batch_x, sub_batch_x_num_pads, output_words

    print("Finished preds")
    ddp_model.train()

    if rank == 0 and verbose:
        # dirty code to leave the evaluation code untouched
        list_predictions = [sub_predictions for sub_predictions in sub_list_predictions]
        list_list_references = [
            [validate_y[i][j] for j in range(len(validate_y[i]))] for i in range(len(validate_y))
        ]

        gts_dict = dict()
        for i in range(len(list_list_references)):
            gts_dict[i] = [
                {
                    u'image_id': i,
                    u'caption': list_list_references[i][j]
                } for j in range(len(list_list_references[i]))
            ]

        pred_dict = dict()
        for i in range(len(list_predictions)):
            pred_dict[i] = [{u'image_id': i, u'caption': list_predictions[i]}]
        print("Begin coco eval")
        coco_eval = COCOEvalCap(
            gts_dict,
            pred_dict,
            list(range(len(list_predictions))),
            get_stanford_models_path=stanford_model_path
        )
        print("End coco eval")
        score_results = coco_eval.evaluate(
            bleu=True, rouge=True, cider=True, spice=True, meteor=True, verbose=False
        )
        print("End scores")
        elapsed_ticks = time() - start_time
        print(
            "Evaluation Phase over " + str(len(validate_y)) + " BeamSize: " + str(beam_size) +
            "  elapsed: " + str(int(elapsed_ticks / 60)) + " m " + str(int(elapsed_ticks % 60)) +
            ' s'
        )
        print(score_results)

    if rank == 0:
        return {"pred_dict": pred_dict, "gts_dict": gts_dict, "score_results": score_results}

    return None, None


def evaluate_model_on_set(
    ddp_model,
    caption_idx2word_list,
    sos_idx,
    eos_idx,
    num_samples,
    data_loader,
    dataset_split,
    eval_max_len,
    rank,
    parallel_batches=16,
    beam_sizes=[1],
    stanford_model_path='./eval/get_stanford_models.sh',
    use_images_instead_of_features=False,
    get_scores=False
):

    with torch.no_grad():
        ddp_model.eval()

        for beam in beam_sizes:
            dict_out = evaluate_model(
                ddp_model,
                y_idx2word_list=caption_idx2word_list,
                beam_size=beam,
                max_seq_len=eval_max_len,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                rank=rank,
                parallel_batches=parallel_batches,
                indexes=list(range(num_samples)),
                data_loader=data_loader,
                dataset_split=dataset_split,
                use_images_instead_of_features=use_images_instead_of_features,
                verbose=True,
                stanford_model_path=stanford_model_path
            )

            if rank == 0:
                if get_scores:
                    return dict_out["score_results"], dict_out["pred_dict"], dict_out["gts_dict"]

    return None, None, None


def get_wa_model(checkpoints_paths, map_location, coeffs):

    wa_state_dict = {}
    num_models = len(checkpoints_paths)

    for i in range(num_models):
        checkpoint = torch.load(checkpoints_paths[i], map_location=map_location)
        state_dict = checkpoint['model_state_dict']
        if i == 0:
            wa_state_dict = {k: state_dict[k] * coeffs[i] for k in state_dict.keys()}
        else:
            keys = list(set(state_dict.keys()) & set(wa_state_dict.keys()))
            wa_state_dict = {k: state_dict[k] * coeffs[i] + wa_state_dict[k] for k in keys}

    return wa_state_dict


def get_ensemble_model(reference_model, checkpoints_paths, rank=0, coeffs=None):
    model_list = []
    for i in range(len(checkpoints_paths)):
        model = copy.deepcopy(reference_model)
        model.to(rank)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(checkpoints_paths[i], map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_list.append(model)

    model = EnsembleCaptioningModel(model_list, rank, coeffs=coeffs).to(rank)
    ddp_model = model
    return ddp_model


def test(
    rank, world_size, is_end_to_end, model_args, ensemble_args, coco_dataset,
    eval_parallel_batch_size, eval_beam_sizes, show_predictions, array_of_init_seeds, model_max_len,
    save_model_path
):
    print("GPU: " + str(rank) + "] Process " + str(rank) + " working...")

    img_size = 384
    swin_drop_path_rate=0.
    if is_end_to_end:
        from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
        model = End_ExpansionNet_v2(
            swin_img_size=img_size,
            swin_patch_size=4,
            swin_in_chans=3,
            swin_embed_dim=192,
            swin_depths=[2, 2, 18, 2],
            swin_num_heads=[6, 12, 24, 48],
            swin_window_size=12,
            swin_mlp_ratio=4.,
            swin_qkv_bias=True,
            swin_qk_scale=None,
            swin_drop_rate=0.0,
            swin_attn_drop_rate=0.0,
            swin_drop_path_rate=swin_drop_path_rate,
            swin_norm_layer=torch.nn.LayerNorm,
            swin_ape=False,
            swin_patch_norm=True,
            swin_use_checkpoint=False,
            final_swin_dim=1536,
            d_model=model_args.model_dim,
            N_enc=model_args.N_enc,
            N_dec=model_args.N_dec,
            num_heads=8,
            ff=2048,
            num_exp_enc_list=[32, 64, 128, 256, 512],
            num_exp_dec=16,
            output_word2idx=coco_dataset.caption_word2idx_dict,
            output_idx2word=coco_dataset.caption_idx2word_list,
            max_seq_len=model_max_len,
            drop_args=model_args.drop_args,
            rank=rank
        )
    else:
        from models.ExpansionNet_v2 import ExpansionNet_v2
        model = ExpansionNet_v2(
            d_model=model_args.model_dim,
            N_enc=model_args.N_enc,
            N_dec=model_args.N_dec,
            num_heads=8,
            ff=2048,
            num_exp_enc_list=[32, 64, 128, 256, 512],
            num_exp_dec=16,
            output_word2idx=coco_dataset.caption_word2idx_dict,
            output_idx2word=coco_dataset.caption_idx2word_list,
            max_seq_len=model_max_len,
            drop_args=model_args.drop_args,
            img_feature_dim=1536,
            rank=rank
        )

    model.to(rank)
    ddp_model = model

    data_loader = CocoDataLoader(
        coco_dataset=coco_dataset,
        batch_size=1,
        num_procs=world_size,
        array_of_init_seeds=array_of_init_seeds,
        dataloader_mode='image_wise',
        resize_image_size=img_size if is_end_to_end else None,
        rank=rank,
        verbose=False
    )
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

    if ensemble_args.body_save_path:
        assert is_end_to_end
        print("Body save path:", ensemble_args.body_save_path)
        checkpoint = torch.load(ensemble_args.body_save_path, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if not ensemble_args.ensemble:
        assert len(ensemble_args.coeffs) == 0
        print("Not ensemble nor weight averaging")
        checkpoint = torch.load(save_model_path[0], map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        print("Ensembling:", ensemble_args.ensemble, " for evaluation with coeffs", ensemble_args.coeffs)
        if os.path.isdir(save_model_path[0]):
            print("All pths in dir")
            checkpoints_list = [
                os.path.join(save_model_path[0], elem) for elem in os.listdir(save_model_path[0]) if elem.endswith('.pth')
            ]
        else:
            assert len(save_model_path) > 1
            checkpoints_list = save_model_path
        print("Detected checkpoints: " + str(checkpoints_list))

        if len(checkpoints_list) == 0:
            print("No checkpoints found")
            exit(-1)
        num_models = len(checkpoints_list)

        if len(ensemble_args.coeffs) == 0:
            coeffs = [1./num_models for _ in range(num_models)]
        elif len(ensemble_args.coeffs) == 1:
            if num_models == 2:
                coeff = ensemble_args.coeffs[0]
                coeffs = [1. - coeff, coeff]
            else:
                assert num_models == 1
                coeffs = ensemble_args.coeffs
        else:
            coeffs = [float(c) for c in ensemble_args.coeffs]
            assert len(coeffs) == num_models

        if ensemble_args.ensemble == "wa":
            state_dict = get_wa_model(checkpoints_list, map_location=map_location, coeffs=coeffs)
            model.load_state_dict(state_dict, strict=True)
        else:
            assert ensemble_args.ensemble == "ens"
            ddp_model = get_ensemble_model(model, checkpoints_list, rank=rank)

    print(f"Evaluation")
    score_results, pred_dict, gts_dict = evaluate_model_on_set(
        ddp_model,
        coco_dataset.caption_idx2word_list,
        coco_dataset.get_sos_token_idx(),
        coco_dataset.get_eos_token_idx(),
        coco_dataset.val_num_images,
        data_loader,
        CocoDatasetKarpathy.ValidationSet_ID,
        model_max_len,
        rank=rank,
        parallel_batches=eval_parallel_batch_size,
        use_images_instead_of_features=is_end_to_end,
        beam_sizes=eval_beam_sizes,
        get_scores=True
    )
    if ensemble_args.ensemble == "wa":
        score_results.append(("lambda", coeffs))
        print(score_results)
        if os.environ.get("VERBOSE", "0") != "0":
            print("preds", pred_dict)
            print("gts", gts_dict)


def launch_test(
    is_end_to_end, model_args, ensemble_args, coco_dataset, eval_parallel_batch_size, eval_beam_sizes,
    show_predictions, num_gpus, save_model_path
):

    max_sequence_length = coco_dataset.max_seq_len + 20
    print("Max sequence length: " + str(max_sequence_length))
    print("y vocabulary size: " + str(len(coco_dataset.caption_word2idx_dict)))

    world_size = torch.cuda.device_count()
    assert world_size == 1, "Only 1 GPU is supported for test"
    print("Using - ", world_size, " processes / GPUs!")
    assert (num_gpus <= world_size), "requested num gpus higher than the number of available gpus "
    print("Requested num GPUs: " + str(num_gpus))

    array_of_init_seeds = [random.random() for _ in range(10)]

    test(rank=0,
        world_size=1,
        is_end_to_end=is_end_to_end,
        model_args=model_args,
        ensemble_args=ensemble_args,
        coco_dataset=coco_dataset,
        eval_parallel_batch_size=eval_parallel_batch_size,
        eval_beam_sizes=eval_beam_sizes,
        show_predictions=show_predictions,
        array_of_init_seeds=array_of_init_seeds,
        model_max_len=max_sequence_length,
        save_model_path=save_model_path)


if __name__ == "__main__":
    DATA_DIR = os.environ.get("DATA_DIR","")
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--show_predictions', type=args_utils.str2bool, default=False)

    parser.add_argument('--is_end_to_end', type=args_utils.str2bool, default=False)
    parser.add_argument('--ensemble', default="") # either None, or "wa" for weight averaging or "ens" for functional ensembling
    parser.add_argument('--coeffs', type=args_utils.str2listf, default=[])

    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument(
        '--save_model_path',
        nargs="+", type=str, default=None)

    parser.add_argument('--eval_parallel_batch_size', type=int, default=16)
    parser.add_argument('--eval_beam_sizes', type=args_utils.str2list, default=[5])

    parser.add_argument(
        '--images_path',
        type=str,
        default=os.path.join(DATA_DIR, "raw_data/MS_COCO_2014/")
    )
    parser.add_argument('--body_save_path', type=str, default='')
    parser.add_argument('--preproc_images_hdf5_filepath', type=str, default=None)
    parser.add_argument(
        '--features_path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--captions_path',
        type=str,
        default=os.path.join(DATA_DIR, 'raw_data/')
    )

    args = parser.parse_args()

    print('Test args:')
    args.save_model_path = [s for s in args.save_model_path if s != "no"]
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))
    assert (args.eval_parallel_batch_size % args.num_gpus == 0), \
        "num gpus must be multiple of the requested parallel batch size"

    drop_args = Namespace(enc=0.0, dec=0.0, enc_input=0.0, dec_input=0.0, other=0.0)

    model_args = Namespace(
        model_dim=args.model_dim,
        N_enc=args.N_enc,
        N_dec=args.N_dec,
        dropout=0.0,
        drop_args=drop_args
    )

    ensemble_args = Namespace(
        ensemble=args.ensemble, coeffs=args.coeffs, body_save_path=args.body_save_path
    )

    coco_dataset = CocoDatasetKarpathy(
        images_path=args.images_path,
        coco_annotations_path=args.captions_path + "dataset_coco.json",
        train2014_bboxes_path=args.captions_path + "train2014_instances.json",
        val2014_bboxes_path=args.captions_path + "val2014_instances.json",
        preproc_images_hdf5_filepath=args.preproc_images_hdf5_filepath
        if args.is_end_to_end else None,
        precalc_features_hdf5_filepath=None if args.is_end_to_end else args.features_path,
        limited_num_train_images=None,
        limited_num_val_images=5000
    )

    launch_test(
        is_end_to_end=args.is_end_to_end,
        model_args=model_args,
        ensemble_args=ensemble_args,
        coco_dataset=coco_dataset,
        eval_parallel_batch_size=args.eval_parallel_batch_size,
        eval_beam_sizes=args.eval_beam_sizes,
        show_predictions=args.show_predictions,
        num_gpus=args.num_gpus,
        save_model_path=args.save_model_path
    )
