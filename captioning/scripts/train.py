import argparse
import os
import random
from argparse import Namespace
from time import time
import json
import numpy as np
import torch
import shutil
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from data.coco_dataset import CocoDatasetKarpathy
from data.coco_dataloader import CocoDataLoader
from scripts.test import compute_evaluation_loss, evaluate_model_on_set
from losses.loss import LabelSmoothingLoss
from losses.reward import ReinforceReward
from optims.radam import RAdam
from utils import language_utils
from utils.args_utils import str2bool, str2list, scheduler_type_choice, optim_type_choice
from utils.saving_utils import load_most_recent_checkpoint, save_last_checkpoint, partially_load_state_dict

torch.autograd.set_detect_anomaly(False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import functools

print = functools.partial(print, flush=True)


def convert_time_as_hhmmss(ticks):
    return str(int(ticks / 60)) + " m " + \
           str(int(ticks) % 60) + " s"

def train(
    rank, train_args, path_args, ddp_model, coco_dataset, data_loader, optimizer, sched, max_len
):

    if not train_args.reinforce:
        loss_function = LabelSmoothingLoss(smoothing_coeff=0.1, rank=rank)
        loss_function.to(rank)
    else:  # 'rf'
        num_sampled_captions = 5
        running_logprobs = 0
        running_reward = 0
        running_reward_base = 0

        training_references = coco_dataset.get_all_images_captions(CocoDatasetKarpathy.TrainSet_ID)
        reinforce_reward = ReinforceReward(
            training_references,
            coco_dataset.get_eos_token_str(),
            num_sampled_captions,
            rank,
            reinforce_metric=train_args.reinforce
        )

    algorithm_start_time = time()
    saving_timer_start = time()
    time_to_save = False
    running_loss = 0
    running_time = 0
    already_trained_steps = data_loader.get_num_batches() * data_loader.get_epoch_it(
    ) + data_loader.get_batch_it()
    prev_print_iter = already_trained_steps
    num_iter = data_loader.get_num_batches() * train_args.num_epochs
    for it in range(already_trained_steps, num_iter):
        iter_timer_start = time()
        ddp_model.train()
        if True:
            # here we train
            if not train_args.reinforce:
                batch_input_x, batch_target_y, \
                batch_input_x_num_pads, batch_target_y_num_pads, batch_img_idx \
                    = data_loader.get_next_batch(verbose=True *
                                                        (((it + 1) % train_args.print_every_iter == 0) or
                                                        (it + 1) % data_loader.get_num_batches() == 0),
                                                get_also_image_idxes=True)
                batch_input_x = batch_input_x.to(rank)
                batch_target_y = batch_target_y.to(rank)
                # create a list of sub-batches so tensors can be deleted right-away after being used
                pred_logprobs = ddp_model(
                    enc_x=batch_input_x,
                    dec_x=batch_target_y[:, :-1],
                    enc_x_num_pads=batch_input_x_num_pads,
                    dec_x_num_pads=batch_target_y_num_pads,
                    apply_softmax=False
                )

                loss = loss_function(
                    pred_logprobs, batch_target_y[:, 1:], coco_dataset.get_pad_token_idx()
                )

                running_loss += loss.item()
                loss.backward()
            else:  # rf mode
                batch_input_x, batch_target_y, batch_input_x_num_pads, batch_img_idx \
                    = data_loader.get_next_batch(verbose=True *
                                                        (((it + 1) % train_args.print_every_iter == 0) or
                                                        (it + 1) % data_loader.get_num_batches() == 0),
                                                get_also_image_idxes=True)
                batch_input_x = batch_input_x.to(rank)
                sampling_search_kwargs = {
                    'sample_max_seq_len': train_args.scst_max_len,
                    'how_many_outputs': num_sampled_captions,
                    'sos_idx': coco_dataset.get_sos_token_idx(),
                    'eos_idx': coco_dataset.get_eos_token_idx()
                }
                all_images_pred_idx, all_images_logprob = ddp_model(
                    enc_x=batch_input_x,
                    enc_x_num_pads=batch_input_x_num_pads,
                    mode='sampling',
                    **sampling_search_kwargs
                )

                all_images_pred_caption = [language_utils.convert_allsentences_idx2word(
                    one_image_pred_idx, coco_dataset.caption_idx2word_list) \
                    for one_image_pred_idx in all_images_pred_idx]

                reward_loss, reward, reward_base \
                    = reinforce_reward.compute_reward(all_images_pred_caption=all_images_pred_caption,
                                                    all_images_logprob=all_images_logprob,
                                                    all_images_idx=batch_img_idx)

                running_logprobs += all_images_logprob.sum().item() / len(
                    torch.nonzero(all_images_logprob, as_tuple=False)
                )
                running_reward += reward.sum().item() / len(reward.flatten())
                running_reward_base += reward_base.sum().item() / len(reward_base.flatten())
                running_loss += reward_loss.item()
                reward_loss.backward()

            if it % train_args.num_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            sched.step()

            current_rl = sched.get_last_lr()[0]

            running_time += time() - iter_timer_start

            if (it + 1) % train_args.print_every_iter == 0:
                if not train_args.reinforce:
                    avg_loss = running_loss / (it + 1 - prev_print_iter)
                    tot_elapsed_time = time() - algorithm_start_time
                    avg_time_time_per_iter = running_time / (it + 1 - prev_print_iter)
                    print(
                        '[GPU:' + str(rank) + '] ' + str(round(((it + 1) / num_iter) * 100, 3)) +
                        ' % it: ' + str(it + 1) + ' lr: ' + str(round(current_rl, 12)) +
                        ' n.acc: ' + str(train_args.num_accum) + ' avg loss: ' +
                        str(round(avg_loss, 3)) + ' elapsed: ' +
                        convert_time_as_hhmmss(tot_elapsed_time) + ' sec/iter: ' +
                        str(round(avg_time_time_per_iter, 3))
                    )
                    running_loss = 0
                    running_time = 0
                    prev_print_iter = it + 1
                else:
                    avg_loss = running_loss / (it + 1 - prev_print_iter)
                    tot_elapsed_time = time() - algorithm_start_time
                    avg_time_time_per_iter = running_time / (it + 1 - prev_print_iter)
                    avg_logprobs = running_logprobs / (it + 1 - prev_print_iter)
                    avg_reward = running_reward / (it + 1 - prev_print_iter)
                    avg_reward_base = running_reward_base / (it + 1 - prev_print_iter)
                    print(
                        '[GPU:' + str(rank) + '] ' + str(round(((it + 1) / num_iter) * 100, 3)) +
                        ' % it: ' + str(it + 1) + ' lr: ' + str(round(current_rl, 12)) +
                        ' n.acc: ' + str(train_args.num_accum) + ' avg rew loss: ' +
                        str(round(avg_loss, 3)) + ' elapsed: ' +
                        convert_time_as_hhmmss(tot_elapsed_time) + ' sec/iter: ' +
                        str(round(avg_time_time_per_iter, 3)) + '\n'
                        ' avg reward: ' + str(round(avg_reward, 5)) + ' avg base: ' +
                        str(round(avg_reward_base, 5)) + ' avg logprobs: ' +
                        str(round(avg_logprobs, 5))
                    )
                    running_loss = 0
                    running_time = 0
                    running_logprobs = 0
                    running_reward = 0
                    running_reward_base = 0
                    prev_print_iter = it + 1

        if ((it + 1) % data_loader.get_num_batches()
            == 0) or ((it + 1) % train_args.eval_every_iter == 0):
            if not train_args.reinforce:
                compute_evaluation_loss(
                    loss_function=loss_function,
                    model=ddp_model,
                    data_set=coco_dataset,
                    data_loader=data_loader,
                    num_samples=coco_dataset.val_num_images,
                    sub_batch_size=train_args.eval_parallel_batch_size,
                    dataset_split=CocoDatasetKarpathy.ValidationSet_ID,
                    rank=rank,
                    verbose=True
                )

            if rank == 0:
                print("Evaluation on Validation Set")
            score_results, _, _ = evaluate_model_on_set(
                ddp_model,
                coco_dataset.caption_idx2word_list,
                coco_dataset.get_sos_token_idx(),
                coco_dataset.get_eos_token_idx(),
                coco_dataset.val_num_images,
                data_loader,
                dataset_split=CocoDatasetKarpathy.ValidationSet_ID,
                eval_max_len=max_len,
                rank=rank,
                parallel_batches=train_args.eval_parallel_batch_size,
                beam_sizes=train_args.eval_beam_sizes,
                use_images_instead_of_features=train_args.is_end_to_end,
                get_scores=True,
            )
            time_to_save = True

        # saving
        if time_to_save:
            if rank == 0:
                module = ddp_model if not hasattr(ddp_model, "module") else ddp_model.module
                new_checkpoint_filename = save_last_checkpoint(
                    module,
                    optimizer,
                    sched,
                    data_loader,
                    path_args.save_path,
                    num_max_checkpoints=train_args.how_many_checkpoints,
                    additional_info=train_args.reinforce or 'xe'
                )
                shutil.copyfile(
                    new_checkpoint_filename,
                    os.path.join(os.path.split(new_checkpoint_filename)[0], "last_model.pth")
                )
            try:
                epochs_path = os.path.join(path_args.save_path, 'results.json')
                dict_scores = {k: v for k, v in score_results}
                dict_scores["reinforce"] = train_args.reinforce
                dict_scores["step"] = it + 1
                dict_scores["epoch"] = (it + 1) / data_loader.get_num_batches()
                dict_scores["path"] = new_checkpoint_filename
                with open(epochs_path, 'a') as f:
                    f.write(json.dumps(dict_scores, sort_keys=True) + "\n")
            except Exception as e:
                print(e)
                pass
            time_to_save = False


def singlegpu_train(
    model_args, optim_args, coco_dataset, array_of_init_seeds, model_max_len,
    train_args, path_args
):
    rank = 0
    print("GPU: " + str(1) + "] Process " + str(rank) + " working...")

    img_size = 384
    if train_args.is_end_to_end:
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
            swin_drop_path_rate=0.1,
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
        )

    model.to(rank)
    ddp_model = model

    if train_args.reinforce:
        print("REINFORCE mode")
        data_loader = CocoDataLoader(
            coco_dataset=coco_dataset,
            batch_size=train_args.batch_size,
            num_procs=1,
            array_of_init_seeds=array_of_init_seeds,
            dataloader_mode='image_wise',
            resize_image_size=img_size if train_args.is_end_to_end else None,
            rank=rank,
            verbose=True
        )
    else:
        print("Cross Entropy learning mode")
        data_loader = CocoDataLoader(
            coco_dataset=coco_dataset,
            batch_size=train_args.batch_size,
            num_procs=1,
            array_of_init_seeds=array_of_init_seeds,
            dataloader_mode='caption_wise',
            resize_image_size=img_size if train_args.is_end_to_end else None,
            rank=rank,
            verbose=True
        )

    base_lr = 1.0
    if optim_args.optim_type == 'radam':
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, ddp_model.parameters()),
            lr=base_lr,
            betas=(0.9, 0.98),
            eps=1e-9
        )
    else:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=base_lr
        )

    if optim_args.sched_type == 'annealing':
        sched_func = lambda it: (min(it, optim_args.warmup_iters) / optim_args.warmup_iters) * \
                        optim_args.lr * (0.8 ** (it // (optim_args.anneal_every_epoch * data_loader.get_num_batches())))
    else:
        num_batches = data_loader.get_num_batches()
        sched_func = lambda it: max((it >= optim_args.warmup_iters) * optim_args.min_lr,
                                    (optim_args.lr / (max(optim_args.warmup_iters - it, 1))) * \
                                    (pow(optim_args.anneal_coeff, it // (num_batches * optim_args.anneal_every_epoch)))
                                    )

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sched_func)

    if path_args.backbone_save_path != '' or path_args.body_save_path != '':
        if train_args.is_end_to_end:
            if path_args.backbone_save_path != '':
                map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                checkpoint = torch.load(path_args.backbone_save_path, map_location=map_location)
                if 'model' in checkpoint.keys():
                    partially_load_state_dict(model.swin_transf, checkpoint['model'])
                else:
                    assert 'model_state_dict' in checkpoint.keys()
                    partially_load_state_dict(model, checkpoint['model_state_dict'])
                print("Backbone loaded...")
            else:
                print("Skip backbone loaded")
            if path_args.body_save_path != '':
                map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                checkpoint = torch.load(path_args.body_save_path, map_location=map_location)
                partially_load_state_dict(model, checkpoint['model_state_dict'])
                print("Body loaded")
        else:
            assert train_args.partial_load
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(path_args.body_save_path, map_location=map_location)
            partially_load_state_dict(model, checkpoint['model_state_dict'])
            print("Partial load done.")
    else:
        print("No checkpoint path provided")
        change_from_xe_to_rf = False
        if path_args.save_path is not None:
            module = ddp_model if not hasattr(ddp_model, "module") else ddp_model.module
            _, additional_info = load_most_recent_checkpoint(
                module, optimizer, sched, data_loader, rank, path_args.save_path
            )
            if additional_info == 'xe' and train_args.reinforce:
                change_from_xe_to_rf = True
            else:
                print("Training mode still in the same stage: " + additional_info)
        changed_batch_size = data_loader.get_batch_size() != train_args.batch_size
        if changed_batch_size or change_from_xe_to_rf:
            raise ValueError("not same config")

    train(
        rank,
        train_args,
        path_args,
        ddp_model,
        coco_dataset,
        data_loader,
        optimizer,
        sched,
        model_max_len,
    )

    print("[GPU: " + str(rank) + " ] Closing...")


def spawn_train(model_args, optim_args, coco_dataset, train_args, path_args):

    max_sequence_length = coco_dataset.max_seq_len + 20
    print("Max sequence length: " + str(max_sequence_length))
    print("y vocabulary size: " + str(len(coco_dataset.caption_word2idx_dict)))

    world_size = torch.cuda.device_count()
    assert world_size == 1

    array_of_init_seeds = [random.random() for _ in range(train_args.num_epochs * 2)]
    print("Begin train")

    singlegpu_train(
        model_args=model_args,
        optim_args=optim_args,
        coco_dataset=coco_dataset,
        array_of_init_seeds=array_of_init_seeds,
        model_max_len=max_sequence_length,
        train_args=train_args,
        path_args=path_args
    )

if __name__ == "__main__":
    DATA_DIR = os.environ.get("DATA_DIR", "")
    print("DATA_DIR:", DATA_DIR)

    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--enc_drop', type=float, default=0.1)
    parser.add_argument('--dec_drop', type=float, default=0.1)
    parser.add_argument('--enc_input_drop', type=float, default=0.1)
    parser.add_argument('--dec_input_drop', type=float, default=0.1)
    parser.add_argument('--drop_other', type=float, default=0.1)

    parser.add_argument('--optim_type', type=optim_type_choice, default='radam')
    parser.add_argument('--sched_type', type=scheduler_type_choice, default='custom_warmup_anneal')

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--min_lr', type=float, default=5e-7)
    parser.add_argument('--warmup_iters', type=int, default=1)
    parser.add_argument('--anneal_coeff', type=float, default=0.8)
    parser.add_argument('--anneal_every_epoch', type=float, default=1.0)

    parser.add_argument('--batch_size', type=int, default=18)
    parser.add_argument('--num_accum', type=int, default=2)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--save_every_minutes', type=int, default=120)
    parser.add_argument('--how_many_checkpoints', type=int, default=1)
    parser.add_argument('--print_every_iter', type=int, default=1000)

    parser.add_argument('--eval_every_iter', type=int, default=999999)
    parser.add_argument('--eval_parallel_batch_size', type=int, default=16)
    parser.add_argument('--eval_beam_sizes', type=str2list, default=[5])

    parser.add_argument('--reinforce', default="")  # any metric
    parser.add_argument('--scst_max_len', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=6)

    parser.add_argument('--image_path', type=str, default=os.path.join(DATA_DIR, 'raw_data/MS_COCO_2014'))
    parser.add_argument('--captions_path', type=str, default=os.path.join(DATA_DIR, 'raw_data/'))
    parser.add_argument('--partial_load', type=str2bool, default=True)
    parser.add_argument('--backbone_save_path', type=str, default='')
    parser.add_argument('--body_save_path', type=str, default='')
    parser.add_argument('--is_end_to_end', type=str2bool, default=False)

    parser.add_argument(
        '--images_path', type=str, default=os.path.join(DATA_DIR, "raw_data/MS_COCO_2014/")
    )
    parser.add_argument('--preproc_images_hdf5_filepath', type=str, default=None)
    parser.add_argument('--features_path', type=str, default=os.path.join(DATA_DIR, "raw_data/"))

    parser.add_argument('--seed', type=int, default=775533)

    args = parser.parse_args()

    # Seed setting ---------------------------------------------
    seed = args.seed
    print("seed: " + str(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    drop_args = Namespace(
        enc=args.enc_drop,
        dec=args.dec_drop,
        enc_input=args.enc_input_drop,
        dec_input=args.dec_input_drop,
        other=args.drop_other
    )

    model_args = Namespace(
        model_dim=args.model_dim,
        N_enc=args.N_enc,
        N_dec=args.N_dec,
        dropout=args.dropout,
        drop_args=drop_args
    )
    optim_args = Namespace(
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        anneal_coeff=args.anneal_coeff,
        anneal_every_epoch=args.anneal_every_epoch,
        optim_type=args.optim_type,
        sched_type=args.sched_type
    )

    path_args = Namespace(
        save_path=args.save_path,
        images_path=args.images_path,
        captions_path=args.captions_path,
        features_path=args.features_path,
        backbone_save_path=args.backbone_save_path,
        body_save_path=args.body_save_path,
        preproc_images_hdf5_filepath=args.preproc_images_hdf5_filepath
    )

    assert args.num_gpus == 1
    train_args = Namespace(
        is_end_to_end=args.is_end_to_end,
        batch_size=args.batch_size,
        num_accum=args.num_accum,
        save_every_minutes=args.save_every_minutes,
        how_many_checkpoints=args.how_many_checkpoints,
        print_every_iter=args.print_every_iter,
        eval_every_iter=args.eval_every_iter,
        eval_parallel_batch_size=args.eval_parallel_batch_size,
        eval_beam_sizes=args.eval_beam_sizes,
        reinforce=args.reinforce,
        num_epochs=args.num_epochs,
        partial_load=args.partial_load,
        scst_max_len=args.scst_max_len
    )

    print("train batch_size:", str(args.batch_size))
    print("num_accum:", str(args.num_accum))
    print("save_path:", str(args.save_path))
    print("captions_path:", str(path_args.captions_path))

    coco_dataset = CocoDatasetKarpathy(
        images_path=path_args.images_path,
        coco_annotations_path=path_args.captions_path + "dataset_coco.json",
        train2014_bboxes_path=path_args.captions_path + "train2014_instances.json",
        val2014_bboxes_path=path_args.captions_path + "val2014_instances.json",
        preproc_images_hdf5_filepath=path_args.preproc_images_hdf5_filepath
        if train_args.is_end_to_end else None,
        precalc_features_hdf5_filepath=None
        if train_args.is_end_to_end else path_args.features_path,
        limited_num_train_images=None,
        limited_num_val_images=5000
    )

    spawn_train(
        model_args=model_args,
        optim_args=optim_args,
        coco_dataset=coco_dataset,
        train_args=train_args,
        path_args=path_args
    )
