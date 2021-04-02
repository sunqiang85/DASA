import os
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import utils
from env import R2RBatch
from param import args
from utils import read_vocab, write_vocab, build_vocab, Tokenizer, BTokenizer, padding_idx, read_img_features, reduce_dict_list

warnings.filterwarnings("ignore")
from collections import OrderedDict
from pytorch_transformers import BertConfig
from r2rpretrain_class import DicPMActionPreTrain, DicAddActionPreTrain
import torch.nn as nn
from torch.autograd import Variable
import random
from validation import ValidBatch

from torch.utils.tensorboard import SummaryWriter


def create_folders(path):
    """ recursively create folders """
    if not os.path.isdir(path):
        while True:
            try:
                os.makedirs(path)
            except:
                pass
                time.sleep(1)
            else:
                break


data_log = defaultdict(list)
log_dir = 'snap/%s' % args.name
plot_dir = 'snap/%s' % args.name
if args.philly:
    log_dir = os.path.join(os.getenv('PT_OUTPUT_DIR'), 'snap/%s' % args.name)
    plot_dir = log_dir

if not os.path.exists(log_dir):
    if args.philly:
        create_folders(log_dir)
    else:
        os.makedirs(log_dir)
    # os.makedirs(log_dir)

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
PLACE365_FEATURES = 'img_features/ResNet-152-places365.tsv'

if args.features == 'imagenet':
    features = IMAGENET_FEATURES

if args.fast_train:
    name, ext = os.path.splitext(features)
    features = name + "-fast" + ext

feedback_method = args.feedback  # teacher or sample

print(args)


def _sort_batch(obs):
    ''' Extract instructions from a list of observations and sort by descending
        sequence length (to enable PyTorch packing). '''

    seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
    seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
    seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length

    seq_tensor = torch.from_numpy(seq_tensor)
    seq_lengths = torch.from_numpy(seq_lengths)

    # Sort sequences by lengths
    seq_lengths, perm_idx = seq_lengths.sort(0, True)  # True -> descending
    sorted_tensor = seq_tensor[perm_idx]
    mask = (sorted_tensor == padding_idx)  # seq_lengths[0] is the Maximum length
    target_viewIds = torch.Tensor([ob['target_viewId'] for ob in obs])[perm_idx]
    progresses = torch.Tensor([ob['progress'] for ob in obs])[perm_idx]

    return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
           mask.bool().cuda(), \
           list(seq_lengths), list(perm_idx), target_viewIds.long().cuda(), progresses.float().cuda()


def get_input_feat(obs):
    input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
    for i, ob in enumerate(obs):
        input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
    input_a_t = torch.from_numpy(input_a_t).cuda()

    f_t = _feature_variable(obs)  # Image features from obs
    candidate_feat, candidate_leng = _candidate_variable(obs)

    return input_a_t, f_t, candidate_feat, candidate_leng


def _feature_variable(obs):
    ''' Extract precomputed features into variable. '''
    features = np.empty((len(obs), args.views, args.feature_size + args.angle_feat_size), dtype=np.float32)
    for i, ob in enumerate(obs):
        features[i, :, :] = ob['feature']  # Image feat
    return Variable(torch.from_numpy(features), requires_grad=False).cuda()


def _candidate_variable(obs):
    candidate_leng = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
    candidate_feat = np.zeros((len(obs), max(candidate_leng), args.feature_size + args.angle_feat_size),
                              dtype=np.float32)
    # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
    # which is zero in my implementation
    for i, ob in enumerate(obs):
        for j, c in enumerate(ob['candidate']):
            candidate_feat[i, j, :] = c['feature']  # Image feat
    return torch.from_numpy(candidate_feat).cuda(), candidate_leng


def random_word(tokens, seq_len, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of int, tokenized index of sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        if i == 0 or i >= (seq_len - 1):  # [CLS], [SEP] and [PAD]
            output_label.append(-1)
        else:
            prob = random.random()
            # mask token with probability
            ratio = args.word_mask_rate
            if prob < ratio:
                output_label.append(token.cpu().item())
                prob /= ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = tokenizer.vocab['[MASK]']

                # 10% randomly change token to random token
                # elif prob < 0.9:
                #     tokens[i] = random.choice(list(tokenizer.vocab.items()))[1]

                # 10% remain itself

            else:
                output_label.append(-1)

    return tokens, output_label


def mask_words(seq, seq_lengths, tok):
    """
    :param seq: B, LEN
    :param seq_length:
    :return: masked_seq
    :return: output_label
    """
    masked_tokens = []
    masked_labels = []
    for tokens, seq_len in zip(seq, seq_lengths):
        masked_token, masked_label = random_word(tokens, seq_len, tok.tokenizer)
        masked_tokens.append(masked_token)
        masked_labels.append(masked_label)
    masked_tokens = torch.stack(masked_tokens)
    masked_labels = torch.Tensor(masked_labels).long().cuda()
    return masked_tokens, masked_labels

def save(model, optimizer, idx, path):
    ''' Snapshot models '''
    the_dir, _ = os.path.split(path)
    os.makedirs(the_dir, exist_ok=True)
    states = {
            'idx': idx + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
    }
    torch.save(states, path)


def load(path, model, optimizer):
    ''' Loads parameters (but not training state) '''
    states = torch.load(path)

    state = model.state_dict()
    model_keys = set(state.keys())
    load_keys = set(states['state_dict'].keys())
    if model_keys != load_keys:
        print("NOTICE: DIFFERENT KEYS IN THE MODEL")
    state.update(states['state_dict'])
    model.load_state_dict(state)

    if args.loadOptim:
        optimizer.load_state_dict(states['optimizer'])
    return states['idx'] - 1




def train(train_env, tok, n_iters, log_every=100, val_envs={}, aug_env=None, stok=None, press_env=None,
          tasks=['lmask', 'action', 'pm'],
          loss_weights={'lmask': args.lmask_weight, 'action': args.action_weight, 'pm': args.pm_weight}):
    # iterations
    myidx = 0
    start_iter = 0
    idx_step = 1

    # log
    writer = SummaryWriter(log_dir=log_dir)

    # model
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.img_feature_dim = 2048 + args.angle_feat_size
    config.img_feature_type = ""
    config.update_lang_bert = True
    config.update_add_layer = True
    config.vl_layers = args.d_vl_layers  # 4
    config.la_layers = args.d_la_layers  # 9
    config.action_space = 36
    print(config)
    print("=" * 50)

    encoder = DicPMActionPreTrain(config)
    if args.pretrain_model_name is not None:
        print("Using the pretrained lm model from %s" % (args.pretrain_model_name))
        premodel = DicAddActionPreTrain.from_pretrained(args.pretrain_model_name)
        print(premodel.config)
        encoder.bert = premodel.bert
        encoder.drop = nn.Dropout(p=args.d_dropout_ratio)
        encoder.bert.update_lang_bert, encoder.bert.config.update_lang_bert = args.d_transformer_update, args.d_transformer_update
        encoder.bert.update_add_layer, encoder.bert.config.update_add_layer = args.d_update_add_layer, args.d_update_add_layer

    # optimizer
    encoder_optimizer = args.optimizer(encoder.parameters(), lr=args.lr)
    if args.load:
        load(args.load, encoder, encoder_optimizer)

    # best
    best_metric = {}

    if args.fast_train:
        log_every = 40

    # train 61 scans， 14039 instructions
    encoder = encoder.cuda()
    for idx in range(start_iter, start_iter + n_iters, idx_step):
        encoder_optimizer.zero_grad()
        # data preparing

        envs = [train_env]
        if aug_env is not None:
            envs.append(aug_env)

        for env in envs:
            obs = np.array(env.random_start_reset())
            seq, seq_mask, seq_lengths, perm_idx, target_viewIds, progresses = _sort_batch(obs)
            masked_tokens, masked_labels = mask_words(seq, seq_lengths, tok)
            perm_obs = obs[perm_idx]
            input_a_t, f_t, candidate_feat, candidate_leng = get_input_feat(perm_obs)

            # loss
            loss, scores, losses = encoder(seq=masked_tokens, labels=masked_labels, isnext=target_viewIds, f_t_all=f_t,
                                           lang_mask=seq_mask,
                                           progresses=progresses, tasks=tasks, loss_weights=loss_weights)
            train_metrics = {}
            if 'lmask' in tasks:
                prediction_scores = scores['lmask_scores']
                predict_label = prediction_scores.max(dim=-1)[1][masked_labels != -1]
                gt_label = masked_tokens[masked_labels != -1]
                train_acc = (predict_label == gt_label).float().mean()
                train_metrics['lmask_acc'] = train_acc

            if 'action' in tasks:
                action_scores = scores['action_scores']
                action_label = action_scores.max(dim=-1)[1]
                train_action_acc = (action_label == target_viewIds).float().mean()
                train_metrics['action_acc'] = train_action_acc

            if 'pm' in tasks:
                pm_scores = scores['pm_scores']
                pm_mse = losses['pm_loss']
                train_metrics['pm_mse'] = pm_mse
            loss.backward()
            encoder_optimizer.step()

        if idx % log_every == 0:
            metric_str = ",".join(["{}:{:.4f}".format(k, v) for k, v in train_metrics.items()])
            loss_str = ",".join(["{}:{:.4f}".format(k, v) for k, v in losses.items()])
            print("\nPROGRESS: {}%, loss: {:.4f}, {}, {} ".format(round((idx) * 100 / n_iters, 4), loss.cpu().item(),
                                                             metric_str, loss_str))
            writer.add_scalar("loss/pretrain", loss.cpu().item(), idx)

            for k, v in train_metrics.items():
                writer.add_scalar("tasks/{}".format(k), v, idx)
            for k, v in losses.items():
                writer.add_scalar("loss/{}".format(k), v, idx)

        # Run validation
        if idx % (args.val_every * log_every) == 0:
            encoder.eval()
            for env_name, env in val_envs.items():
                env.reset_epoch()
                total_loss = 0
                total_valid_metrics = []
                total_losses = []
                batch_cnt = 0
                masked_labels_cnt = 0
                for obs in env.get_valid_batch():
                    seq, seq_mask, seq_lengths, perm_idx, target_viewIds, progresses = _sort_batch(obs)
                    masked_tokens, masked_labels = mask_words(seq, seq_lengths, tok)
                    input_a_t, f_t, candidate_feat, candidate_leng = get_input_feat(obs)

                    # loss
                    # scanIds = [ob['scan'] for ob in obs]
                    # viewpointIds = [ob['viewpoint'] for ob in obs]
                    # pathIds = [ob['path_id'] for ob in obs]
                    # keys = ['%s_%s_%s' % (scanId, viewpointId, pathId) for scanId, viewpointId, pathId in
                    #         zip(scanIds, viewpointIds, pathIds)]
                    # isnext = torch.tensor([env.target_dict[key]['target_viewId'] for key in keys]).long().cuda()

                    with torch.no_grad():
                        loss, scores, losses = encoder(seq=masked_tokens, labels=masked_labels, isnext=target_viewIds,
                                                       f_t_all=f_t,
                                                       lang_mask=seq_mask,
                                                       progresses=progresses, tasks=tasks, loss_weights=loss_weights)

                        valid_metrics = {}
                        if 'lmask' in tasks:
                            prediction_scores = scores['lmask_scores']
                            predict_label = prediction_scores.max(dim=-1)[1][masked_labels != -1]
                            gt_label = masked_tokens[masked_labels != -1]
                            masked_labels_cnt += torch.sum(masked_labels != -1).item()
                            valid_acc = (predict_label == gt_label).float().sum()
                            valid_metrics['lmask_acc'] = valid_acc

                        if 'action' in tasks:
                            action_scores = scores['action_scores']
                            action_label = action_scores.max(dim=-1)[1]
                            valid_action_acc = (action_label == target_viewIds).float().mean()
                            valid_metrics['action_acc'] = valid_action_acc

                        if 'pm' in tasks:
                            pm_scores = scores['pm_scores']
                            pm_mse = losses['pm_loss']
                            valid_metrics['pm_mse'] = pm_mse

                        total_loss += loss
                        total_valid_metrics.append(valid_metrics)
                        total_losses.append(losses)
                        batch_cnt += 1

                total_loss /= batch_cnt
                total_valid_metrics = reduce_dict_list(total_valid_metrics)
                if 'lmask' in tasks:
                    total_valid_metrics['lmask_acc'] = total_valid_metrics['lmask_acc'] * batch_cnt / masked_labels_cnt
                total_losses = reduce_dict_list(total_losses)

                metric_str = ",".join(["{}:{:.4f}".format(k, v) for k, v in total_valid_metrics.items()])
                print("\t{}: loss: {:.4f}, {}".format(env_name, total_loss.cpu().item(), metric_str))
                writer.add_scalar("loss/%s" % env_name, total_loss.cpu().item(), idx)

                for k, v in total_valid_metrics.items():
                    writer.add_scalar("tasks/{}/{}".format(k, env_name), v, idx)
                for k, v in total_losses.items():
                    writer.add_scalar("loss/{}/{}".format(k, env_name), v, idx)
                for k, v in total_valid_metrics.items():
                    keyname = "{}_{}".format(env_name, k)
                    if k in ['action_acc']: # 'lmask_acc', 'action_acc'
                        if (keyname not in best_metric) or (v > best_metric[keyname]):
                            best_metric[keyname]=v
                            save(model=encoder, optimizer=encoder_optimizer, idx=idx, path=os.path.join("snap", args.name, "best_{}_{}.pth".format(env_name, k)))
                    if k in []: # 'pm_mse'
                        if (keyname not in best_metric) or (v < best_metric[keyname]):
                            best_metric[keyname] = v
                            save(model=encoder, optimizer=encoder_optimizer, idx=idx,
                                 path=os.path.join("snap", args.name, "best_{}_{}.pth".format(env_name, k)))
            encoder.train()


    #     for env_name in best_val:
    #         if best_val[env_name]['update']:
    #             best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
    #             best_val[env_name]['update'] = False
    #             if args.philly: #False
    #                 listner.save(idx, os.path.join(log_dir, "state_dict", "best_%s" % (env_name)))
    #             else:
    #                 listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))
    #
    #     for metric in best_val_sr_sum:
    #         if best_val_sr_sum[metric]['update']:
    #             best_val_sr_sum[metric]['state'] = 'Iter %d %s' % (iter, loss_str)
    #             best_val_sr_sum[metric]['update'] = False
    #             if args.philly:
    #                 listner.save(idx, os.path.join(log_dir, "state_dict", "best_%s" % (metric)))
    #             else:
    #                 listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (metric)))
    #
    #
    #     print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
    #                                          iter, float(iter)/n_iters*100, loss_str)))
    #     if iter % 1000 == 0:
    #         print("BEST RESULT TILL NOW")
    #         for env_name in best_val:
    #             print(env_name, best_val[env_name]['state'])
    #
    #
    #     df = pd.DataFrame(data_log)
    #     df.set_index('iteration')
    #     df_path = '%s/plot_log.csv' % (plot_dir)
    #     write_num = 0
    #     while (write_num < 20):
    #         try:
    #             df.to_csv(df_path)
    #             break
    #         except:
    #             write_num += 1
    #
    #     #if iter % 50000 == 0:
    #     #    if args.philly:
    #     #        listner.save(idx, os.path.join(log_dir, "state_dict", "Iter_%06d" % (iter)))
    #     #    else:
    #     #        listner.save(idx, os.path.join(log_dir, "state_dict", "Iter_%06d" % (iter)))
    #     #    #listner.save(idx, os.path.join(log_dir, "state_dict", "Iter_%06d" % (iter)))
    #
    # if args.philly:
    #     listner.save(idx, os.path.join(log_dir, "state_dict", "LAST_iter%d" % (idx)))
    # else:
    #     listner.save(idx, os.path.join('snap', args.name, "state_dict", "LAST_iter%d" % (idx)))


def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train', 'val_seen', 'val_unseen']), TRAINVAL_VOCAB)


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''
    # args.fast_train = True
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)

    if args.encoderType in ['DicEncoder', 'CEncoder', 'Dic']:  # Dic
        tok = BTokenizer(encoding_length=args.maxInput)
    else:
        tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    feat_dict = read_img_features(features)

    print("tok.tokenizer.vocab['[PAD]']", tok.tokenizer.vocab['[PAD]'])  # 0
    print("tok.tokenizer.vocab['[CLS]']", tok.tokenizer.vocab['[CLS]'])  # 101
    print("tok.tokenizer.vocab['[SEP]']", tok.tokenizer.vocab['[SEP]'])  # 102
    print("tok.tokenizer.vocab['[MASK]']", tok.tokenizer.vocab['[MASK]'])  # 103

    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    if args.aug:
        aug_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['aug_paths_full'], tokenizer=tok)
    else:
        aug_env = None
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    val_env_names = ['val_seen', 'val_unseen']
    val_envs = OrderedDict(
        ((split,
          ValidBatch(feat_dict, batch_size=args.batchSize * 2, splits=[split], tokenizer=tok))
         for split in val_env_names
         )
    )

    tasks = args.tasks.split(",")
    loss_weights = {'lmask': args.lmask_weight,
                    'action': args.action_weight,
                    'pm': args.pm_weight}
    train(train_env, tok, args.iters, val_envs=val_envs, aug_env=aug_env, tasks=tasks, loss_weights=loss_weights)


if __name__ == "__main__":
    print("Pretrain Staring ...")
    if args.train in ['pretrain']:
        train_val()
