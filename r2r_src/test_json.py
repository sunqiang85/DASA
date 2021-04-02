import torch
import os
import time
import json
import numpy as np
from collections import defaultdict
from speaker import Speaker

from utils import read_vocab, write_vocab, build_vocab, Tokenizer, BTokenizer, padding_idx, timeSince, read_img_features
import utils
from env import R2RBatch

# from eval import Evaluation
from eval_plain import Evaluation
from param import args
import pandas as pd
import pdb
import sys
from torch.utils.tensorboard import SummaryWriter
import warnings
import tqdm

warnings.filterwarnings("ignore")

if args.agent_type == 'mcatt':
    from agent_mcatt import Seq2SeqAgent
elif args.agent_type == 'advanced':
    from agent_advanced import Seq2SeqAgent
elif args.agent_type == 'new':
    from agent_new import Seq2SeqAgent
elif args.agent_type == 'kvmem':
    from agent_kvmem import Seq2SeqAgent
elif args.agent_type == 'mutan':
    from agent_mutan import Seq2SeqAgent
elif args.agent_type == 'double':
    from agent_double import Seq2SeqAgent
elif args.agent_type == 'dg':
    from agent_dg import Seq2SeqAgent
elif args.agent_type == 'mt':
    from agent_mt import Seq2SeqAgentk
elif args.agent_type == 'dyrelu':
    from agent_dyrelu import Seq2SeqAgent
elif args.agent_type == 'default':
    from agent import Seq2SeqAgent
else:
    print("wrong agent type")
    sys.exit(-1)


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



def valid(train_env, tok, val_envs={}):

    for env_name, (env, evaluator) in val_envs.items():
        result_file = os.path.join(log_dir, "submit_%s.json" % env_name)
        result = json.load(open(result_file))
        if env_name != '':
            score_summary, _ = evaluator.score(result_file)
            loss_str = "Env name: %s" % env_name
            for metric, val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)




def beam_valid(train_env, tok, val_envs={}):
    listener = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    speaker = Speaker(train_env, listener, tok)
    if args.speaker is not None:
        print("Load the speaker from %s." % args.speaker)
        speaker.load(args.speaker)

    print("Loaded the listener model at iter % d" % listener.load(args.load))

    final_log = ""
    for env_name, (env, evaluator) in val_envs.items():
        listener.logs = defaultdict(list)
        listener.env = env

        listener.beam_search_test(speaker)
        results = listener.results

        def cal_score(x, alpha, avg_speaker, avg_listener):
            speaker_score = sum(x["speaker_scores"]) * alpha
            if avg_speaker:
                speaker_score /= len(x["speaker_scores"])
            # normalizer = sum(math.log(k) for k in x['listener_actions'])
            normalizer = 0.
            listener_score = (sum(x["listener_scores"]) + normalizer) * (1 - alpha)
            if avg_listener:
                listener_score /= len(x["listener_scores"])
            return speaker_score + listener_score

        if args.param_search:
            # Search for the best speaker / listener ratio
            interval = 0.01
            logs = []
            for avg_speaker in [False, True]:
                for avg_listener in [False, True]:
                    for alpha in np.arange(0, 1 + interval, interval):
                        result_for_eval = []
                        for key in results:
                            result_for_eval.append({
                                "instr_id": key,
                                "trajectory": max(results[key]['paths'],
                                                  key=lambda x: cal_score(x, alpha, avg_speaker, avg_listener)
                                                  )['trajectory']
                            })
                        score_summary, _ = evaluator.score(result_for_eval)
                        for metric, val in score_summary.items():
                            if metric in ['success_rate']:
                                print(
                                    "Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
                                    (avg_speaker, avg_listener, alpha, val))
                                logs.append((avg_speaker, avg_listener, alpha, val))
            tmp_result = "Env Name %s\n" % (env_name) + \
                         "Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f\n" % max(
                logs, key=lambda x: x[3])
            print(tmp_result)
            # print("Env Name %s" % (env_name))
            # print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
            #       max(logs, key=lambda x: x[3]))
            final_log += tmp_result
            print()
        else:
            avg_speaker = True
            avg_listener = True
            alpha = args.alpha

            result_for_eval = []
            for key in results:
                result_for_eval.append({
                    "instr_id": key,
                    "trajectory": [(vp, 0, 0) for vp in results[key]['dijk_path']] + \
                                  max(results[key]['paths'],
                                      key=lambda x: cal_score(x, alpha, avg_speaker, avg_listener)
                                      )['trajectory']
                })
            # result_for_eval = utils.add_exploration(result_for_eval)
            score_summary, _ = evaluator.score(result_for_eval)

            if env_name != 'test':
                loss_str = "Env Name: %s" % env_name
                for metric, val in score_summary.items():
                    if metric in ['success_rate']:
                        print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
                              (avg_speaker, avg_listener, alpha, val))
                    loss_str += ",%s: %0.4f " % (metric, val)
                print(loss_str)
            print()

            if args.submit:
                json.dump(
                    result_for_eval,
                    open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
    print(final_log)


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
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    if args.encoderType in ['DicEncoder', 'CEncoder', 'Dic']:
        tok = BTokenizer(encoding_length=args.maxInput)

    feat_dict = read_img_features(features)

    # featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])


    from collections import OrderedDict

    val_env_names = ['val_unseen']

    # val_env_names = ['val_unseen', 'val_seen']
    # if args.submit:
    #     val_env_names.append('test')
    # else:
    #     pass
        # val_env_names.append('train')

    # if not args.beam:
    #    val_env_names.append("train")

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
           Evaluation([split]))
          )
         for split in val_env_names
         )
    )

    # val_envs = OrderedDict(
    #     ((split,
    #       (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
    #        Evaluation([split], featurized_scans, tok))
    #       )
    #      for split in val_env_names
    #      )
    # )

    if args.train == 'validlistener':
        if args.beam:
            beam_valid(None, tok, val_envs=val_envs)
        else:
            valid(None, tok, val_envs=val_envs)
    else:
        assert False


def valid_speaker(tok, val_envs):

    listner = Seq2SeqAgent(None, "", tok, args.maxAction)
    speaker = Speaker(None, listner, tok)
    speaker.load(args.load)

    for env_name, (env, evaluator) in val_envs.items():
        if env_name == 'train':
            continue
        print("............ Evaluating %s ............." % env_name)
        speaker.env = env
        path2inst, loss, word_accu, sent_accu = speaker.valid(wrapper=tqdm.tqdm)
        path_id = next(iter(path2inst.keys()))
        print("Inference: ", tok.decode_sentence(path2inst[path_id]))
        print("GT: ", evaluator.gt[path_id]['instructions'])
        pathXinst = list(path2inst.items())
        name2score = evaluator.lang_eval(pathXinst, no_metrics={'METEOR'})
        score_string = " "
        for score_name, score in name2score.items():
            score_string += "%s_%s: %0.4f " % (env_name, score_name, score)
        print("For env %s" % env_name)
        print(score_string)
        print("Average Length %0.4f" % utils.average_length(path2inst))





if __name__ == "__main__":
    if args.train in ['validlistener']:
        train_val()
    else:
        assert False
