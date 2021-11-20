from __future__ import division
import argparse
from constants.constants import UNK_WORD
from data.Tree import Tree, json2tree_binary
from loss.Loss import weighted_mse
from model.CodeEncoder import CodeEncoder
from model.HybridDecoder import HybridDecoder
from model.TextEncoder import TextEncoder
from model.Hybrid2Seq import Hybrid2Seq
from model.Generator import Generator
import os
import sys
import datetime
import numpy as np
import os.path
import random
import time
import pickle
import tensorflow as tf

def get_opt():
    parser = argparse.ArgumentParser(description='a2c-train.py')
    # Data options
    parser.add_argument('-data', required=False, help='Path to the *train.pickle file from preprocess.py')
    parser.add_argument('-train_portion', type=float, default=0.6)
    parser.add_argument('-dev_portion', type=float, default=0.2)

    # Optimization options
    parser.add_argument('-batch_size', type=int, default=64, help='Maximum batch size')

    parser.add_argument("-end_epoch", type=int, default=50, help="Epoch to stop training.")
    parser.add_argument("-start_epoch", type=int, default=1, help="Epoch to start training.")

    parser.add_argument('-param_init', type=float, default=0.1, help="""Parameters are initialized over uniform distribution with support (-param_init, param_init). Use 0 to not use initialization""")
    parser.add_argument('-optim', default='adam', help="Optimization method. [sgd|adagrad|adadelta|adam]")
    parser.add_argument("-lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('-max_grad_norm', type=float, default=5, help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('-dropout', type=float, default=0.3, help='Dropout probability; applied between LSTM stacks.')

    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=5,
                        help="""Start decaying every epoch after and including this epoch""")
    # GPU
    # Critic
    parser.add_argument("-start_reinforce", type=int, default=None, help="""Epoch to start reinforcement training. Use -1 to start immediately.""")
    parser.add_argument("-critic_pretrain_epochs", type=int, default=0, help="Number of epochs to pretrain critic (actor fixed).")
    parser.add_argument("-reinforce_lr", type=float, default=1e-4, help="""Learning rate for reinforcement training.""")

    # Evaluation
    parser.add_argument("-eval", action="store_true", help="Evaluate model only")
    parser.add_argument("-eval_one", action="store_true", help="Evaluate only one sample.")
    parser.add_argument("-eval_sample", action="store_true", default=False, help="Eval by sampling")
    parser.add_argument("-max_predict_length", type=int, default=50,help="Maximum length of predictions.")

    # Reward shaping
    parser.add_argument("-pert_func", type=str, default=None, help="Reward-shaping function.")
    parser.add_argument("-pert_param", type=float, default=None,help="Reward-shaping parameter.")

    # Others
    parser.add_argument("-no_update", action="store_true", default=False, help="No update round. Use to evaluate model samples.")
    parser.add_argument("-sup_train_on_bandit", action="store_true", default=False, help="Supervised learning update round.")

    parser.add_argument("-var_length", action="store_true", help="Evaluate model only")
    parser.add_argument('-var_type', default='code', help="Type of var.")

    opt = parser.parse_args()
    opt.iteration = 0
    return opt

def get_data_trees(trees):
    data_trees = []
    for t_json in trees:
        for k, node in t_json.iteritems():
            if node['parent'] == None:
                root_idx = k
        tree = json2tree_binary(t_json, Tree(), root_idx)
        data_trees.append(tree)

    return np.array(data_trees)

def get_data_leafs(trees, srcDicts):
    leafs = []
    for tree in trees:
        leaf_contents = tree.leaf_contents()

        leafs.append(srcDicts.convertToIdx(leaf_contents, UNK_WORD))
    return np.array(leafs)

def sort_test(dataset):
    if opt.var_type == 'code':
        length = [l.size(0) for l in dataset["test"]['src']]
    elif opt.var_type == 'comment':
        length = [l.size(0) for l in dataset["test"]['tgt']]

    length, code, comment, trees = zip(*sorted(zip(length, dataset["test"]['src'], dataset["test"]['tgt'], dataset["test"]['trees']), key=lambda x: x[0]))

    return length, code, comment, trees

def load_data(filename, batch_size):
    dataset = pickle.load(open(filename,'rb'))

    dicts = dataset['dicts']

    dataset["train_xe"]['trees'] = get_data_trees(dataset["train_xe"]['trees'])
    dataset["train_pg"]['trees'] = get_data_trees(dataset["train_pg"]['trees'])
    dataset["valid"]['trees'] = get_data_trees(dataset["valid"]['trees'])
    dataset["test"]['trees'] = get_data_trees(dataset["test"]['trees'])

    dataset["train_xe"]['leafs'] = get_data_leafs(dataset["train_xe"]['trees'], dicts['src'])
    dataset["train_pg"]['leafs'] = get_data_leafs(dataset["train_pg"]['trees'], dicts['src'])
    dataset["valid"]['leafs'] = get_data_leafs(dataset["valid"]['trees'], dicts['src'])
    dataset["test"]['leafs'] = get_data_leafs(dataset["test"]['trees'], dicts['src'])

    supervised_data_gen = Generator(dataset["train_xe"], batch_size)
    rl_data_gen = Generator(dataset["train_pg"], batch_size)
    valid_data_gen = Generator(dataset["valid"], batch_size)
    test_data_gen = Generator(dataset["test"], batch_size)
    vis_data_gen = Generator(dataset["test"], 1) # batch_size set to 1 for case study

    print(" * vocabulary size. source = %d; target = %d" % (dicts["src"].size(), dicts["tgt"].size()))
    print(" * number of XENT training sentences. %d" % len(dataset["train_xe"]["src"]))
    print(" * number of PG training sentences. %d" % len(dataset["train_pg"]["src"]))
    print(" * maximum batch size. %d" % opt.batch_size)

    return dicts, supervised_data_gen, rl_data_gen, valid_data_gen, test_data_gen, vis_data_gen

def init(model):
    for p in model.parameters():
        p.data.uniform_(-opt.param_init, opt.param_init)

def create_optim(model):
    optim = lib.Optim(
        model.parameters(), opt.optim, opt.lr, opt.max_grad_norm,
        lr_decay=opt.learning_rate_decay, start_decay_at=opt.start_decay_at
    )
    return optim

def create_model(dicts):
    
    # code_encoder = lib.TreeEncoder(opt, dicts["src"])
    # text_encoder = lib.Encoder(opt, dicts["src"])
    # decoder = lib.HybridDecoder(opt, dicts["tgt"])

    

    # Use memory efficient generator when output size is large and
    # max_generator_batches is smaller than batch_size.
    
    
    model = Hybrid2Seq(dicts, dicts['src'].size(),dicts['tgt'].size())
    

    return model



def main():
    print("Start...")
    global opt
    opt = get_opt()


   

    


    dicts, supervised_data_gen, rl_data_gen, valid_data_gen, test_data_gen, vis_data_gen = load_data(opt.data,opt.batch_size)

    print("Building model...")

    use_critic = opt.start_reinforce is not None
    print("use_critic: ", use_critic)

    model = create_model(Hybrid2Seq, dicts, dicts["tgt"].size())

    

    

    

    

    # Metrics.
    metrics = {}
    metrics["xent_loss"] = tf.nn.weighted_cross_entropy_with_logits
    metrics["critic_loss"] = weighted_mse
    metrics["sent_reward"] = lib.Reward.sentence_bleu
    metrics["corp_reward"] = lib.Reward.corpus_bleu
    if opt.pert_func is not None:
        opt.pert_func = lib.PertFunction(opt.pert_func, opt.pert_param)

    print("opt.eval: ", opt.eval)
    print("opt.eval_sample: ", opt.eval_sample)

    # Evaluate model on heldout dataset.

    
    # xent_trainer = Trainer(model, supervised_data, valid_data, metrics, dicts, optim, opt)
    
    start_time = time.time()
    # Supervised training.
    print("supervised training..")
    print("start_epoch: ", opt.start_epoch)

    # xent_trainer.train(opt.start_epoch, opt.start_reinforce - 1, start_time)
    # # Create critic here to not affect random seed.
    # critic = create_model(dicts)
    # # Pretrain critic.
    # print("pretrain critic...")
    # if opt.critic_pretrain_epochs > 0:
    #     reinforce_trainer = lib.ReinforceTrainer(model, critic, supervised_data, test_data, metrics, dicts, optim, critic_optim, opt)
    #     reinforce_trainer.train(opt.start_reinforce, opt.start_reinforce + opt.critic_pretrain_epochs - 1, True, start_time)
    # # Reinforce training.
    # print("reinforce training...")
    # reinforce_trainer = lib.ReinforceTrainer(model, critic, rl_data, test_data, metrics, dicts, optim, critic_optim, opt)
    # reinforce_trainer.train(opt.start_reinforce + opt.critic_pretrain_epochs, opt.end_epoch, False, start_time)

    # Supervised training only.
    

if __name__ == '__main__':
    main()
