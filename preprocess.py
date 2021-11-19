# -*- coding: utf-8 -*-
import argparse
import codecs
from data.Tree import python2tree, python_tokenize, traverse_python_tree, split_tree, merge_tree, initVocabulary
from constants.constants import UNK_WORD, EOS_WORD


# from .Dict import Dict
def get_opt():
    parser = argparse.ArgumentParser(description='preprocess.py')
    parser.add_argument('-data_name', help="Data name")
    parser.add_argument("-train_src", required=True,
                        help="Path to the training source data")
    parser.add_argument("-train_tgt", required=True,
                        help="Path to the training target data")
    parser.add_argument("-train_xe_src", required=True,
                        help="Path to the pre-training source data")
    parser.add_argument("-train_xe_tgt", required=True,
                        help="Path to the pre-training target data")
    parser.add_argument("-train_pg_src", required=False,
                        help="Path to the bandit training source data")
    parser.add_argument("-train_pg_tgt", required=False,
                        help="Path to the bandit training target data")
    parser.add_argument("-valid_src", required=True,
                        help="Path to the validation source data")
    parser.add_argument("-valid_tgt", required=True,
                        help="Path to the validation target data")
    parser.add_argument("-test_src", required=True,
                        help="Path to the test source data")
    parser.add_argument("-test_tgt", required=True,
                        help="Path to the test target data")
    parser.add_argument('-save_data', required=True,
                        help="Output file for the prepared data")
    parser.add_argument('-src_vocab_size', type=int,
                        default=50000, help="Size of the source vocabulary")
    parser.add_argument('-tgt_vocab_size', type=int,
                        default=50000, help="Size of the target vocabulary")
    parser.add_argument('-src_seq_length', type=int,
                        default=150, help="Maximum source sequence length")
    parser.add_argument('-tgt_seq_length', type=int, default=50,
                        help="Maximum target sequence length to keep.")

    # parser.add_argument('-shuffle',    type=int, default=1,
    #                     help="Shuffle data")
    parser.add_argument('-seed',       type=int,
                        default=3435, help="Random seed")
    parser.add_argument('-lower', action='store_true', help='lowercase data')
    parser.add_argument('-report_every', type=int, default=1000,
                        help="Report status every this many sentences")

    opt = parser.parse_args()
    return opt


def makeData(which, srcFile, tgtFile, srcDicts, tgtDicts):
    src, tgt, trees = [], [], []
    code_sentences, comment_sentences = [], []
    sizes = []
    ignored, exceps = 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = codecs.open(srcFile, 'r', 'utf-8', errors='ignore')
    tgtF = codecs.open(tgtFile, 'r', 'utf-8', errors='ignore')

    while True:
        sline = srcF.readline().strip()
        tline = tgtF.readline().strip()

        # source or target does not have same number of lines
        if sline == '' or tline == '':
            print('WARNING: src and tgt do not have the same # of sentences')
            break

        if opt.data_name == 'github-python':
            srcLine = python_tokenize(sline.replace(
                ' DCNL DCSP ', '').replace(' DCNL ', '').replace(' DCSP ', ''))
            tgtLine = tline.replace(' DCNL DCSP ', '').replace(
                ' DCNL ', '').replace(' DCSP ', '').split()
            sline = sline.replace(' DCNL DCSP ', '\n\t').replace(' DCNL  DCSP ', '\n\t').replace(
                ' DCNL   DCSP ', '\n\t').replace(' DCNL ', '\n').replace(' DCSP ', '\t')
            code_sentences.append(sline.replace(' DCNL DCSP ', '').replace(
                ' DCNL ', '').replace(' DCSP ', '').split())
            code_sentences.append(srcLine)
            comment_sentences.append(tgtLine)
        else:
            srcLine = sline.split()
            tgtLine = tline.split()

        # len(srcLine) <= opt.src_seq_length and
        if len(srcLine) <= opt.src_seq_length and len(tgtLine) <= opt.tgt_seq_length:
            try:
                # Given a line of source code, build a tree and save it as dictionary

                atok, tree = python2tree(sline)
                tree_json = traverse_python_tree(atok, tree)

                tree_json = split_tree(tree_json, len(tree_json))
                tree_json = merge_tree(tree_json)
                # if len(tree_json) < opt.src_seq_length:
                trees += [tree_json]

                src += [srcDicts.convert_to_index(srcLine, UNK_WORD)]
                tgt += [tgtDicts.convert_to_index(tgtLine,
                                                  UNK_WORD, eos_word=EOS_WORD)]
                sizes += [len(src)]
            except Exception as e:
                print('Exception: ', e)
                print(sline)
                exceps += 1
        else:
            print('Too long')
            ignored += 1

    srcF.close()
    tgtF.close()

    # print('... sorting sentences by size')
    # _, perm = torch.sort(torch.Tensor(sizes))
    # src = [src[idx] for idx in perm]
    # tgt = [tgt[idx] for idx in perm]
    # trees = [trees[idx] for idx in perm]
    print(('Prepared %d sentences ' +
          '(%d ignored due to length == 0 or src len > %d or tgt len > %d)') %
          (len(src), ignored, opt.src_seq_length, opt.tgt_seq_length))
    print(('Prepared %d sentences ' + '(%d ignored due to Exception)') %
          (len(src), exceps))
    return src, tgt, trees, code_sentences, comment_sentences


def makeDataGeneral(which, src_path, tgt_path, dicts):
    print('Preparing ' + which + '...')
    res = {}
    res['src'], res['tgt'], res['trees'], code_sentences, comment_sentences = makeData(
        which, src_path, tgt_path, dicts['src'], dicts['tgt'])
    return res, code_sentences, comment_sentences


def main():

    dicts = {}
    dicts['src'] = initVocabulary(
        opt, 'code', opt.train_src, opt.src_vocab_size)
    dicts['tgt'] = initVocabulary(
        opt, 'comment', opt.train_tgt, opt.tgt_vocab_size)

    print("code vocab size",dicts['src'].size)
    print("comment vocab size",dicts['tgt'].size)
    # print(opt.save_data)

    dicts['src'].write_to_file(opt.save_data + '.code.dict')
    dicts['tgt'].write_to_file(opt.save_data + '.comment.dict')

    save_data = {}
    save_data['dicts'] = dicts
    save_data['train_xe'], train_xe_code_sentences, train_xe_comment_sentences = makeDataGeneral(
        'train_xe', opt.train_xe_src, opt.train_xe_tgt, dicts)
    save_data['train_pg'], train_pg_code_sentences, train_pg_comment_sentences = makeDataGeneral(
        'train_pg', opt.train_pg_src, opt.train_pg_tgt, dicts)
    save_data['valid'], valid_code_sentences, valid_comment_sentences = makeDataGeneral(
        'valid', opt.valid_src, opt.valid_tgt, dicts)
    save_data['test'], test_code_sentences, test_comment_sentences = makeDataGeneral(
        'test', opt.test_src, opt.test_tgt, dicts)

    print("Saving data to \"" + opt.save_data + ".train.pt\"...")

    # word2vec dump
    # print('code_sentences: ', train_xe_code_sentences[0])
    # print('comment_sentences: ', train_xe_comment_sentences[0])
    # code_w2v_model = gensim.models.Word2Vec(train_xe_code_sentences, size=512, window=5, min_count=5, workers=16)
    # code_w2v_model.save(opt.save_data + '.train_xe.code.gz')
    # comment_w2v_model = gensim.models.Word2Vec(train_xe_comment_sentences, size=512, window=5, min_count=5, workers=16)
    # comment_w2v_model.save(opt.save_data + '.train_xe.comment.gz')


if __name__ == "__main__":
    global opt
    opt = get_opt()
    main()
    # takes about 40min
