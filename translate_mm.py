#!/usr/bin/env python

from __future__ import division, unicode_literals
import os
import argparse
import math
import codecs
import torch

from itertools import count

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import opts
import tables

parser = argparse.ArgumentParser(
    description='translate_mm.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.translate_opts(parser)
opts.translate_mm_opts(parser)

opt = parser.parse_args()


def _report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total / words_total)))


def _report_bleu():
    import subprocess
    print()
    res = subprocess.check_output(
        "perl tools/multi-bleu.perl %s < %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(">> " + res.strip())


def _report_rouge():
    import subprocess
    res = subprocess.check_output(
        "python tools/test_rouge.py -r %s -c %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(res.strip())


def main():
    dummy_parser = argparse.ArgumentParser(description='train_mm.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu >= -1

    if opt.cuda:
        torch.cuda.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
    # loading checkpoint just to find multimodal model type
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)
    opt.multimodal_model_type = checkpoint['opt'].multimodal_model_type
    del checkpoint

    if opt.batch_size > 1:
        print("Batch size > 1 not implemented! Falling back to batch_size = 1 ...")
        opt.batch_size = 1

    # load test image features
    test_file = tables.open_file(opt.path_to_test_img_feats, mode='r')
    if opt.multimodal_model_type in ['imgd', 'imge', 'imgw']:
        test_img_feats = test_file.root.global_feats[:]
    elif opt.multimodal_model_type in ['src+img']:
        test_img_feats = test_file.root.local_feats[:]
    else:
        raise Exception("Model type not implemented: %s"%opt.multimodal_model_type)
    test_file.close()

    # Load the model.
    fields, model, model_opt = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)
    #opt.multimodal_model_type = checkpoint['opt'].multimodal_model_type

    model = model.cuda()
    
    # File to write sentences to.
    out_file = codecs.open(opt.output, 'w', 'utf-8')

    # Test data
    data = onmt.io.build_dataset(fields, opt.data_type,
                                 opt.src, opt.tgt,
                                 src_dir=opt.src_dir,
                                 sample_rate=opt.sample_rate,
                                 window_size=opt.window_size,
                                 window_stride=opt.window_stride,
                                 window=opt.window,
                                 use_filter_pred=False)

    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
    data_iter = onmt.io.OrderedIterator(
        dataset=data, device=device,
        batch_size=opt.batch_size, train=False, sort=False,
        sort_within_batch=True, shuffle=False)
    # Translator
    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha, opt.beta)
    translator = onmt.translate.TranslatorMultimodal(model, fields,
                                           beam_size=opt.beam_size,
                                           n_best=opt.n_best,
                                           global_scorer=scorer,
                                           max_length=opt.max_length,
                                           copy_attn=model_opt.copy_attn,
                                           cuda=True,
                                           beam_trace=opt.dump_beam != "",
                                           min_length=opt.min_length,
                                           test_img_feats=test_img_feats,
                                           multimodal_model_type=opt.multimodal_model_type)
    builder = onmt.translate.TranslationBuilder(
        data, translator.fields,
        opt.n_best, opt.replace_unk, opt.tgt)

    # Statistics
    counter = count(1)
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0

    for sent_idx, batch in enumerate(data_iter):
        batch_data = translator.translate_batch(batch, data, sent_idx)
        translations = builder.from_batch(batch_data)

        for trans in translations:
            pred_score_total += trans.pred_scores[0]
            pred_words_total += len(trans.pred_sents[0])
            if opt.tgt:
                gold_score_total += trans.gold_score
                gold_words_total += len(trans.gold_sent)

            n_best_preds = [" ".join(pred)
                            for pred in trans.pred_sents[:opt.n_best]]
            out_file.write('\n'.join(n_best_preds))
            out_file.write('\n')
            out_file.flush()

            if opt.verbose:
                sent_number = next(counter)
                output = trans.log(sent_number)
                os.write(1, output.encode('utf-8'))

    _report_score('PRED', pred_score_total, pred_words_total)
    
    if opt.tgt:
        _report_score('GOLD', gold_score_total, gold_words_total)
        if opt.report_bleu:
            _report_bleu()
        if opt.report_rouge:
            _report_rouge()

    if opt.dump_beam:
        import json
        json.dump(translator.beam_accum,
                  codecs.open(opt.dump_beam, 'w', 'utf-8'))


if __name__ == "__main__":
    main()
