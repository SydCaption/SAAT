import sys
import os
import json

import numpy as np
from collections import OrderedDict


sys.path.append('coco-caption')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
#from pyciderevalcap.ciderD.ciderD import CiderD

from six.moves import cPickle
from pdb import set_trace

def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every [lr_update] epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if isinstance(score, list):
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


def load_gt_refs(cocofmt_file):
    d = json.load(open(cocofmt_file))
    out = {}
    for i in d['annotations']:
        out.setdefault(i['image_id'], []).append(i['caption'])
    return out


def compute_score(gt_refs, predictions, scorer):
    # use with standard package https://github.com/tylin/coco-caption
    # hypo = {p['image_id']: [p['caption']] for p in predictions}

    # use with Cider provided by https://github.com/ruotianluo/cider
    hypo = [{'image_id': p['image_id'], 'caption':[p['caption']]}
            for p in predictions]
    set_trace()
    # standard package requires ref and hypo have same keys, i.e., ref.keys()
    # == hypo.keys()
    ref = {p['image_id']: gt_refs[p['image_id']] for p in predictions}

    score, scores = scorer.compute_score(ref, hypo)

    return score, scores


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                #set_trace()
                txt = txt + ix_to_word[ix.item()].decode()
            else:
                break
        out.append(txt)
    #set_trace()
    return out

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def compute_avglogp(seq, logseq, eos_token=0):
    seq = seq.cpu().numpy()
    logseq = logseq.cpu().numpy()
    
    N, D = seq.shape
    out_avglogp = []
    for i in range(N):
        avglogp = []
        for j in range(D):
            ix = seq[i, j]
            avglogp.append(logseq[i, j])
            if ix == eos_token:
                break
        avg = 0 if len(avglogp) == 0 else sum(avglogp)/float(len(avglogp))
        out_avglogp.append(avg)
    return out_avglogp

def language_eval(gold_file, pred_file):
    #set_trace()
    # save the current stdout
    #temp = sys.stdout
    #sys.stdout = open(os.devnull, 'w')

    coco = COCO(gold_file)
    cocoRes = coco.loadRes(pred_file)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = round(score, 5)

    # restore the previous stdout
    #sys.stdout = temp
    return out


def array_to_str(arr, use_eos=0):
    out = ''
    for i in range(len(arr)):
        if use_eos == 0 and arr[i] == 0:
            break
        
        # skip the <bos> token    
        if arr[i] == 1:
            continue
            
        out += str(arr[i]) + ' '
        
        # return if encouters the <eos> token
        # this will also guarantees that the first <eos> will be rewarded
        if arr[i] == 0:
            break
            
    return out.strip()


def get_self_critical_reward2(model_res, greedy_res, gt_refs, scorer):

    model_score, model_scores = compute_score(model_res, gt_refs, scorer)
    greedy_score, greedy_scores = compute_score(greedy_res, gt_refs, scorer)
    scores = model_scores - greedy_scores

    m_score = np.mean(model_scores)
    g_score = np.mean(greedy_scores)

    #rewards = np.repeat(scores[:, np.newaxis], model_res.shape[1], 1)

    return m_score, g_score


def get_self_critical_reward(
        model_res,
        greedy_res,
        data_gts,
        bcmr_scorer,
        expand_feat=0,
        seq_per_img=20,
        use_eos=0):
    
    batch_size = model_res.size(0)

    model_res = model_res.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()
    
    res = OrderedDict()
    for i in range(batch_size):
        res[i] = [array_to_str(model_res[i], use_eos)]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i], use_eos)]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j], use_eos)
                  for j in range(len(data_gts[i]))]
    
    #_, scores = Bleu(4).compute_score(gts, res)
    #scores = np.array(scores[3])
    if isinstance(bcmr_scorer, CiderD):    
        res = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
        
    if expand_feat == 1:
        gts = {i: gts[(i % batch_size) // seq_per_img]
               for i in range(2 * batch_size)}
    else:
        gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}
    #set_trace()
    score, scores = bcmr_scorer.compute_score(gts, res)
    
    # if bleu, only use bleu_4
    if isinstance(bcmr_scorer, Bleu):
        score = score[-1]
        scores = scores[-1]
    
    # happens for BLeu and METEOR
    if type(scores) == list:
        scores = np.array(scores)
    
    m_score = np.mean(scores[:batch_size])
    g_score = np.mean(scores[batch_size:])

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], model_res.shape[1], 1)

    return rewards, m_score, g_score


def get_cst_reward(
        model_res,
        data_gts,
        bcmr_scorer,
        bcmrscores=None,
        expand_feat=0,
        seq_per_img=20,
        scb_captions=20,
        scb_baseline=1,
        use_eos=0,
        use_mixer=0):
    
    """
    Arguments:
        bcmrscores: precomputed scores of GT sequences
        scb_baseline: 1 - use GT to compute baseline, 
                      2 - use MS to compute baseline
    """
    
    if bcmrscores is None or use_mixer == 1:
        batch_size = model_res.size(0)

        model_res = model_res.cpu().numpy()
        
        res = OrderedDict()
        for i in range(batch_size):
            res[i] = [array_to_str(model_res[i], use_eos)]

        gts = OrderedDict()
        for i in range(len(data_gts)):
            gts[i] = [array_to_str(data_gts[i][j], use_eos)
                      for j in range(len(data_gts[i]))]

        if isinstance(bcmr_scorer, CiderD):    
            res = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]
        
        if expand_feat == 1:
            gts = {i: gts[(i % batch_size) // seq_per_img]
                   for i in range(batch_size)}
        else:
            gts = {i: gts[i % batch_size] for i in range(batch_size)}
        
        _, scores = bcmr_scorer.compute_score(gts, res)
            
        # if bleu, only use bleu_4
        if isinstance(bcmr_scorer, Bleu):
            scores = scores[-1]
    
        # happens for BLeu and METEOR
        if type(scores) == list:
            scores = np.array(scores)

        scores = scores.reshape(-1, seq_per_img)
            
    elif bcmrscores is not None and use_mixer == 0:
        # use pre-computed scores only when mixer is not used
        scores = bcmrscores.copy()
    else:
        raise ValueError('bcmrscores is not set!')
        
    if scb_captions > 0:
        
        sorted_scores = np.sort(scores, axis=1)
        
        if scb_baseline == 1:
            # compute baseline from GT scores
            sorted_bcmrscores = np.sort(bcmrscores, axis=1)
            m_score = np.mean(scores)
            b_score = np.mean(bcmrscores)
        elif scb_baseline == 2:
            # compute baseline from sampled scores
            m_score = np.mean(sorted_scores)
            b_score = np.mean(sorted_scores[:,:scb_captions])
        else:
            raise ValueError('unknown scb_baseline!')
        
        for ii in range(scores.shape[0]):
            if scb_baseline == 1:
                b = np.mean(sorted_bcmrscores[ii,:scb_captions])
            elif scb_baseline == 2:
                b = np.mean(sorted_scores[ii,:scb_captions])
            else:
                b = 0
            scores[ii] = scores[ii] - b
                
    else:
        m_score = np.mean(scores)
        b_score = 0
    
    scores = scores.reshape(-1)
    rewards = np.repeat(scores[:, np.newaxis], model_res.shape[1], 1)
    
    return rewards, m_score, b_score
