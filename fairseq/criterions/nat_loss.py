# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import pdb
import torch.nn.functional as F
import torch
from torch import Tensor
from collections import Counter
import random
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import numpy as np

@register_criterion("ctc_loss")
class LabelSmoothedDualImitationCriterion(FairseqCriterion):
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument(
            '--use-ngram',
            action="store_true")
        parser.add_argument(
            '--ngram-size',
            type=int,
            default=2)
        parser.add_argument(
            '--use-hung',
            action="store_true")
        parser.add_argument(
            '--use-word',
            action="store_true")
        parser.add_argument(
            '--sctc-loss',
            action="store_true")
        # fmt: on

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def compute_ctc_1gram_loss(self, log_probs, targets):

        batch_size, length_ctc, vocab_size = log_probs.size()
        _, length_tgt = targets.size()
        probs = torch.exp(log_probs)
        bow = probs[:,0,] + torch.sum(probs[:,1:,:] * (1 - probs[:,:-1,:]), dim = 1)
        bow[:,4] = 0
        ref_bow = torch.zeros(batch_size, vocab_size).cuda(probs.get_device())
        ones = torch.ones(batch_size, vocab_size).cuda(probs.get_device())
        ref_bow.scatter_add_(-1, targets, ones).detach()
        expected_length = torch.sum(bow).div(batch_size)
        loss = torch.mean(torch.norm(bow-ref_bow,p=1,dim=-1))/ (length_tgt + expected_length)
        return loss

    def compute_ctc_bigram_loss(self, log_probs, targets):

        batch_size, length_ctc, vocab_size = log_probs.size()
        _, length_tgt = targets.size()
        probs = torch.exp(log_probs)
        targets = targets.tolist()
        probs_blank = probs[:,:,4]
        length = probs[:,1:,:] * (1 - probs[:,:-1,:])
        length[:,:,4] = 0
        expected_length = torch.sum(length).div(batch_size)

        logprobs_blank = log_probs[:,:,4]
        cumsum_blank = torch.cumsum(logprobs_blank, dim = -1)
        cumsum_blank_A = cumsum_blank.view(batch_size, 1, length_ctc).expand(-1, length_ctc, -1)
        cumsum_blank_B = cumsum_blank.view(batch_size, length_ctc, 1).expand(-1, -1, length_ctc)
        cumsum_blank_sub = cumsum_blank_A - cumsum_blank_B
        cumsum_blank_sub = torch.cat((torch.zeros(batch_size, length_ctc,1).cuda(cumsum_blank_sub.get_device()), cumsum_blank_sub[:,:,:-1]), dim = -1)
        tri_mask = torch.tril(utils.fill_with_neg_inf(torch.zeros([batch_size, length_ctc, length_ctc]).cuda(cumsum_blank_sub.get_device())), 0)
        cumsum_blank_sub = cumsum_blank_sub + tri_mask
        blank_matrix = torch.exp(cumsum_blank_sub)

        gram_1 = []
        gram_2 = []
        gram_count = []
        rep_gram_pos = []
        num_grams = length_tgt - 1
        for i in range(batch_size):
            two_grams = Counter()
            gram_1.append([])
            gram_2.append([])
            gram_count.append([])
            for j in range(num_grams):
                two_grams[(targets[i][j], targets[i][j+1])] += 1
            j = 0
            for two_gram in two_grams:
                gram_1[-1].append(two_gram[0])
                gram_2[-1].append(two_gram[1])
                gram_count[-1].append(two_grams[two_gram])
                if two_gram[0] == two_gram[1]:
                    rep_gram_pos.append((i, j))
                j += 1
            while len(gram_count[-1]) < num_grams:
                gram_1[-1].append(1)
                gram_2[-1].append(1)
                gram_count[-1].append(0)
        gram_1 = torch.LongTensor(gram_1).cuda(blank_matrix.get_device())
        gram_2 = torch.LongTensor(gram_2).cuda(blank_matrix.get_device())
        gram_count = torch.Tensor(gram_count).cuda(blank_matrix.get_device()).view(batch_size, num_grams,1)
        gram_1_probs = torch.gather(probs, -1, gram_1.view(batch_size,1,num_grams).expand(batch_size,length_ctc,num_grams)).transpose(1,2).view(batch_size, num_grams, length_ctc, 1)
        gram_2_probs = torch.gather(probs, -1, gram_2.view(batch_size,1,num_grams).expand(batch_size,length_ctc,num_grams)).transpose(1,2).view(batch_size, num_grams, 1, length_ctc)
        probs_matrix = torch.matmul(gram_1_probs, gram_2_probs)
        bag_grams = blank_matrix.view(batch_size, 1, length_ctc, length_ctc) * probs_matrix
        bag_grams = torch.sum(bag_grams.view(batch_size, num_grams, -1), dim = -1).view(batch_size, num_grams,1)
        if len(rep_gram_pos) > 0:
            for pos in rep_gram_pos:
                i, j = pos
                gram_id = gram_1[i, j]
                gram_prob = probs[i, :, gram_id]
                rep_gram_prob = torch.sum(gram_prob[1:] * gram_prob[:-1])
                bag_grams[i, j, 0] = bag_grams[i, j, 0] - rep_gram_prob
        match_gram = torch.min(torch.cat([bag_grams,gram_count],dim = -1), dim = -1)[0]
        match_gram = torch.sum(match_gram).div(batch_size)
        loss = (- 2 * match_gram).div(length_tgt + expected_length - 2)
        return loss

    def compute_sctc_ngram_loss(self, log_probs, targets, n):

        batch_size, length_ctc, vocab_size = log_probs.size()
        _, length_tgt = targets.size()
        probs = torch.exp(log_probs)
        targets = targets.tolist()
        probs_blank = probs[:,:,4]
        length = probs[:,1:,:] * (1 - probs[:,:-1,:])
        length[:,:,4] = 0
        expected_length = torch.sum(length).div(batch_size) - n + 1

        logprobs_blank = log_probs[:,:,4]
        cumsum_blank = torch.cumsum(logprobs_blank, dim = -1)
        cumsum_blank_A = cumsum_blank.view(batch_size, 1, length_ctc).expand(-1, length_ctc, -1)
        cumsum_blank_B = cumsum_blank.view(batch_size, length_ctc, 1).expand(-1, -1, length_ctc)
        cumsum_blank_sub = cumsum_blank_A - cumsum_blank_B
        cumsum_blank_sub = torch.cat((torch.zeros(batch_size, length_ctc,1).cuda(cumsum_blank_sub.get_device()), cumsum_blank_sub[:,:,:-1]), dim = -1)
        tri_mask = torch.tril(utils.fill_with_neg_inf(torch.zeros([batch_size, length_ctc, length_ctc]).cuda(cumsum_blank_sub.get_device())), 0)
        cumsum_blank_sub = cumsum_blank_sub + tri_mask
        blank_matrix = torch.exp(cumsum_blank_sub)

        gram_idx = []
        gram_count = []
        rep_gram_pos = []
        num_grams = length_tgt - n + 1
        for i in range(batch_size):
            ngrams = Counter()
            gram_idx.append([])
            gram_count.append([])
            for j in range(num_grams):
                idx = []
                for k in range(n):
                    idx.append(targets[i][j+k])
                idx = tuple(idx)
                ngrams[idx] += 1

            for k in range(n):
                gram_idx[-1].append([])
            for ngram in ngrams:
                for k in range(n):
                    gram_idx[-1][k].append(ngram[k])
                gram_count[-1].append(ngrams[ngram])

            while len(gram_count[-1]) < num_grams:
                for k in range(n):
                    gram_idx[-1][k].append(1)
                gram_count[-1].append(0)

        gram_idx = torch.LongTensor(gram_idx).cuda(blank_matrix.get_device()).transpose(0,1)
        gram_count = torch.Tensor(gram_count).cuda(blank_matrix.get_device()).view(batch_size, num_grams,1)
        blank_matrix = blank_matrix.view(batch_size, 1, length_ctc, length_ctc)
        for k in range(n):
            gram_k_probs = torch.gather(probs, -1, gram_idx[k].view(batch_size,1,num_grams).expand(batch_size,length_ctc,num_grams)).transpose(1,2).view(batch_size, num_grams, 1, length_ctc)
            if k == 0:
                state = gram_k_probs
            else:
                state = torch.matmul(state, blank_matrix) * gram_k_probs
        bag_grams = torch.sum(state, dim=-1)
        match_gram = torch.min(torch.cat([bag_grams,gram_count],dim = -1), dim = -1)[0]
        match_gram = torch.sum(match_gram).div(batch_size)
        loss = (- 2 * match_gram).div(length_tgt + expected_length - n + 1)
        return loss

    def compute_gram_loss(self, log_probs, targets, ctc_input_lengths):

        if not self.args.sctc_loss:
            if self.args.ngram_size == 1:
                loss = self.compute_ctc_1gram_loss(log_probs, targets)
            elif self.args.ngram_size == 2:
                loss = self.compute_ctc_bigram_loss(log_probs, targets)
            else:
                raise NotImplementedError
        else:
            loss = self.compute_sctc_ngram_loss(log_probs, targets, self.args.ngram_size)
        
        return loss

    def compute_hung_loss(self, log_probs, targets, ctc_input_lengths):

        from scipy.optimize import linear_sum_assignment
        margin = -math.log(0.1, 2)
        l_ctc = log_probs.size(1)
        batch_size, l_tgt = targets.size()
        targets = targets.eq(1) * 4 + targets
        if l_ctc <= l_tgt:
            return torch.sum(log_probs-log_probs)
        pad_targets = torch.cat((targets,torch.full((batch_size, l_ctc - l_tgt),4).to(targets)),dim = -1)
        pad_targets = pad_targets.repeat(1, l_ctc).view(batch_size, l_ctc, l_ctc)
        log_probs_matrix = log_probs.gather(dim=-1, index=pad_targets)
        best_match = np.repeat(np.arange(l_ctc).reshape(1, -1, 1), batch_size, axis=0)
        log_probs_numpy = log_probs_matrix.detach().cpu().numpy()

        for i in range(batch_size):
            raw_index, col_index = linear_sum_assignment(-log_probs_numpy[i])
            best_match[i] = col_index.reshape(-1, 1)
        best_match = torch.Tensor(best_match).to(targets).long()
        loss = - log_probs_matrix.gather(dim=-1, index=best_match)
        loss[loss>margin] = 0
        loss = loss.sum().div(batch_size * l_tgt)

        return loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, _ = sample["target"], sample["prev_target"]
        tgt_lengths = torch.sum(sample["target"].ne(1), dim = -1)
        src_length = src_tokens.size(1)
        if src_length > 200:
            max_length = 2 * src_length
        else:
            max_length = 3 * src_length
        ctc_input_lengths = torch.LongTensor(nsentences).cuda(src_tokens.get_device())
        ctc_input_lengths[:] = max_length

        prev_output_tokens = torch.LongTensor(nsentences, max_length).cuda(src_tokens.get_device())
        prev_output_tokens[:] = 3
        prev_output_tokens[:,0] = 0
        prev_output_tokens[:,-1] = 2

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        log_probs = F.log_softmax(outputs['word_ins']['out'], dim=-1).float()
        ctc_loss = 0
        if self.args.use_ngram:
            ctc_loss += self.compute_gram_loss(log_probs, tgt_tokens, ctc_input_lengths)
        if self.args.use_hung:
            ctc_loss += self.compute_hung_loss(log_probs, tgt_tokens, ctc_input_lengths)
        if self.args.use_word:
            log_probs = log_probs.transpose(0,1)
            ctc_loss += F.ctc_loss(log_probs, tgt_tokens, ctc_input_lengths,
                tgt_lengths, blank = 4, reduction = 'mean', zero_infinity = True)
        
        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": ctc_loss.data,
            "nll_loss": ctc_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        return ctc_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        loss = utils.item(sum(log.get("loss", 0).float() for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0).float() for log in logging_outputs))

        metrics.log_scalar('loss', loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

