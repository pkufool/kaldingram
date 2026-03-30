#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Entropy-based pruning for back-off n-gram ARPA language models."""

import argparse
import gzip
import logging
import math
import re
from collections import OrderedDict, defaultdict
from enum import Enum, unique
from io import StringIO


class Context(dict):
    """Store values for one context h and its back-off weight."""

    def __init__(self):
        super().__init__()
        self.log_bo = None


class Arpa:
    """Data structure for an ARPA language model."""

    UNK = "<unk>"
    SOS = "<s>"
    EOS = "</s>"
    FLOAT_NDIGITS = 7
    base = 10

    @staticmethod
    def _check_input(my_input):
        if not my_input:
            raise ValueError
        if isinstance(my_input, tuple):
            return my_input
        if isinstance(my_input, list):
            return tuple(my_input)
        if isinstance(my_input, str):
            return tuple(my_input.strip().split(" "))
        raise ValueError

    @staticmethod
    def _check_word(input_word):
        if not isinstance(input_word, str):
            raise ValueError
        if " " in input_word:
            raise ValueError

    def _replace_unks(self, words):
        return tuple((w if w in self else self._unk) for w in words)

    def __init__(self, path=None, encoding=None, unk=None):
        self._counts = OrderedDict()
        self._ngrams = OrderedDict()
        self._vocabulary = set()
        if unk is None:
            self._unk = self.UNK

        if path is not None:
            self.loadf(path, encoding)

    def __contains__(self, ngram):
        h = ngram[:-1]
        w = ngram[-1]
        return h in self._ngrams[len(h)] and w in self._ngrams[len(h)][h]

    def contains_word(self, word):
        self._check_word(word)
        return word in self._vocabulary

    def add_count(self, order, count):
        self._counts[order] = count
        self._ngrams[order - 1] = defaultdict(Context)

    def update_counts(self):
        for order in range(1, self.order() + 1):
            count = sum([len(wlist) for _, wlist in self._ngrams[order - 1].items()])
            if count > 0:
                self._counts[order] = count

    def add_entry(self, ngram, p, bo=None, order=None):
        del order
        h = ngram[:-1]
        w = ngram[-1]

        h_context = self._ngrams[len(h)][h]
        h_context[w] = p
        if bo is not None:
            self._ngrams[len(ngram)][ngram].log_bo = bo

        for word in ngram:
            self._vocabulary.add(word)

    def counts(self):
        return sorted(self._counts.items())

    def order(self):
        return max(self._counts.keys(), default=None)

    def vocabulary(self, sort=True):
        if sort:
            return sorted(self._vocabulary)
        return self._vocabulary

    def _entries(self, order):
        return (
            self._entry(h, w)
            for h, wlist in self._ngrams[order - 1].items()
            for w in wlist
        )

    def _entry(self, h, w):
        ngram = h + (w,)
        log_p = self._ngrams[len(h)][h][w]
        log_bo = self._log_bo(ngram)
        if log_bo is not None:
            return (
                round(log_p, self.FLOAT_NDIGITS),
                ngram,
                round(log_bo, self.FLOAT_NDIGITS),
            )
        return round(log_p, self.FLOAT_NDIGITS), ngram

    def _log_bo(self, ngram):
        if len(ngram) in self._ngrams and ngram in self._ngrams[len(ngram)]:
            return self._ngrams[len(ngram)][ngram].log_bo
        return None

    def _log_p(self, ngram):
        h = ngram[:-1]
        w = ngram[-1]
        if h in self._ngrams[len(h)] and w in self._ngrams[len(h)][h]:
            return self._ngrams[len(h)][h][w]
        return None

    def log_p_raw(self, ngram):
        log_p = self._log_p(ngram)
        if log_p is not None:
            return log_p

        if len(ngram) == 1:
            raise KeyError

        log_bo = self._log_bo(ngram[:-1])
        if log_bo is None:
            log_bo = 0
        return log_bo + self.log_p_raw(ngram[1:])

    def log_joint_prob(self, sequence):
        log_joint_p = 0
        seq = sequence
        while len(seq) > 0:
            log_joint_p += self.log_p_raw(seq)
            seq = seq[:-1]

            if len(seq) == 1 and seq[0] == self.SOS:
                seq = (self.EOS,)

        return log_joint_p

    def set_new_context(self, h):
        old_context = self._ngrams[len(h)][h]
        self._ngrams[len(h)][h] = Context()
        return old_context

    def log_p(self, ngram):
        words = self._check_input(ngram)
        if self._unk:
            words = self._replace_unks(words)
        return self.log_p_raw(words)

    def log_s(self, sentence, sos=SOS, eos=EOS):
        words = self._check_input(sentence)
        if self._unk:
            words = self._replace_unks(words)
        if sos:
            words = (sos,) + words
        if eos:
            words = words + (eos,)
        result = sum(self.log_p_raw(words[:i]) for i in range(1, len(words) + 1))
        if sos:
            result = result - self.log_p_raw(words[:1])
        return result

    def p(self, ngram):
        return self.base ** self.log_p(ngram)

    def s(self, sentence):
        return self.base ** self.log_s(sentence)

    def write(self, fp):
        fp.write("\n\\data\\\n")
        for order, count in self.counts():
            fp.write("ngram {}={}\n".format(order, count))
        fp.write("\n")
        for order, _ in self.counts():
            fp.write("\\{}-grams:\n".format(order))
            for e in self._entries(order):
                prob = e[0]
                ngram = " ".join(e[1])
                if len(e) == 2:
                    fp.write("{}\t{}\n".format(prob, ngram))
                elif len(e) == 3:
                    backoff = e[2]
                    fp.write("{}\t{}\t{}\n".format(prob, ngram, backoff))
                else:
                    raise ValueError
            fp.write("\n")
        fp.write("\\end\\\n")


class ArpaParser:
    """Parser and serializer for ARPA files."""

    @unique
    class State(Enum):
        DATA = 1
        COUNT = 2
        HEADER = 3
        ENTRY = 4

    re_count = re.compile(r"^ngram (\d+)=(\d+)$")
    re_header = re.compile(r"^\\(\d+)-grams:$")
    re_entry = re.compile(
        "^(-?\\d+(\\.\\d+)?([eE]-?\\d+)?)"
        "\\t"
        "(\\S+( \\S+)*)"
        "(\\t((-?\\d+(\\.\\d+)?)([eE]-?\\d+)?))?$"
    )

    def _parse(self, fp):
        self._result = []
        self._state = self.State.DATA
        self._tmp_model = None
        self._tmp_order = None
        for line in fp:
            line = line.strip()
            if self._state == self.State.DATA:
                self._data(line)
            elif self._state == self.State.COUNT:
                self._count(line)
            elif self._state == self.State.HEADER:
                self._header(line)
            elif self._state == self.State.ENTRY:
                self._entry(line)
        if self._state != self.State.DATA:
            raise Exception(line)
        return self._result

    def _data(self, line):
        if line == "\\data\\":
            self._state = self.State.COUNT
            self._tmp_model = Arpa()

    def _count(self, line):
        match = self.re_count.match(line)
        if match:
            order = match.group(1)
            count = match.group(2)
            self._tmp_model.add_count(int(order), int(count))
        elif not line:
            self._state = self.State.HEADER
        else:
            raise Exception(line)

    def _header(self, line):
        match = self.re_header.match(line)
        if match:
            self._state = self.State.ENTRY
            self._tmp_order = int(match.group(1))
        elif line == "\\end\\":
            self._result.append(self._tmp_model)
            self._state = self.State.DATA
            self._tmp_model = None
            self._tmp_order = None
        elif not line:
            pass
        else:
            raise Exception(line)

    def _entry(self, line):
        match = self.re_entry.match(line)
        if match:
            p = self._float_or_int(match.group(1))
            ngram = tuple(match.group(4).split(" "))
            bo_match = match.group(7)
            bo = self._float_or_int(bo_match) if bo_match else None
            self._tmp_model.add_entry(ngram, p, bo, self._tmp_order)
        elif not line:
            self._state = self.State.HEADER
        else:
            raise Exception(line)

    @staticmethod
    def _float_or_int(s):
        f = float(s)
        i = int(f)
        if str(i) == s:
            return i
        return f

    def load(self, fp):
        return self._parse(fp)

    def loadf(self, path, encoding=None):
        path = str(path)
        if path.endswith(".gz"):
            with gzip.open(path, mode="rt", encoding=encoding) as f:
                return self.load(f)
        with open(path, mode="rt", encoding=encoding) as f:
            return self.load(f)

    def loads(self, s):
        with StringIO(s) as f:
            return self.load(f)

    def dump(self, obj, fp):
        obj.write(fp)

    def dumpf(self, obj, path, encoding=None):
        path = str(path)
        if path.endswith(".gz"):
            with gzip.open(path, mode="wt", encoding=encoding) as f:
                return self.dump(obj, f)
        with open(path, mode="wt", encoding=encoding) as f:
            self.dump(obj, f)

    def dumps(self, obj):
        with StringIO() as f:
            self.dump(obj, f)
            return f.getvalue()


def add_log_p(prev_log_sum, log_p, base):
    return math.log(base**log_p + base**prev_log_sum, base)


def compute_numerator_denominator(lm, h):
    log_sum_seen_h = -math.inf
    log_sum_seen_h_lower = -math.inf
    base = lm.base
    for w, log_p in lm._ngrams[len(h)][h].items():
        log_sum_seen_h = add_log_p(log_sum_seen_h, log_p, base)

        ngram = h + (w,)
        log_p_lower = lm.log_p_raw(ngram[1:])
        log_sum_seen_h_lower = add_log_p(log_sum_seen_h_lower, log_p_lower, base)

    numerator = 1.0 - base**log_sum_seen_h
    denominator = 1.0 - base**log_sum_seen_h_lower
    return numerator, denominator


def prune(lm, threshold, minorder):
    for i in range(lm.order(), max(minorder - 1, 1), -1):
        logging.info("processing %d-grams ...", i)
        count_pruned_ngrams = 0

        h_dict = lm._ngrams[i - 1]
        for h in list(h_dict.keys()):
            log_bow = lm._log_bo(h)
            if log_bow is None:
                log_bow = 0

            numerator, denominator = compute_numerator_denominator(lm, h)
            h_log_p = lm.log_joint_prob(h)

            all_pruned = True
            pruned_w_set = set()

            for w, log_p in h_dict[h].items():
                ngram = h + (w,)
                backoff_prob = lm.log_p_raw(ngram[1:])

                new_log_bow = math.log(
                    numerator + lm.base**log_p, lm.base
                ) - math.log(denominator + lm.base**backoff_prob, lm.base)

                delta_prob = backoff_prob + new_log_bow - log_p
                delta_entropy = -(lm.base**h_log_p) * (
                    (lm.base**log_p) * delta_prob
                    + numerator * (new_log_bow - log_bow)
                )

                perp_change = lm.base**delta_entropy - 1.0

                pruned = threshold > 0 and perp_change < threshold

                if (
                    pruned
                    and len(ngram) in lm._ngrams
                    and len(lm._ngrams[len(ngram)][ngram]) > 0
                ):
                    pruned = False

                logging.debug(
                    "CONTEXT %s WORD %s CONTEXTPROB %f OLDPROB %f NEWPROB %f "
                    "DELTA-H %f DELTA-LOGP %f PPL-CHANGE %f PRUNED %s",
                    h,
                    w,
                    h_log_p,
                    log_p,
                    backoff_prob + new_log_bow,
                    delta_entropy,
                    delta_prob,
                    perp_change,
                    pruned,
                )

                if pruned:
                    pruned_w_set.add(w)
                    count_pruned_ngrams += 1
                else:
                    all_pruned = False

            if all_pruned and len(pruned_w_set) == len(h_dict[h]):
                del h_dict[h]
            elif len(pruned_w_set) > 0:
                old_context = lm.set_new_context(h)

                for w, p_w in old_context.items():
                    if w not in pruned_w_set:
                        lm.add_entry(h + (w,), p_w)

        logging.info("pruned %d %d-grams", count_pruned_ngrams, i)

    for i in range(max(minorder - 1, 1) + 1, lm.order() + 1):
        for h in lm._ngrams[i - 1]:
            numerator, denominator = compute_numerator_denominator(lm, h)
            new_log_bow = math.log(numerator, lm.base) - math.log(denominator, lm.base)
            lm._ngrams[len(h)][h].log_bo = new_log_bow

    lm.update_counts()


def add_arguments(parser):
    parser.add_argument(
        "-threshold",
        "--threshold",
        type=float,
        default=1e-6,
        help="Relative perplexity change threshold",
    )
    parser.add_argument("-lm", "--lm", type=str, required=True, help="Input ARPA file")
    parser.add_argument(
        "-write-lm",
        "--write-lm",
        type=str,
        required=True,
        help="Output ARPA path after pruning",
    )
    parser.add_argument(
        "-minorder",
        "--minorder",
        type=int,
        default=1,
        help="Only prune n-grams with this order and above",
    )
    parser.add_argument(
        "-encoding",
        "--encoding",
        type=str,
        default="utf-8",
        help="Encoding of the ARPA file",
    )
    parser.add_argument(
        "-verbose",
        "--verbose",
        type=int,
        default=2,
        choices=[0, 1, 2, 3, 4, 5],
        help="Verbose level, where 0 is most noisy and 5 is most silent",
    )


def run(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        level=args.verbose * 10,
    )

    logging.info("Loading arpa file from %s", args.lm)
    parser = ArpaParser()
    models = parser.loadf(args.lm, encoding=args.encoding)
    lm = models[0]

    logging.info("Stats before pruning:")
    for i, cnt in lm.counts():
        logging.info("ngram %d=%d", i, cnt)

    logging.info("Start pruning model with threshold=%.3E...", args.threshold)
    prune(lm, args.threshold, args.minorder)

    logging.info("Stats after pruning:")
    for i, cnt in lm.counts():
        logging.info("ngram %d=%d", i, cnt)

    logging.info("Saving pruned arpa file to %s", args.write_lm)
    parser.dumpf(lm, args.write_lm, encoding=args.encoding)
    logging.info("Done.")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Prune an n-gram language model based on relative entropy between "
            "the original and pruned model."
        )
    )
    add_arguments(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
