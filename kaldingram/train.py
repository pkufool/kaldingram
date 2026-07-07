#!/usr/bin/env python3

# Copyright 2016  Johns Hopkins University (Author: Daniel Povey)
#           2018  Ruizhe Huang
#           2026  Wei Kang

"""Train Kneser-Ney n-gram language models in ARPA format.

This is an implementation of computing Kneser-Ney smoothed language model
in the same way as srilm. This is a back-off, unmodified version of
Kneser-Ney smoothing, which produces the same results as the following
command (as an example) of srilm:

    ngram-count -order 4 -kn-modify-counts-at-end -ukndiscount \
        -gt1min 0 -gt2min 0 -gt3min 0 -gt4min 0 -text corpus.txt -lm lm.arpa

The data structure is based on:
    kaldi/egs/wsj/s5/utils/lang/make_phone_lm.py
The smoothing algorithm is based on:
    http://www.speech.sri.com/projects/srilm/manpages/ngram-discount.7.html
"""

import argparse
import gc
import io
import math
import os
import pickle
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from multiprocessing import Pool

from tqdm import tqdm


# For encoding-agnostic scripts, we assume byte stream as input.
# Need to be very careful about the use of strip() and split()
# in this case, because there is a latin-1 whitespace character
# (nbsp) which is part of the unicode encoding range.
# Ref: kaldi/egs/wsj/s5/utils/lang/bpe/prepend_words.py @ 69cd717
default_encoding = "latin-1"

strip_chars = " \t\r\n"
whitespace = re.compile("[ \t]+")


class _GcSuspended:
    """Temporarily suspend cyclic GC around object-heavy hot loops.

    Counting creates millions of dictionaries, tuples and sets, but they do not
    form reference cycles.  CPython's reference counting is enough to reclaim
    them, while periodic cyclic-GC scans become very expensive as the count
    tables grow.  Keep the previous GC state so callers/tests that deliberately
    disabled GC stay unchanged.
    """

    def __enter__(self):
        self._was_enabled = gc.isenabled()
        if self._was_enabled:
            gc.disable()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._was_enabled:
            gc.enable()
        return False


class CountsForHistory:
    """Store counts/statistics for one history state.

    This class (which is more like a struct) is used inside class NgramCounts.
    It really does the job of a dict from word to float, but it also keeps
    track of the total count.
    """

    def __init__(self):
        self.word_to_count = defaultdict(int)
        # using a set to count the number of unique contexts
        self.word_to_context = defaultdict(set)
        self.word_to_f = {}  # discounted probability
        self.word_to_bow = {}  # back-off weight
        self.total_count = 0

    def add_count(self, predicted_word, context_word, count):
        assert count >= 0

        self.total_count += count
        self.word_to_count[predicted_word] += count
        if context_word is not None:
            self.word_to_context[predicted_word].add(context_word)


class NgramCounts:
    """Store counts and probabilities for back-off KN n-grams."""

    # A note on data-structure.  We store n-gram counts as a list, indexed by
    # (history-length == n-gram order minus one) of dicts from histories to
    # CountsForHistory, where histories are tuples of tokens.  For instance,
    # when accumulating the 4-gram count for the token 'd' in the sequence
    # 'a b c d', we'd access self.counts[3][('a','b','c')] and then update the
    # count for 'd', where the [3] indexes the list, the [('a','b','c')]
    # indexes a dict, and 'd' indexes CountsForHistory.word_to_count.

    def __init__(self, ngram_order, verbose=0, bos_symbol="<s>", eos_symbol="</s>"):
        assert ngram_order >= 1

        self.ngram_order = ngram_order
        self.verbose = verbose
        self.bos_symbol = sys.intern(bos_symbol)
        self.eos_symbol = sys.intern(eos_symbol)

        self.counts = []
        for _ in range(ngram_order):
            self.counts.append(defaultdict(CountsForHistory))

        self.d = []  # list of discounting factor for each order of ngram

    # adds a raw count (called while processing input data).
    # Suppose we see the sequence 'b c d e' and ngram_order=4, 'history'
    # would be ('b','c','d') and 'predicted_word' would be 'e'; 'count' would
    # be 1.
    def add_count(self, history, predicted_word, context_word, count):
        self.counts[len(history)][history].add_count(predicted_word, context_word, count)

    def merge_counts(self, other):
        """Merge counts from another NgramCounts instance into this one.

        This is on the hot path of parallel training.  Most high-order
        histories are owned by a single shard, so move those
        CountsForHistory objects wholesale instead of recreating an empty
        history and copying every word/context in Python.  When a history is
        shared, also move per-word context sets for words that are not yet
        present; only the genuinely overlapping entries need arithmetic/set
        union.
        """
        assert self.ngram_order == other.ngram_order
        for n in tqdm(range(self.ngram_order), desc="merge", unit="order", leave=False):
            this_order = self.counts[n]
            for history, other_hist in tqdm(
                other.counts[n].items(),
                desc="  {}-gram".format(n + 1),
                unit="hist",
                leave=False,
            ):
                my_hist = this_order.get(history)
                if my_hist is None:
                    this_order[history] = other_hist
                    continue

                my_hist.total_count += other_hist.total_count

                my_word_to_count = my_hist.word_to_count
                for word, count in other_hist.word_to_count.items():
                    my_word_to_count[word] += count

                my_word_to_context = my_hist.word_to_context
                for word, contexts in other_hist.word_to_context.items():
                    my_contexts = my_word_to_context.get(word)
                    if my_contexts is None:
                        my_word_to_context[word] = contexts
                    else:
                        my_contexts.update(contexts)

    # 'line' is a string containing a sequence of tokens.
    # This function adds the un-smoothed counts from this line of text.
    def add_raw_counts_from_line(self, line):
        if line == "":
            words = [self.bos_symbol, self.eos_symbol]
        else:
            # Intern tokens so repeated vocabulary items share one Python string
            # object inside a process.  This reduces memory, speeds dict/tuple
            # comparisons, and makes pickle files for worker shards much smaller
            # because pickle can memoize repeated token objects by identity.
            words = (
                [self.bos_symbol]
                + [sys.intern(w) for w in whitespace.split(line)]
                + [self.eos_symbol]
            )

        for i in range(len(words)):
            for n in range(1, self.ngram_order + 1):
                if i + n > len(words):
                    break
                ngram = words[i : i + n]
                predicted_word = ngram[-1]
                history = tuple(ngram[:-1])
                if i == 0 or n == self.ngram_order:
                    context_word = None
                else:
                    context_word = words[i - 1]

                self.add_count(history, predicted_word, context_word, 1)

    def add_raw_counts_from_standard_input(self):
        lines_processed = 0
        # byte stream as input
        infile = io.TextIOWrapper(sys.stdin.buffer, encoding=default_encoding)
        with _GcSuspended():
            for line in tqdm(infile, desc="counting", unit="lines"):
                line = line.strip(strip_chars)
                self.add_raw_counts_from_line(line)
                lines_processed += 1

        if lines_processed == 0 or self.verbose > 0:
            print(
                "kaldingram train: processed {0} lines of input".format(lines_processed),
                file=sys.stderr,
            )

    def add_raw_counts_from_file(self, filename):
        file_size = os.path.getsize(filename)
        lines_processed = 0
        with _GcSuspended():
            with open(filename, encoding=default_encoding) as fp:
                # default_encoding is latin-1, so len(line) (chars) == bytes read.
                with tqdm(total=file_size, unit="B", unit_scale=True, desc="counting") as pbar:
                    pending_update = 0
                    for line in fp:
                        pending_update += len(line)
                        if pending_update >= 1024 * 1024:
                            pbar.update(pending_update)
                            pending_update = 0
                        line = line.strip(strip_chars)
                        if self.ngram_order == 1:
                            self.add_raw_counts_from_line(line.split()[0])
                        else:
                            self.add_raw_counts_from_line(line)
                        lines_processed += 1
                    if pending_update:
                        pbar.update(pending_update)

        if lines_processed == 0 or self.verbose > 0:
            print(
                "kaldingram train: processed {0} lines of input".format(lines_processed),
                file=sys.stderr,
            )

    def add_raw_counts_from_file_parallel(self, filename, num_workers):
        """Count n-grams in parallel using byte-range file sharding.

        Workers count independently and write results to temp files (so the
        large NgramCounts never travels through the IPC pipe; only the line
        count is returned). Each worker shows its own counting progress bar.
        Finally the main process loads and merges each shard.
        """
        file_size = os.path.getsize(filename)
        offsets = _compute_shard_offsets(filename, num_workers, file_size)
        num_shards = len(offsets) - 1

        tmp_dir = tempfile.mkdtemp(prefix="kaldingram_")
        tmp_files = [
            os.path.join(tmp_dir, "shard_{}.bin".format(i))
            for i in range(num_shards)
        ]
        worker_args = [
            (filename, offsets[i], offsets[i + 1], self.ngram_order,
             self.bos_symbol, self.eos_symbol, tmp_files[i], i)
            for i in range(num_shards)
        ]

        total_lines = 0
        try:
            with _GcSuspended():
                # Phase 1: parallel counting; each worker renders its own
                # progress bar.  On Unix/fork workers inherit the suspended-GC
                # state; _count_worker also suspends GC for spawn platforms.
                with Pool(num_workers) as pool:
                    for lines_in_shard in pool.imap_unordered(_count_worker, worker_args):
                        total_lines += lines_in_shard

                # Phase 2: load and merge each shard sequentially.
                for tmp_file in tqdm(tmp_files, desc="merging", unit="shard"):
                    with open(tmp_file, "rb") as f:
                        partial_counts, _ = pickle.load(f)
                    self.merge_counts(partial_counts)
                    os.unlink(tmp_file)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if total_lines == 0 or self.verbose > 0:
            print(
                "kaldingram train: processed {0} lines of input "
                "({1} workers)".format(total_lines, num_workers),
                file=sys.stderr,
            )

    def cal_discounting_constants(self):
        # For each order N of N-grams, we calculate discounting constant
        # D_N = n1_N / (n1_N + 2 * n2_N), where n1_N is the number of unique
        # N-grams with count = 1 (counts-of-counts).  This constant is used
        # similarly to absolute discounting.
        # Return value: d is a list of floats, where d[N+1] = D_N

        # For the lowest order, i.e., 1-gram, we do not need to discount, thus
        # the constant is 0.  This is a special case: as we currently assume
        # having seen all vocabularies in the dictionary, but perhaps this is
        # not the case for some other scenarios.
        self.d = [0]
        for n in tqdm(range(1, self.ngram_order), desc="discounting", unit="order"):
            this_order_counts = self.counts[n]
            n1 = 0
            n2 = 0
            for _, counts_for_hist in tqdm(
                this_order_counts.items(),
                desc="  {}-gram".format(n + 1),
                unit="hist",
                leave=False,
            ):
                stat = Counter(counts_for_hist.word_to_count.values())
                n1 += stat[1]
                n2 += stat[2]

            if n1 + 2 * n2 == 0:
                # Not enough counts-of-counts to estimate D (e.g. after
                # aggressive count cutoffs removed all count-1/2 n-grams at
                # this order, or a very small corpus). Fall back to a small
                # non-zero discount so that back-off weights remain finite.
                self.d.append(0.1)
                continue

            # We do this max(0.1, xxx) to avoid a zero discounting constant D
            # due to n1=0, which could happen if the number of symbols is small.
            # Otherwise, a zero discounting constant can cause division by zero
            # in computing BOW.
            self.d.append(max(0.1, float(n1)) / (n1 + 2 * n2))

    def apply_count_cutoffs(self, gtmin):
        # Remove n-grams whose effective count is below the per-order minimum,
        # matching SRILM's -gt{N}min option (with -kn-modify-counts-at-end).
        #
        # gtmin is a list indexed by order-1 (i.e. gtmin[k] is the minimum
        # count for (k+1)-grams, k = history length). An n-gram is kept iff its
        # effective count >= gtmin[k]; the default 0 keeps everything.
        #
        # The effective count is the same count used in the probability
        # computation: the raw count c(a_z) for the highest order, and the
        # modified (Kneser-Ney) count n(*_z) = number of distinct contexts for
        # lower orders (falling back to the raw count for <s>-initial histories
        # that have no modified count).
        #
        # Pruning proceeds from high to low order so that, when we decide
        # whether to keep a lower-order n-gram, its modified count already
        # reflects any higher-order n-grams that were pruned. Removing an
        # n-gram (h, w) at order k+1 also discards the context h[0] from the
        # modified count of the lower-order n-gram (h[1:], w) at order k.

        for k in tqdm(
            range(self.ngram_order - 1, -1, -1),
            desc="count cutoffs",
            unit="order",
        ):
            order_min = gtmin[k]
            if order_min <= 0:
                continue  # no cutoff at this order
            this_order = self.counts[k]
            for h in tqdm(
                list(this_order.keys()),
                desc="  {}-gram".format(k + 1),
                unit="hist",
                leave=False,
            ):
                cfh = this_order[h]
                to_remove = []
                for w in list(cfh.word_to_count.keys()):
                    ctx = cfh.word_to_context.get(w)
                    if k == self.ngram_order - 1:
                        eff = cfh.word_to_count[w]  # highest order: raw count
                    else:
                        eff = len(ctx) if ctx else cfh.word_to_count[w]
                    if eff < order_min:
                        # Backoff closure: keep this n-gram if any surviving
                        # higher-order n-gram backs off to it. ctx (already
                        # cascaded from higher-order pruning) holds exactly the
                        # surviving references; if it is non-empty we must keep
                        # (h, w) so the backoff path stays valid.
                        if ctx:
                            continue
                        to_remove.append(w)

                for w in to_remove:
                    del cfh.word_to_count[w]
                    cfh.word_to_context.pop(w, None)
                    # Cascade: this (k+1)-gram (h, w) contributed the context
                    # h[0] to the modified count of the lower-order k-gram
                    # (h[1:], w) stored at counts[k-1].
                    if k >= 1:
                        lower = self.counts[k - 1].get(h[1:])
                        if lower is not None:
                            lower_ctx = lower.word_to_context.get(w)
                            if lower_ctx is not None:
                                lower_ctx.discard(h[0])

                if to_remove:
                    cfh.total_count = sum(cfh.word_to_count.values())
                if not cfh.word_to_count:
                    del this_order[h]

    def cal_f(self):
        # f(a_z) is a probability distribution of word sequence a_z.
        # Typically f(a_z) is discounted to be less than the ML estimate so we
        # have some leftover probability for the z words unseen in the context
        # (a_).
        #
        # f(a_z) = (c(a_z) - D0) / c(a_)    ;; for highest order N-grams
        # f(_z)  = (n(*_z) - D1) / n(*_*)   ;; for lower order N-grams

        # highest order N-grams
        n = self.ngram_order - 1
        this_order_counts = self.counts[n]
        for _, counts_for_hist in tqdm(
            this_order_counts.items(),
            desc="cal_f {}-gram".format(n + 1),
            unit="hist",
        ):
            for w, c in counts_for_hist.word_to_count.items():
                counts_for_hist.word_to_f[w] = (
                    max((c - self.d[n]), 0) * 1.0 / counts_for_hist.total_count
                )

        # lower order N-grams
        for n in tqdm(range(0, self.ngram_order - 1), desc="cal_f", unit="order"):
            this_order_counts = self.counts[n]
            for _, counts_for_hist in tqdm(
                this_order_counts.items(),
                desc="  {}-gram".format(n + 1),
                unit="hist",
                leave=False,
            ):
                n_star_star = 0
                for w in counts_for_hist.word_to_count.keys():
                    n_star_star += len(counts_for_hist.word_to_context[w])

                if n_star_star != 0:
                    for w in counts_for_hist.word_to_count.keys():
                        n_star_z = len(counts_for_hist.word_to_context[w])
                        counts_for_hist.word_to_f[w] = (
                            max((n_star_z - self.d[n]), 0) * 1.0 / n_star_star
                        )
                else:  # patterns begin with <s>, they do not have "modified count", so use raw count instead
                    for w in counts_for_hist.word_to_count.keys():
                        n_star_z = counts_for_hist.word_to_count[w]
                        counts_for_hist.word_to_f[w] = (
                            max((n_star_z - self.d[n]), 0)
                            * 1.0
                            / counts_for_hist.total_count
                        )

    def cal_bow(self):
        # Backoff weights are only necessary for ngrams which form a prefix of
        # a longer ngram.  Thus, two sorts of ngrams do not have a bow:
        # 1) highest order ngram
        # 2) ngrams ending in </s>
        #
        # bow(a_) = (1 - Sum_Z1 f(a_z)) / (1 - Sum_Z1 f(_z))
        # Note that Z1 is the set of all words with c(a_z) > 0

        # highest order N-grams
        n = self.ngram_order - 1
        this_order_counts = self.counts[n]
        for _, counts_for_hist in tqdm(
            this_order_counts.items(),
            desc="cal_bow {}-gram".format(n + 1),
            unit="hist",
        ):
            for w in counts_for_hist.word_to_count.keys():
                counts_for_hist.word_to_bow[w] = None

        # lower order N-grams
        for n in tqdm(range(0, self.ngram_order - 1), desc="cal_bow", unit="order"):
            this_order_counts = self.counts[n]
            for hist, counts_for_hist in tqdm(
                this_order_counts.items(),
                desc="  {}-gram".format(n + 1),
                unit="hist",
                leave=False,
            ):
                for w in counts_for_hist.word_to_count.keys():
                    if w == self.eos_symbol:
                        counts_for_hist.word_to_bow[w] = None
                    else:
                        a_ = hist + (w,)

                        assert len(a_) < self.ngram_order
                        if a_ not in self.counts[len(a_)]:
                            # No longer n-gram uses (hist, w) as context (e.g.
                            # after count cutoffs pruned all of its extensions),
                            # so no back-off weight is needed for this n-gram.
                            counts_for_hist.word_to_bow[w] = None
                            continue

                        a_counts_for_hist = self.counts[len(a_)][a_]

                        sum_z1_f_a_z = 0
                        for u in a_counts_for_hist.word_to_count.keys():
                            sum_z1_f_a_z += a_counts_for_hist.word_to_f[u]

                        sum_z1_f_z = 0
                        lower_hist = a_[1:]
                        lower_counts_for_hist = self.counts[len(lower_hist)][lower_hist]
                        # Should be careful here: what is Z1
                        for u in a_counts_for_hist.word_to_count.keys():
                            sum_z1_f_z += lower_counts_for_hist.word_to_f[u]

                        if sum_z1_f_z < 1:
                            counts_for_hist.word_to_bow[w] = (1.0 - sum_z1_f_a_z) / (
                                1.0 - sum_z1_f_z
                            )
                        else:
                            counts_for_hist.word_to_bow[w] = None

    def print_as_arpa(self, fout=None):
        # print as ARPA format.
        if fout is None:
            fout = io.TextIOWrapper(sys.stdout.buffer, encoding=default_encoding)
        print("\\data\\", file=fout)
        for hist_len in range(self.ngram_order):
            # print the number of n-grams.
            print(
                "ngram {0}={1}".format(
                    hist_len + 1,
                    sum(
                        [
                            len(counts_for_hist.word_to_f)
                            for counts_for_hist in self.counts[hist_len].values()
                        ]
                    ),
                ),
                file=fout,
            )

        print("", file=fout)

        for hist_len in tqdm(range(self.ngram_order), desc="print_arpa", unit="order"):
            print("\\{0}-grams:".format(hist_len + 1), file=fout)

            this_order_counts = self.counts[hist_len]
            for hist, counts_for_hist in tqdm(
                this_order_counts.items(),
                desc="  {}-gram".format(hist_len + 1),
                unit="hist",
                leave=False,
            ):
                for word in counts_for_hist.word_to_count.keys():
                    ngram = hist + (word,)
                    prob = counts_for_hist.word_to_f[word]
                    bow = counts_for_hist.word_to_bow[word]

                    if prob == 0:  # f(<s>) is always 0
                        prob = 1e-99

                    line = "{0}\t{1}".format("%.7f" % math.log10(prob), " ".join(ngram))
                    if bow is not None:
                        line += "\t{0}".format("%.7f" % math.log10(bow))
                    print(line, file=fout)
            print("", file=fout)
        print("\\end\\", file=fout)


def _compute_shard_offsets(filename, num_shards, file_size):
    """Compute byte offsets that split the file into shards at line boundaries."""
    offsets = [0]
    with open(filename, "rb") as f:
        for i in range(1, num_shards):
            target = file_size * i // num_shards
            f.seek(target)
            f.readline()
            offsets.append(f.tell())
    offsets.append(file_size)
    return offsets


def _count_worker(args):
    """Worker: count n-grams in a byte range of the file.

    Runs add_raw_counts_from_line on each line in the byte range [start, end),
    pickles the resulting NgramCounts to out_path, and returns the line count.
    Only the small line count is sent back through the IPC pipe. Each worker
    shows its own counting progress bar (positioned by worker index).
    """
    (filename, start, end, ngram_order, bos_symbol, eos_symbol, out_path,
     worker_idx) = args
    with _GcSuspended():
        local_counts = NgramCounts(ngram_order, bos_symbol=bos_symbol, eos_symbol=eos_symbol)
        lines_processed = 0
        with open(filename, "rb") as f:
            f.seek(start)
            with tqdm(
                total=end - start,
                position=worker_idx,
                desc="worker {}".format(worker_idx),
                unit="B",
                unit_scale=True,
                leave=False,
            ) as pbar:
                pos = start
                pending_update = 0
                while pos < end:
                    raw_line = f.readline()
                    if not raw_line:
                        break
                    raw_len = len(raw_line)
                    pos += raw_len
                    pending_update += raw_len
                    if pending_update >= 1024 * 1024 or pos >= end:
                        pbar.update(pending_update)
                        pending_update = 0
                    line = raw_line.decode(default_encoding).strip(strip_chars)
                    if ngram_order == 1 and line:
                        parts = line.split()
                        if parts:
                            local_counts.add_raw_counts_from_line(parts[0])
                    else:
                        local_counts.add_raw_counts_from_line(line)
                    lines_processed += 1
        with open(out_path, "wb") as f:
            pickle.dump((local_counts, lines_processed), f, protocol=pickle.HIGHEST_PROTOCOL)
        return lines_processed


def add_arguments(parser):
    parser.add_argument(
        "--ngram-order",
        type=int,
        default=4,
        choices=[1, 2, 3, 4, 5],
        help="Order of n-gram",
    )
    parser.add_argument("--text", type=str, default=None, help="Path to corpus")
    parser.add_argument(
        "--lm",
        type=str,
        default=None,
        help="Path to output ARPA LM",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help="Verbose level",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for counting (default: 1, set 0 for auto)",
    )
    # Count cutoffs, matching SRILM's -gt{N}min (with -kn-modify-counts-at-end).
    # An n-gram is dropped if its effective count is below the threshold for its
    # order; the default 0 keeps everything (backwards compatible). The highest
    # order thresholds on the raw count, lower orders on the modified count.
    # Values > 2 are not recommended (they can leave too few counts-of-counts to
    # estimate the discounting constants).
    for _n in range(1, 6):
        parser.add_argument(
            "--gt{}min".format(_n),
            type=int,
            default=0,
            help="Minimum count for {}-grams (default: 0, keep all)".format(_n),
        )


def run(args):
    num_workers = args.num_workers
    if num_workers == 0:
        num_workers = os.cpu_count() or 1

    ngram_counts = NgramCounts(args.ngram_order, verbose=args.verbose)

    if args.text is None:
        ngram_counts.add_raw_counts_from_standard_input()
    elif num_workers > 1:
        assert os.path.isfile(args.text)
        ngram_counts.add_raw_counts_from_file_parallel(args.text, num_workers)
    else:
        assert os.path.isfile(args.text)
        ngram_counts.add_raw_counts_from_file(args.text)

    # Apply per-order count cutoffs (SRILM -gt{N}min) before discounting so
    # that discounting constants and probabilities are estimated from the
    # pruned model.
    gtmin = [getattr(args, "gt{}min".format(n)) for n in range(1, args.ngram_order + 1)]
    if any(g > 0 for g in gtmin):
        ngram_counts.apply_count_cutoffs(gtmin)

    ngram_counts.cal_discounting_constants()
    ngram_counts.cal_f()
    ngram_counts.cal_bow()

    if args.lm is None:
        ngram_counts.print_as_arpa()
    else:
        with open(args.lm, "w", encoding=default_encoding) as f:
            ngram_counts.print_as_arpa(fout=f)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Kneser-Ney smoothed language model in ARPA format. "
            "By default, reads corpus from stdin and writes to stdout."
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
