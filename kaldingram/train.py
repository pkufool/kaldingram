#!/usr/bin/env python3

"""Train Kneser-Ney n-gram language models in ARPA format."""

import argparse
import io
import math
import os
import pickle
import re
import sys
import tempfile
from collections import Counter, defaultdict
from multiprocessing import Pool

from tqdm import tqdm


# For encoding-agnostic scripts, we assume byte stream as input.
# Need to be very careful about the use of strip() and split()
# in this case, because there is a latin-1 whitespace character
# (nbsp) which is part of the unicode encoding range.
default_encoding = "latin-1"

strip_chars = " \t\r\n"
whitespace = re.compile("[ \t]+")


class CountsForHistory:
    """Store counts/statistics for one history state."""

    def __init__(self):
        self.word_to_count = defaultdict(int)
        self.word_to_context = defaultdict(set)
        self.word_to_f = {}
        self.word_to_bow = {}
        self.total_count = 0

    def add_count(self, predicted_word, context_word, count):
        assert count >= 0

        self.total_count += count
        self.word_to_count[predicted_word] += count
        if context_word is not None:
            self.word_to_context[predicted_word].add(context_word)


class NgramCounts:
    """Store counts and probabilities for back-off KN n-grams."""

    def __init__(self, ngram_order, verbose=0, bos_symbol="<s>", eos_symbol="</s>"):
        assert ngram_order >= 1

        self.ngram_order = ngram_order
        self.verbose = verbose
        self.bos_symbol = bos_symbol
        self.eos_symbol = eos_symbol

        self.counts = []
        for _ in range(ngram_order):
            self.counts.append(defaultdict(CountsForHistory))

        self.d = []

    def add_count(self, history, predicted_word, context_word, count):
        self.counts[len(history)][history].add_count(predicted_word, context_word, count)

    def merge_counts(self, other):
        """Merge counts from another NgramCounts instance into this one."""
        assert self.ngram_order == other.ngram_order
        for n in range(self.ngram_order):
            for history, other_hist in other.counts[n].items():
                my_hist = self.counts[n][history]
                for word, count in other_hist.word_to_count.items():
                    my_hist.word_to_count[word] += count
                for word, contexts in other_hist.word_to_context.items():
                    my_hist.word_to_context[word] |= contexts
                my_hist.total_count += other_hist.total_count

    def add_raw_counts_from_line(self, line):
        if line == "":
            words = [self.bos_symbol, self.eos_symbol]
        else:
            words = [self.bos_symbol] + whitespace.split(line) + [self.eos_symbol]

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
        infile = io.TextIOWrapper(sys.stdin.buffer, encoding=default_encoding)
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
        with open(filename, encoding=default_encoding) as fp:
            # default_encoding is latin-1, so len(line) (chars) == bytes read.
            with tqdm(total=file_size, unit="B", unit_scale=True, desc="counting") as pbar:
                for line in fp:
                    pbar.update(len(line))
                    line = line.strip(strip_chars)
                    if self.ngram_order == 1:
                        self.add_raw_counts_from_line(line.split()[0])
                    else:
                        self.add_raw_counts_from_line(line)
                    lines_processed += 1

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

        # Phase 1: parallel counting; each worker renders its own progress bar.
        total_lines = 0
        with Pool(num_workers) as pool:
            for lines_in_shard in pool.imap_unordered(_count_worker, worker_args):
                total_lines += lines_in_shard

        # Phase 2: load and merge each shard sequentially.
        for tmp_file in tqdm(tmp_files, desc="merging", unit="shard"):
            with open(tmp_file, "rb") as f:
                partial_counts, _ = pickle.load(f)
            self.merge_counts(partial_counts)
            os.unlink(tmp_file)
        os.rmdir(tmp_dir)

        if total_lines == 0 or self.verbose > 0:
            print(
                "kaldingram train: processed {0} lines of input "
                "({1} workers)".format(total_lines, num_workers),
                file=sys.stderr,
            )

    def cal_discounting_constants(self):
        # For unigrams, no discounting is applied.
        self.d = [0]
        for n in range(1, self.ngram_order):
            this_order_counts = self.counts[n]
            n1 = 0
            n2 = 0
            for _, counts_for_hist in this_order_counts.items():
                stat = Counter(counts_for_hist.word_to_count.values())
                n1 += stat[1]
                n2 += stat[2]
            assert n1 + 2 * n2 > 0

            # Keep a non-zero floor to avoid division-by-zero in BOW computation.
            self.d.append(max(0.1, float(n1)) / (n1 + 2 * n2))

    def cal_f(self):
        n = self.ngram_order - 1
        this_order_counts = self.counts[n]
        for _, counts_for_hist in this_order_counts.items():
            for w, c in counts_for_hist.word_to_count.items():
                counts_for_hist.word_to_f[w] = (
                    max((c - self.d[n]), 0) * 1.0 / counts_for_hist.total_count
                )

        for n in range(0, self.ngram_order - 1):
            this_order_counts = self.counts[n]
            for _, counts_for_hist in this_order_counts.items():
                n_star_star = 0
                for w in counts_for_hist.word_to_count.keys():
                    n_star_star += len(counts_for_hist.word_to_context[w])

                if n_star_star != 0:
                    for w in counts_for_hist.word_to_count.keys():
                        n_star_z = len(counts_for_hist.word_to_context[w])
                        counts_for_hist.word_to_f[w] = (
                            max((n_star_z - self.d[n]), 0) * 1.0 / n_star_star
                        )
                else:
                    for w in counts_for_hist.word_to_count.keys():
                        n_star_z = counts_for_hist.word_to_count[w]
                        counts_for_hist.word_to_f[w] = (
                            max((n_star_z - self.d[n]), 0)
                            * 1.0
                            / counts_for_hist.total_count
                        )

    def cal_bow(self):
        n = self.ngram_order - 1
        this_order_counts = self.counts[n]
        for _, counts_for_hist in this_order_counts.items():
            for w in counts_for_hist.word_to_count.keys():
                counts_for_hist.word_to_bow[w] = None

        for n in range(0, self.ngram_order - 1):
            this_order_counts = self.counts[n]
            for hist, counts_for_hist in this_order_counts.items():
                for w in counts_for_hist.word_to_count.keys():
                    if w == self.eos_symbol:
                        counts_for_hist.word_to_bow[w] = None
                    else:
                        a_ = hist + (w,)

                        assert len(a_) < self.ngram_order
                        assert a_ in self.counts[len(a_)].keys()

                        a_counts_for_hist = self.counts[len(a_)][a_]

                        sum_z1_f_a_z = 0
                        for u in a_counts_for_hist.word_to_count.keys():
                            sum_z1_f_a_z += a_counts_for_hist.word_to_f[u]

                        sum_z1_f_z = 0
                        lower_hist = a_[1:]
                        lower_counts_for_hist = self.counts[len(lower_hist)][lower_hist]
                        for u in a_counts_for_hist.word_to_count.keys():
                            sum_z1_f_z += lower_counts_for_hist.word_to_f[u]

                        if sum_z1_f_z < 1:
                            counts_for_hist.word_to_bow[w] = (1.0 - sum_z1_f_a_z) / (
                                1.0 - sum_z1_f_z
                            )
                        else:
                            counts_for_hist.word_to_bow[w] = None

    def print_as_arpa(
        self, fout=io.TextIOWrapper(sys.stdout.buffer, encoding=default_encoding)
    ):
        print("\\data\\", file=fout)
        for hist_len in range(self.ngram_order):
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

        for hist_len in range(self.ngram_order):
            print("\\{0}-grams:".format(hist_len + 1), file=fout)

            this_order_counts = self.counts[hist_len]
            for hist, counts_for_hist in this_order_counts.items():
                for word in counts_for_hist.word_to_count.keys():
                    ngram = hist + (word,)
                    prob = counts_for_hist.word_to_f[word]
                    bow = counts_for_hist.word_to_bow[word]

                    if prob == 0:
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
            while f.tell() < end:
                raw_line = f.readline()
                if not raw_line:
                    break
                pbar.update(len(raw_line))
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
        choices=[1, 2, 3, 4, 5, 6, 7],
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
