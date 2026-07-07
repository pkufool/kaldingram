#!/usr/bin/env python3

"""Compute perplexity of an ARPA n-gram language model on held-out text."""

import argparse
import io
import math
import os
import sys

from tqdm import tqdm

from .prune import ArpaParser

default_encoding = "latin-1"

strip_chars = " \t\r\n"


def compute_ppl(lm, lines, verbose=0):
    """Compute perplexity statistics over a list of text lines.

    Args:
        lm: An Arpa model loaded via ArpaParser.
        lines: Iterable of raw text lines (one sentence per line).
        verbose: Print per-sentence stats at higher levels.

    Returns:
        A dict with keys: logprob, num_sentences, num_words, num_oov,
        num_zeroprobs, ppl, ppl1.
    """
    logprob = 0.0
    num_sentences = 0
    num_words = 0
    num_oov = 0
    num_zeroprobs = 0
    # Words/sentences that contributed to logprob (exclude zeroprobs).
    scored_words = 0
    scored_sentences = 0

    unk_token = lm._unk

    for lineno, line in enumerate(tqdm(lines, desc="scoring", unit="sent"), 1):
        line = line.strip(strip_chars)
        if not line:
            continue

        tokens = line.split()

        # Count OOVs (words not in the model vocabulary).
        for w in tokens:
            if w != unk_token and not lm.contains_word(w):
                num_oov += 1

        num_sentences += 1
        num_words += len(tokens)

        # Score the sentence.
        try:
            sent_logprob = lm.log_s(line)
        except KeyError:
            num_zeroprobs += 1
            if verbose >= 1:
                print(
                    "sentence {}: zeroprob".format(lineno),
                    file=sys.stderr,
                )
            continue

        logprob += sent_logprob
        scored_sentences += 1
        scored_words += len(tokens)

        if verbose >= 2:
            sent_ppl = 10 ** (-sent_logprob / max(len(tokens), 1))
            print(
                "sentence {}: words={} logprob={:.4f} ppl={:.4f}".format(
                    lineno, len(tokens), sent_logprob, sent_ppl
                ),
                file=sys.stderr,
            )

    if scored_sentences == 0:
        return {
            "logprob": 0.0,
            "num_sentences": num_sentences,
            "num_words": num_words,
            "num_oov": num_oov,
            "num_zeroprobs": num_zeroprobs,
            "ppl": float("inf"),
            "ppl1": float("inf"),
        }

    # ppl: denominator includes one </s> per sentence (SRILM convention).
    denom_with_eos = scored_words + scored_sentences
    # ppl1: denominator excludes </s>.
    denom_words_only = scored_words if scored_words > 0 else 1

    ppl = 10 ** (-logprob / denom_with_eos) if denom_with_eos > 0 else float("inf")
    ppl1 = 10 ** (-logprob / denom_words_only) if denom_words_only > 0 else float("inf")

    return {
        "logprob": logprob,
        "num_sentences": num_sentences,
        "num_words": num_words,
        "num_oov": num_oov,
        "num_zeroprobs": num_zeroprobs,
        "ppl": ppl,
        "ppl1": ppl1,
    }


def print_ppl_report(stats, source_label="stdin"):
    """Print SRILM-style PPL report to stdout."""
    print(
        "file {}: {} sentences, {} words, {} OOVs".format(
            source_label,
            stats["num_sentences"],
            stats["num_words"],
            stats["num_oov"],
        )
    )
    if stats["num_sentences"] == 0:
        print("0 zeroprobs")
        return

    print(
        "{} zeroprobs, logprob= {:.4f} ppl= {:.4f} ppl1= {:.4f}".format(
            stats["num_zeroprobs"],
            stats["logprob"],
            stats["ppl"],
            stats["ppl1"],
        )
    )


def add_arguments(parser):
    parser.add_argument(
        "-lm",
        "--lm",
        type=str,
        required=True,
        help="Input ARPA language model",
    )
    parser.add_argument(
        "-text",
        "--text",
        type=str,
        default=None,
        help="Test corpus (one sentence per line); reads from stdin if omitted",
    )
    parser.add_argument(
        "-encoding",
        "--encoding",
        type=str,
        default=default_encoding,
        help="Encoding of input files (default: %(default)s)",
    )
    parser.add_argument(
        "-verbose",
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help="Verbose level (default: %(default)s)",
    )


def run(args):
    parser = ArpaParser()
    models = parser.loadf(args.lm, encoding=args.encoding)
    lm = models[0]

    if args.text is not None:
        assert os.path.isfile(args.text), "File not found: {}".format(args.text)
        with open(args.text, encoding=args.encoding) as fp:
            lines = list(fp)
        source_label = args.text
    else:
        infile = io.TextIOWrapper(sys.stdin.buffer, encoding=args.encoding)
        lines = list(infile)
        source_label = "stdin"

    stats = compute_ppl(lm, lines, verbose=args.verbose)
    print_ppl_report(stats, source_label=source_label)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compute perplexity of an ARPA n-gram language model on held-out text. "
            "By default, reads test text from stdin."
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
