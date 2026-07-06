#!/usr/bin/env python3

"""Top-level CLI for kaldingram."""

import argparse

from . import __version__
from . import ppl as ppl_cmd
from . import prune as prune_cmd
from . import train as train_cmd


def build_parser():
    parser = argparse.ArgumentParser(
        prog="kaldingram",
        description="Train, prune, and evaluate n-gram language models.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train a Kneser-Ney n-gram model",
    )
    train_cmd.add_arguments(train_parser)
    train_parser.set_defaults(func=train_cmd.run)

    prune_parser = subparsers.add_parser(
        "prune",
        help="Entropy-prune an ARPA n-gram model",
    )
    prune_cmd.add_arguments(prune_parser)
    prune_parser.set_defaults(func=prune_cmd.run)

    ppl_parser = subparsers.add_parser(
        "ppl",
        help="Compute perplexity of an ARPA model on held-out text",
    )
    ppl_cmd.add_arguments(ppl_parser)
    ppl_parser.set_defaults(func=ppl_cmd.run)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
