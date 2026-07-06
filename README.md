# kaldingram

`kaldingram` provides Python and CLI tools to:

- train Kneser-Ney back-off n-gram language models in ARPA format
- entropy-prune ARPA language models

The implementation is based on Kaldi WSJ scripts and matches SRILM-style behavior.

## Install

```bash
pip install kaldingram
```

## CLI Usage

### Train an n-gram LM

```bash
kaldingram train --ngram-order 4 --text corpus.txt --lm 4gram.arpa
```

Or stream text from stdin and write ARPA to stdout:

```bash
cat corpus.txt | kaldingram train --ngram-order 3 > 3gram.arpa
```

### Prune an n-gram LM

```bash
kaldingram prune --threshold 1e-8 --lm 4gram.arpa --write-lm 4gram_pruned.arpa
```

### Evaluate perplexity

```bash
kaldingram ppl --lm 4gram.arpa --text test.txt
cat test.txt | kaldingram ppl --lm 4gram.arpa
```

## Development

Build package locally:

```bash
python -m pip install --upgrade build
python -m build
```

