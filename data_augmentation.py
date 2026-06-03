"""
Iterates through a corpus, treating each sentence as a sequence, and augments it for code-infilling.
Original input: the quick brown fox jumped over the fence
Augmented input:
Prefix Suffix Mid (PSM) order: <eos> <pre> the quick brown fox <suf> over the fence <mid> jumped <eos>
Suffix Prefix Mid (SPM) order: <eos> <suf> over the fence <pre> the quick brown fox <mid> jumped <eos>

args:
infile: path to original corpus (in .txt format)
outfile: path and filename of augmented corpus
fim_rate: proportion of sequences to augment in the corpus, with valid values ranging from 0 to 1.0.
spm_rate: proportion of augmented sequences (as determined by fim_rate) to be augmented in the SPM order
"""

import os
import re
import argparse
import numpy as np
from scipy.stats import bernoulli
 
 
def fim(corpus, fim_rate, spm_rate):
    augmented_contexts = []
    for line in corpus:
        line = re.sub(r"<eos>", "", line)
        aug = bernoulli.rvs(fim_rate, size=1)[0]
        tokens = line.strip().split()                # was [1:-1] -> that dropped the
        num_tokens = len(tokens)                      # first & last real words (the
                                                      # <eos> are already gone above)
        if aug == 1 and num_tokens > 2:
            mid_idx = np.random.randint(1, num_tokens - 1, 1)[0]
            mid = tokens[mid_idx]
            pref = " ".join(tokens[:mid_idx])
            suf = " ".join(tokens[mid_idx + 1:])
            spm = bernoulli.rvs(spm_rate, size=1)[0]
            if spm == 1:
                newline = "<eos>" + " " + "<suf>" + " " + suf + " " + "<pre>" + " " + pref + " " + "<mid>" + " " + mid + " " + "<eos>" + " "
                augmented_contexts.append(newline)
            else:
                newline = "<eos>" + " " + "<pre>" + " " + pref + " " + "<suf>" + " " + suf + " " + "<mid>" + " " + mid + " " + "<eos>" + " "
                augmented_contexts.append(newline)
        else:                                         # was `elif aug == 0:` -> short
            newline = "<eos>" + " " + " ".join(tokens) + " " + "<eos>"   # augmented
            augmented_contexts.append(newline)        # lines were silently dropped
 
    return augmented_contexts
 
 
def main():
    parser = argparse.ArgumentParser(description="Augment a corpus for FIM training.")
    parser.add_argument("--infile", required=True, help="corpus to modify")
    parser.add_argument("--outfile", required=True, help="output file")
    parser.add_argument("--fim-rate", type=float, default=1.0,
                        help="fraction of sentences that are augmented")
    parser.add_argument("--spm-rate", type=float, default=0.5,
                        help="fraction of augmented sentences in <suf> <pre> <mid> order")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    args = parser.parse_args()
 
    if not os.path.isfile(args.infile):              # fail fast with a clear message
        parser.error(f"input file not found: {args.infile}")
    out_dir = os.path.dirname(args.outfile)
    if out_dir:                                       # create output dir if needed
        os.makedirs(out_dir, exist_ok=True)
 
    np.random.seed(args.seed)                         # seeds both np.random and scipy.bernoulli
 
    with open(args.infile, 'r') as fin:
        corpus = fin.readlines()
    fim_contexts = fim(corpus, args.fim_rate, args.spm_rate)
    with open(args.outfile, 'w') as fout:
        for context in fim_contexts:
            fout.write(context + "\n")
 
 
if __name__ == '__main__':
    main()