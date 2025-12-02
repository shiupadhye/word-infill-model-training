"""
Iterates through a corpus sentence-by-sentence and augments it for code-infilling
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.stats import bernoulli

def fim(corpus,fim_rate,spm_rate):
    augmented_contexts = []
    for line in corpus:
        line = re.sub(r"<eos>","",line)
        aug = bernoulli.rvs(fim_rate, size=1)[0]
        tokens = line.strip().split()[1:-1]
        num_tokens = len(tokens)
        if aug == 1 and num_tokens > 2:
            mid_idx = np.random.randint(1,num_tokens-1,1)[0]
            mid = tokens[mid_idx]
            pref = " ".join(tokens[:mid_idx])
            suf = " ".join(tokens[mid_idx+1:])
            spm = bernoulli.rvs(spm_rate, size=1)[0]
            if spm == 1:
                newline = "<eos>" + " " + "<suf>" + " " + suf + " " + "<pre>" + " " + pref + " " + "<mid>" + " " + mid + " " + "<eos>" + " "
                augmented_contexts.append(newline)
            else:
                newline = "<eos>" + " " + "<pre>" + " " + pref + " " + "<suf>" + " " + suf + " " + "<mid>" + " " + mid + " " + "<eos>" + " "
                augmented_contexts.append(newline)
        elif aug == 0:
            newline = "<eos>" + " " + " ".join(tokens) + " " + "<eos>"
            augmented_contexts.append(newline)
        
    return augmented_contexts


def main():
    # corpus to modify
    infile = "wikitext-2-v1/wikitext-2-v1_validation.txt"
    # output file
    outfile = "wikitext-2-v1_fim/wikitext-2-v1_validation.txt"
    # fim_rate = 1 (all sentences are augmented)
    fim_rate = 1.0
    # spm_rate = 0.5 (50% of the augmented sentences are <suf> <pre> <mid>)
    spm_rate = 0.5

    with open(infile,'r') as fin:
        corpus = fin.readlines()

    fim_contexts = fim(corpus,fim_rate,spm_rate)
    with open(outfile,'w') as fout:
        for context in fim_contexts:
            fout.write(context + "\n")


if __name__ == '__main__':
    main()

