"""
Train tokenizer for FIM training and inference, either by training a word-level tokenizer from scratch 
or injecting the FIM sentinels into an existing tokenizer (word-level OR BPE).
"""
import os
import argparse
import datasets
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer

 
# Single source of truth: the sentinels that make infilling work.
# Required tokens for infill inference 
FIM_TOKENS = ['<eos>', '<pre>', '<suf>', '<mid>']
 
# Full set of sentinel tokens (for training word-level tokenizer from scratch)
SPECIAL_TOKENS = ['<unk>', '<mask>', '<pad>'] + FIM_TOKENS
 
 
def yield_batches(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]["text"]
 
 
def train_tokenizer(corpus_path, out_path, min_frequency=1):
    """
    Train a word-level tokenizer from scratch, with normalization and min frequency
    threshold of 1 (at least one occurrence guarantees inclusion in vocabulary)
    """
    if not os.path.isfile(corpus_path):
        raise FileNotFoundError(f"corpus not found: {corpus_path}")
    raw_dataset = datasets.load_dataset("text", data_files={"train": corpus_path})
    train_data = raw_dataset["train"]
    # Build word-level tokenizer
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.StripAccents()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = trainers.WordLevelTrainer(
        special_tokens=SPECIAL_TOKENS,
        min_frequency=min_frequency
    )
    tokenizer.train_from_iterator(
        yield_batches(train_data),
        trainer=trainer
    )
    # Save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tokenizer.save(out_path)
    print(f"saved tokenizer ({tokenizer.get_vocab_size()} tokens) -> {out_path}")
 
 
def add_fim_tokens(base_path, out_path, tokens=FIM_TOKENS):
    """
    Inject the FIM sentinels into an EXISTING tokenizer that lacks them.
    Tokens are treated as special tokens (i.e., never split).
    """
    if not os.path.isfile(base_path):
        raise FileNotFoundError(f"base tokenizer not found: {base_path}")
    tokenizer = Tokenizer.from_file(base_path)
    vocab = tokenizer.get_vocab()
    to_add = [t for t in tokens if t not in vocab]
    already = [t for t in tokens if t in vocab]
    if already:
        print(f"Already present: {already}")
    n = tokenizer.add_special_tokens(to_add) if to_add else 0
    print(f"Added {n} special tokens: {to_add}")
    print(f"New vocab size: {len(tokenizer.get_vocab())}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tokenizer.save(out_path)
    print(f"saved to {out_path}")
    if n:
        print("NOTE: resize the model to match before training/inference")
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Train a word-level (whitespace delimited) tokenizer from scratch or add FIM tokens to an existing tokenizer.")
    sub = parser.add_subparsers(dest="mode", required=True)
 
    pt = sub.add_parser("train", help="train a tokenizer from scratch")
    pt.add_argument("--corpus", required=True, help="training corpus (one sentence/sequence per line)")
    pt.add_argument("--out", required=True, help="output tokenizer .json")
    pt.add_argument("--min-frequency", type=int, default=1, help="minimum token frequency to include in vocabulary")
 
    pa = sub.add_parser("add-specials",
                        help="inject the FIM sentinels into an existing tokenizer (no new vocab)")
    pa.add_argument("--base", required=True, help="base tokenizer .json")
    pa.add_argument("--out", required=True, help="output tokenizer .json")
    pa.add_argument("--tokens", nargs="+", default=FIM_TOKENS,
                    help="special tokens to inject if missing")
 
    args = parser.parse_args()
    if args.mode == "train":
        train_tokenizer(args.corpus, args.out, min_frequency=args.min_frequency)
    else:  # add-specials
        add_fim_tokens(args.base, args.out, tokens=args.tokens)
 
 
if __name__ == "__main__":
    main()

