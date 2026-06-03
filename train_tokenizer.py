import os
import torch
import datasets

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)


from transformers import PreTrainedTokenizerFast


ROOT = 'data/tokenizer_data'
CORPUS_FILE = 'wikitext-2-v1.txt'
TOKENIZER_DIR = 'tokenizer'
BASE_TOKENIZER = 'wikitext_tokenizer.json'

# if fine-tuning
NEW_CORPUS_FILE = 'candor.txt'
EXTENDED_TOKENIZER = 'wikitext_tokenizer_ext.json'

SPECIAL_TOKENS = ['<unk>', '<mask>', '<pad>', '<eos>', '<pre>', '<suf>', '<mid>']

def yield_batches(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]["text"]


# helper function for extending vocabulary
def get_oov_tokens(lines, existing_vocab):
    oov = set()
    for line in lines:
        for tok in line.split():
            if tok not in existing_vocab:
                oov.add(tok)
    return sorted(oov)


def train_tokenizer():
    corpus_path = os.path.join(ROOT, CORPUS_FILE)
    raw_dataset = datasets.load_dataset("text", data_files={"train": wikitext_path})
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
        min_frequency=1
    )

    tokenizer.train_from_iterator(
        yield_batches(train_data),
        trainer=trainer
    )

    # Save
    out_path = os.path.join(TOKENIZER_DIR, BASE_TOKENIZER)
    tokenizer.save(out_path)


def extend_tokenizer():
    base_path = os.path.join(TOKENIZER_DIR, BASE_TOKENIZER)
    new_path = os.path.join(TOKENIZER_DIR, EXTENDED_TOKENIZER)

    # Load existing tokenizer
    tokenizer = Tokenizer.from_file(base_path)
    existing_vocab = tokenizer.get_vocab().keys()


    with open(os.path.join(ROOT, NEW_CORPUS_FILE), "r") as f:
        new_lines - [line.strip() for line in f if line.strip()]
    new_tokens = get_oov_tokens(new_lines, existing_vocab)
    print(f"{len(new_tokens)} new tokens")

    # Append
    for tok in new_tokens:
        tokenizer.add_tokens([tok])

    print(f"New vocab size: {len(tokenizer.get_vocab())}")
    # Save
    tokenizer.save(new_path)



if __name__ == "__main__":
    FINETUNE = True
    if not FINETUNE:
        train_tokenizer()
    else:
        extend_tokenizer()

