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

ROOT = 'tokenizer_data'
train_file = 'candor.txt'
tokenizer_dir = 'tokenizer'
tokenizer_id = 'candor_tokenizer.json'

train_filepath = os.path.join(ROOT,train_file)

raw_dataset = datasets.load_dataset('text',data_files={'train': train_filepath})

def get_training_corpus():
    dataset = raw_dataset['train']
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples['text']

# tokenization
# define model (Word-level)
wl_tokenizer = Tokenizer(models.WordLevel())
# define normalizers
wl_tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.StripAccents()]
)
# define whitespace pre-tokenizer/splitter
wl_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit()]
)
# define special tokens
special_tokens = ['<unk>','<mask>','<pad>','<eos>','<pre>','<suf>','<mid>']

# define trainer
trainer = trainers.WordLevelTrainer(special_tokens=special_tokens)

# train tokenizer on training data
wl_tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)


# save tokenizer
if not os.path.isdir(tokenizer_dir):
	os.mkdir(tokenizer_dir)

tokenizer_path = os.path.join(tokenizer_dir,tokenizer_id)
wl_tokenizer.save(tokenizer_path)




