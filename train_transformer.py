import os
import wandb
import torch
import datasets
import evaluate
from transformers.integrations import WandbCallback

os.environ["CUDA_VISIBLE_DEVICES"]="0"

FINETUNE_MODEL = False

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoForCausalLM,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

context_length = 1024

# load tokenizer
tokenizer_dir = 'tokenizer'
tokenizer_pt = "wikitext_tokenizer.json"
tokenizer_path = os.path.join(tokenizer_dir,tokenizer_pt)
tokenizer = PreTrainedTokenizerFast(
	tokenizer_file=tokenizer_path,
	model_max_length = context_length,
    unk_token = '<unk>',
    pad_token = '<pad>',
    mask_token = '<mask>',
    eos_token = '<eos>')


def tokenize(element):
    outputs = tokenizer(
        element["text"],
        # truncate to max length
        truncation=True,
        # hard limit on length of sequences
        max_length=context_length,
        # pad to max length
        #padding='longest',
    )
    return {"input_ids": outputs['input_ids']}


# init collator for lm
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False)

# load dataset
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR,'data/wikitext-2-v1/fim')
train_file = 'wikitext-2-v1_train.txt'
valid_file = 'wikitext-2-v1_validation.txt'
train_filepath = os.path.join(ROOT,train_file)
valid_filepath = os.path.join(ROOT,valid_file)
raw_dataset = datasets.load_dataset('text', data_files={'train': train_filepath,'valid':valid_filepath})

# model path
model_path = "models/gpt2-wikitext/checkpoint-133777"
# Tokenize datasets
tokenized_datasets = raw_dataset.map(tokenize, batched=True, remove_columns=raw_dataset["train"].column_names)

if not FINETUNE_MODEL:
    # load LLM
    config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(tokenizer),
            n_ctx=context_length,
            bos_token_id = tokenizer.eos_token_id,
            eos_token_id = tokenizer.eos_token_id
    )


    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    print("Training model")

else:
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    print("Fine-tuning model")

wandb.init(project="infill-model", name="gpt2-infill-pretrain-wiki", mode="online")

training_args = TrainingArguments(
    output_dir="models/gpt2-wikitext",
    overwrite_output_dir=True,
    num_train_epochs=10,
    evaluation_strategy = "epoch",
    logging_strategy='steps',
    logging_steps=100,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    report_to="none",     
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['valid'],
    callbacks=[WandbCallback(), EarlyStoppingCallback(early_stopping_patience=2)]
)


trainer.train()
