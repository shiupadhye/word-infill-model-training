"""
Train a GPT-2 language model from scratch on FIM-augmented text.
"""
import os
import argparse
 
import datasets
from transformers import (
    AutoConfig,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
 
 
def main():
    p = argparse.ArgumentParser(description="Train a GPT-2 LM from scratch on FIM data.")
    p.add_argument("--train", required=True, help="training corpus (one FIM sequence per line)")
    p.add_argument("--valid", required=True, help="validation corpus (must also be formatted in FIM style")
    p.add_argument("--tokenizer", required=True, help="tokenizer .json (must carry the FIM sentinels)")
    p.add_argument("--output-dir", required=True, help="where to save checkpoints")
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--gpu", default=None, help="set CUDA_VISIBLE_DEVICES, e.g. '0' (default: all visible)")
    p.add_argument("--wandb-project", default=None,
                   help="if set, log to this Weights & Biases project")
    args = p.parse_args()
 
    if args.gpu is not None:                      # must be set before CUDA initializes
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    for f in (args.train, args.valid, args.tokenizer):
        if not os.path.isfile(f):
            p.error(f"file not found: {f}")
 
    # load tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer,
        model_max_length=args.context_length,
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
    )
 
    # load datasets
    raw_dataset = datasets.load_dataset(
        "text", data_files={"train": args.train, "valid": args.valid})
 
    def tokenize(element):
        outputs = tokenizer(element["text"], truncation=True, max_length=args.context_length)
        return {"input_ids": outputs["input_ids"]}
 
    tokenized_datasets = raw_dataset.map(
        tokenize, batched=True, remove_columns=raw_dataset["train"].column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
 
    # init model from scratch
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=args.context_length,
        n_positions=args.context_length,       
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)

    print(f"Training GPT-2 from scratch: {model.num_parameters():,} params, vocab {len(tokenizer)}")
 
    # log to W&B (optional)
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        report_to, run_name = "wandb", "gpt2-infill-pretrain"
    else:
        report_to, run_name = "none", None
 
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=100,
        weight_decay=0.01,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        report_to=report_to,
        run_name=run_name,
    )
 
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
 
 
if __name__ == "__main__":
    main()
