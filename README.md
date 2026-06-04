# word-infill-model-training
Requisite scripts for training and running inference on an infill-augmented GPT-2 based the fill-in-the-middle (FIM) approach originally proposed by [Bavarian et al. (2022)](https://arxiv.org/pdf/2207.14255) for code-infilling. More details about the current implementation of a word-level FIM GPT-2 model are reported in [Upadhye \& Futrell (2025)](https://arxiv.org/pdf/2511.07752).

## Usage

### Augmenting data for training and inference
<img width="8269" height="5710" alt="augmentation" src="https://github.com/user-attachments/assets/6356d3b1-7b53-447a-b0a5-771a9be1dc66" />

Run the ```data_augmentation.py``` script on your training and/or evaluation corpora for training and validating the model.

```
python data_augmentation.py \
    --infile  data/wikitext-2-v1/wikitext-2-v1_validation.txt \
    --outfile data/wikitext-2-v1/fim/wikitext-2-v1_validation.txt \
    --fim-rate 1.0 \
    --spm-rate 0.5 \
    --seed 0

```

The ```fim_rate``` parameter determines the proportion of sequences that need to be augmented. An ```fim_rate = 1``` augments every sequence in th corpus.  The ```spm_rate``` parameters assumes values between 0 and 1 and determines the proportion of augmented sequences that should be ordered suffix-first (e.g.,\<eos\> \<suf\> over the lazy dog \<pre\> the quick brown fox \<mid\> jumps \<eos\>). For example, ```spm_rate = 0.5```, augments 50\% of the sequences in a suffix-first order. 


### Training a tokenizer from scratch or including FIM tokens in existing tokenizer
Using the FIM-augmented GPT-2 model requires that the tokenizer vocabulary include the \<pre\>, \<suf\>, and \<mid\> sentinel tokens. This can be done by:

1. Training a tokenizer from scratch:
   
```
`python train_tokenizer.py train \`

`--corpus [path to corpus file].txt \`

`--out    [path to saved tokenizer file].json`
```

This command trains a word-level (whitespace-delimited tokenizer). Note that a word-level tokenizer is not required, as the training and inference can also be conducted using standard byte-pair encoding (BPE) tokenizer, so long as the aforementioned sentinels are included. This can be done by extending the vocabulary of an existing tokenizer as follows:

```
`python train_tokenizer.py add-specials \`

`--base [path to existing tokenizer] \`

`--out  [path to saved tokenizer file].json`
```


### Training an FIM-augmented GPT-2 from scratch


### Using the approach to estimate various conditional probabilities 

## Software
Existing infill-trained GPT-2 models can be found on Huggingface:

Trained on wikitext-v2: https://huggingface.co/shiupadhye/gpt2-small-infill-adapted-wikitext

Trained on CANDOR (speech): https://huggingface.co/shiupadhye/gpt2-small-infill-adapted-candor


