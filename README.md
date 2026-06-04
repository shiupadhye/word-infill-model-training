# word-infill-LM
Requisite scripts for training and running inference on an infill-trained GPT-2 based the fill-in-the-middle (FIM) approach originally proposed by [Bavarian et al. (2022)](https://arxiv.org/pdf/2207.14255) for code-infilling. More details about the current implementation of a word-level FIM GPT-2 model are reported in [Upadhye \& Futrell (2025)](https://arxiv.org/pdf/2511.07752).

## Usage

### Reformulating data for enabling FIM-training and inference
To enable an autoregressive model such as GPT-2 to have access to observed bidirectional context, the training corpus should be reformulated as follows:

<img width="8269" height="5710" alt="augmentation" src="https://github.com/user-attachments/assets/6356d3b1-7b53-447a-b0a5-771a9be1dc66" />

Run the ```prepare_fim_data.py``` script with your training and/or evaluation data as inputs to generate their FIM variants. Note that the script requires the data to be stored in a .txt file, with each line representing a sequence/sentence that begins and ends with an \<eos\> tag.

```
python prepare_fim_data.py \
    --infile  [path to existing corpus file].txt \
    --outfile [path to modifed corpus file].txt \
    --fim-rate 1.0 \
    --spm-rate 0.5 \
    --seed 0

```

The ```fim_rate``` parameter determines the proportion of sequences that need to be reformulated. An ```fim_rate = 1``` reformulates every sequence in the corpus. For a given sequence with N words, each token has a uniform probability (1/N) of being selected as the 'middle' token and transposed to the end of the sequence.  
The ```spm_rate``` parameters assumes values between 0 and 1 and determines the proportion of reformulated sequences that should be ordered suffix-first (e.g.,\<eos\> \<suf\> over the lazy dog \<pre\> the quick brown fox \<mid\> jumps \<eos\>). For example, ```spm_rate = 0.5```, reformulates 50\% of the sequences in a suffix-first order. More details about the algorithmic implementation can be found in [Upadhye \& Futrell (2025)](https://arxiv.org/pdf/2511.07752).


### Training a tokenizer from scratch or including FIM tokens in existing tokenizer
Using the FIM-enabled GPT-2 model requires that the tokenizer vocabulary include the \<pre\>, \<suf\>, and \<mid\> sentinel tokens. This can be done by:

1. Training a tokenizer from scratch:
   
```
python train_tokenizer.py train \

--corpus [path to corpus file].txt \

--out    [path to saved tokenizer file].json
```

This command trains a word-level (whitespace-delimited tokenizer). Note that a word-level tokenizer is not required, as the training and inference can also be conducted using standard byte-pair encoding (BPE) tokenizer, so long as the aforementioned sentinels are included. This can be done by extending the vocabulary of an existing tokenizer as follows:

2. Extending an existing tokenizer:
   
```
python train_tokenizer.py add-specials \

--base [path to existing tokenizer] \

--out  [path to saved tokenizer file].json
```


### Training an FIM-enabled GPT-2 from scratch
Below is the command for pre-training a GPT-2 small transformer model on the augmented training data. 

```
python train_model.py \
--train     [path to training file].txt \
--valid     [path to validation file].txt \
--tokenizer [path to tokenizer].json \
--output-dir [directory where the checkpoints should be saved]
```

Note that depending on the ```fim_rate``` for the training data, the validation data may also need to be similarily modified to ensure that the model encounters sequences that are structured similarly to those it was trained on -- a mis-match in the formatting may produce lower than expected validation perplexity. For example, if only 50\% of the sequences in the training data were reformulated, it is recommended that the validation data also contain a similar proportion of reformulated sequences. Other hyperparameters such as number of epochs, logging steps, learning rate, weights and biases logging etc. can be modified directly in the script.


### Using the approach to estimate various conditional probabilities 

<img width="8631" height="4116" alt="inference" src="https://github.com/user-attachments/assets/497cb75b-dc90-44ce-be3a-55dd1b0cca59" />

The scoring functions for estimating the (log) probability of a word given prefix only, suffix only, and bidirectional context are provided in ```fim_inference.py```. Bidirectional probabilities, in particular, can be estimated by reformulating the input in the (i) the prefix-suffix-mid (PSM) order, (ii) suffix-prefix-mid (SPM) order, or (iii) computing the average of the two orders. If the training data were reformulated such that augmented sequences only showed a PSM order (i.e., when ```spm_rate = 0```), option (i) should suffice. However, if the training data consisted of both orderings, it is recommended to use option (iii). Below is an example command for scoring using the command line:

```
# bidirectional (uses both prefix and suffix)
python score.py \
    --model     models/gpt2-candor/checkpoint-662895 \
    --tokenizer tokenizer/candor_tokenizer.json \
    --prefix    "it's not against the law to" \
    --suffix    "alligators through the mail" \
    --target    send \
    --mode      bidirectional \
    --order     average
```

## Software
Existing infill-trained GPT-2 models can be found on Huggingface:

Trained on wikitext-v2: https://huggingface.co/shiupadhye/gpt2-small-infill-adapted-wikitext

Trained on CANDOR (speech): https://huggingface.co/shiupadhye/gpt2-small-infill-adapted-candor


