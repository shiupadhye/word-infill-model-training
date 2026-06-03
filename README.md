# word-infill-model-training
Requisite scripts for training and running inference on an infill-augmented GPT-2 based the fill-in-the-middle (FIM) approach originally proposed by [Bavarian et al. (2022)](https://arxiv.org/pdf/2207.14255) for code-infilling.

More details about the current implementation of a word-level FIM GPT-2 model are reported in [Upadhye \& Futrell (2025)](https://arxiv.org/pdf/2511.07752).

## Usage

### Augmenting data for training and inference.


### Training a tokenizer from scratch or including FIM tokens in existing tokenizer.


### Training an FIM-augmented GPT-2 from scratch


### Using the approach to estimate various conditional probabilities 

## Software
Existing infill-trained GPT-2 models can be found on Huggingface:

Trained on wikitext-v2: https://huggingface.co/shiupadhye/gpt2-small-infill-adapted-wikitext

Trained on CANDOR (speech): https://huggingface.co/shiupadhye/gpt2-small-infill-adapted-candor


