"""
script containing functions for computing various conditional probabilities from FIM-trained model/
"""
import math
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
 
 
CONTEXT_LENGTH = 1024
# load FIM-enabled tokenizer
tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        model_max_length=context_length,
        unk_token="<unk>", pad_token="<pad>", mask_token="<mask>", eos_token="<eos>",
    )

# load FIM-enabled model
model = GPT2LMHeadModel.from_pretrained(model_path)
model.to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()

 
 
@torch.no_grad()
def _logprob(tokenizer, model, text, target):
    """
    Compute log p(target | context <mid>) 
    """
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    logits = model(ids.to(model.device)).logits[0, -1]      
    logp = torch.log_softmax(logits, dim=-1)
    tid = tokenizer.convert_tokens_to_ids(target)            
    return logp[tid].item()
 
 
def _logmeanexp2(a, b):
    m = max(a, b)
    return m + math.log(0.5 * (math.exp(a - m) + math.exp(b - m)))
 
 
def score_prefixOnly(prefix, target, tokenizer, model):
    """
    Compute log p(target | <eos> <pre> prefix <mid>)
    """
    return _logprob(tokenizer, model, f"<eos> <pre> {prefix} <mid>", target)
 
 
def score_suffixOnly(suffix, target, tokenizer, model):
    """
    Compute log p(target | <eos> <suf> suffix <mid>)
    """
    return _logprob(tokenizer, model, f"<eos> <suf> {suffix} <mid>", target)
 
 
def score_bidirectional(prefix, suffix, target, tokenizer, model, order="psm"):
    """
    Bidirectional: log p(target | past, future).
      order="psm"      <pre> prefix <suf> suffix <mid>
      order="spm"      <suf> suffix <pre> prefix <mid>
      order="average"  RECOMMENDED: average the psm and spm probabilities
                        then take the log. Use this if the model was trained on a 50/50 mix of orderings.
    """
    psm = f"<eos> <pre> {prefix} <suf> {suffix} <mid>"
    spm = f"<eos> <suf> {suffix} <pre> {prefix} <mid>"
    if order == "psm":
        return _logprob(tokenizer, model, psm, target)
    if order == "spm":
        return _logprob(tokenizer, model, spm, target)
    if order == "average":
        return _logmeanexp2(_logprob(tokenizer, model, psm, target),
                            _logprob(tokenizer, model, spm, target))
    raise ValueError(f"order must be 'psm', 'spm', or 'average'; got {order!r}")
 
 
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Score a target word's predictability.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--target", required=True)
    ap.add_argument("--mode", choices=["prefix", "suffix", "bidirectional"], default="bidirectional")
    ap.add_argument("--order", choices=["psm", "spm", "average"], default="psm")
    args = ap.parse_args()
 
    tokenizer, model = load(args.model, args.tokenizer)
    if args.mode == "prefix":
        lp = score_prefixOnly(args.prefix, args.target, tokenizer, model)
    elif args.mode == "suffix":
        lp = score_suffixOnly(args.suffix, args.target, tokenizer, model)
    else:
        lp = score_bidirectional(args.prefix, args.suffix, args.target, tokenizer, model, order=args.order)
 