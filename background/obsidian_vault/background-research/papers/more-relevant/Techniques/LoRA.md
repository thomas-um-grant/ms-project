# LoRA: Low-Rank Adaptation of Large Language Models
## Reference

https://openreview.net/forum?id=nZeVKeeFYf9

## Summary

LoRA (Low-Rank Adaptation) introduces a parameter-efficient fine-tuning method for large pre-trained language models by:
- Freezing all pre-trained weights and injecting trainable low-rank matrices into each layer of the Transformer architecture.
- Allowing adaptation with a drastic reduction in trainable parameters (up to 10,000× fewer) and GPU memory usage (up to 3× less), without increasing inference latency.
- Achieving performance comparable to or better than full fine-tuning across a wide range of tasks and models (RoBERTa, DeBERTa, GPT-2, GPT-3).

## Notes

Their hypothesis is that the change in weights needed during adaptation is of low intrinsic rank, and full-rank weight updates are unnecessary. Proved this on Transformers, but seemed to be applicable to any architecture with dense layers.

Advantages:
- No inference overhead unlike adapters.
- No reduction in sequence length unlike prompt tuning.
- Better scalability with large models (e.g., GPT-3 175B).
- Composable with other adaptation methods like prefix tuning.
