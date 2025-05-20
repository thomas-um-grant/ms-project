# YaRN: Efficient Context Window Extension of Large Language Models
## Reference

https://openreview.net/forum?id=wHBfxhZu1u

## Summary

YaRN is a method for efficiently extending the context window of LLMs using Rotary Position Embeddings (RoPE). It surpasses previous RoPE interpolation approaches by:
- Achieving state-of-the-art long-context performance with up to 128k tokens.
- Requiring ~10× fewer tokens and ~2.5× fewer steps for fine-tuning compared to prior methods.
- Combining interpolation strategies (NTK-based) with attention scaling to preserve both relative and absolute positional information.
- Supporting zero-shot context extension via Dynamic Scaling at inference time.

## Notes

Novelty:
- NTK-by-Parts Interpolation:
	- Decomposes RoPE dimensions by their encoding behavior:
		- Low-frequency (long wavelengths): retain relative position -> no interpolation.
		- High-frequency (short wavelengths): encode absolute info -> apply interpolation.
	- Uses a ramp function gamma to smoothly interpolate based on the rotation ratio 
	- This strategy preserves useful inductive biases from RoPE.

- Apply attention scaling based on scale factor:
	- Adds a temperature term to the attention softmax
	- Reduces perplexity across long sequences, acting as length-aware entropy adjustment
	- Efficiently implemented by scaling RoPE frequencies — no inference overhead.

- Dynamic Scaling:
	- Dynamically updates the interpolation scale factor at inference time.
	- Works without any fine-tuning.
	- Provides graceful degradation when extrapolating to unseen lengths.

In addition:
Can also fine-tune on longer-context data (e.g., 64k tokens) for higher performance.