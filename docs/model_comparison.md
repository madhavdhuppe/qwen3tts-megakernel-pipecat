# Model Comparison

| Parameter | AlpinDale Qwen3-0.6B Megakernel | Qwen3-TTS Talker Decoder |
|---|---:|---:|
| hidden_size | 1024 | 1024 |
| intermediate_size | 3072 | 3072 |
| num_hidden_layers | 28 | 28 |
| num_attention_heads | 16 | 16 |
| num_key_value_heads | 8 | 8 |
| head_dim | 128 | 128 |
| decode vocab | 151936 text tokens | 3072 codec tokens |
| rope_theta | 10000 | 1000000 |
| dtype | bfloat16 | bfloat16 |

The dimensions match closely enough to reuse the persistent RTX 5090 kernel for
the Qwen3-TTS talker decoder. The important TTS-specific changes are:

- Compile the LM projection for a 3072-row codec vocabulary.
- Use the talker `codec_head.weight`, which is separate from the codec embedding table.
- Add the embedding sentinel path so Python can pass precomputed sums of text and codec embeddings.
- Keep the code predictor and vocoder in the TTS engine so the server and Pipecat service stream PCM audio chunks.
