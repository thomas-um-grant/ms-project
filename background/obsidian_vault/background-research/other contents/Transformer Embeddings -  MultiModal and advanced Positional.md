
Medium - Feb 6, 2025
https://medium.com/@sulbha.jindal/transformer-embeddings-multi-modal-and-advanced-positional-440d2d50ee89

Embeddings are concise vector representations of data that encapsulate semantic meaning, enabling comparability and cross-modal connections. These compact representations enhance AI performance in various tasks, including search and classification. This blog post will explore two specific types of embeddings: **multi-modal embeddings**, which can represent and link different types of data (such as text and images), and **advanced positional embeddings**, which capture the relative positions of elements in sequential data, improving performance in tasks like natural language processing.

# Multi-Modal Embeddings (MME)

Multi-Modal Embeddings (MME) enable systems to process and integrate information from diverse data types, such as text, audio, and video. This approach addresses several limitations of traditional single-modality embeddings. MME caters to real-world applications like AI assistants and recommendation systems that require integrating multiple data sources. It enables cross-modal tasks such as image-text search and video captioning, which vanilla embeddings struggle with. MME also tackles semantic misalignment by effectively aligning different structures like pixels and words. Unlike single-modality embeddings, MME facilitates cross-modal linking and provides context by connecting text with images, such as captions to photos. These capabilities make MME a powerful tool for handling complex, multi-faceted data in modern AI applications.

MME enables various applications, including: Image-text search and video captioning, Personalization and contextual search, Cross-modal tasks like searching images by text or vice versa, Brain signal alignment with images and text for potential brain-computer interfaces.

## How do MME work ?

Multi-Modal Embeddings (MME) work through a process that integrates information from various data sources.

1. Data Input: MME systems accept data from multiple sources, including text, images, and audio.

2. Separate Encodings: Each modality is processed independently using specialized models. For example, a text encoder handles text data, while an image encoder processes visual information.

3. Feature Extraction: The encoders convert the input data into numerical embeddings (vectors) that capture key features of each modality.

4. Alignment: A shared semantic space is created where embeddings from different modalities are mapped. Semantically similar concepts across modalities are positioned closer together in this space.

5. Fusion: The embeddings are combined into a single representation using various techniques:

- Early Fusion: Combines modalities immediately after embedding creation.
- Intermediate Fusion: Extracts features from each modality to some extent before fusing.
- Late Fusion: Processes each modality independently before combining the final outputs

Fusion methods include:

- Cross-Attention-based methods: Use transformer architecture to understand relationships between embeddings. uses attention layers to relate modalities, perfect for tasks like visual Q&A with fine grained relationship.
- Concatenation: Merges multiple embeddings into a single feature representation.
- Dot-product: Involves element-wise multiplication of feature vectors from different modalities.
- Contrastive learning: Trains embeddings to align related pairs and separate unrelated ones, ideal for retrieval tasks but requires large paired datasets.
- Multimodal transformers: Advanced models with attention layers integrate modalities, offering top results but needing high computational power.

This integrated representation enables the system to make decisions and predictions based on the combined information from all modalities.

## Advantages

1. Performance on Benchmarks — comparable to image-text retrieval benchmarks and superior on complex benchmarks like long-form text-image retrieval.
2. Fine grained understanding — enables region-level captioning and detailed retrieval tasks without additional training
3. Efficiency — the unified architecture reduces computational overhead while maintaining task specific performance.

## Challenges

1. Computational costs — processing and integrating multiple modalities with techniques like transformers demand s significant computational power and memory
2. Modality-specific Noise — Different modalities can have varying levels of quality or noise (e.g blurry images or incomplete texts) making it hard to create accurate embeddings
3. Interpretability — understanding how the model integrates and prioritizes information from different modalities is difficult, making debugging and improvement harder.

By creating a unified representation of multi-modal data, MME significantly enhances AI performance in tasks requiring integration of diverse information types.

# Advanced positional embeddings

Positional embeddings are crucial mechanisms in sequential data processing, particularly in natural language tasks, as they encode the order of tokens in a sequence. This encoding allows models to maintain awareness of the sequence’s structure, which is essential for understanding context and relationships between elements. Advanced positional embeddings have evolved beyond the original approach used in the Transformer architecture, enhancing the model’s ability to handle long sequences, capture hierarchical structures, and adapt to various data types. These improvements enable more nuanced understanding of positional relationships, leading to better performance in tasks requiring fine-grained sequence comprehension and generation.

## 1. Absolute Positional Embedding

Absolute Position Encodings are a crucial component of Transformer-based models, designed to provide positional information to the otherwise position-agnostic attention mechanism. These encodings are added to input embeddings in both the encoder and decoder sections of the mode. In practice, we create a vector for each position in the sequence.

![](https://miro.medium.com/v2/resize:fit:916/1*5MD6UxKqrCTrxayxYCfjbQ.png)

Where _pi_ is a _d_-dimensional vector, representing the absolute position of token _xi_. Sinusoidal positional encoding and learned positional encoding are two alternatives to generate _pi_.

![](https://miro.medium.com/v2/resize:fit:672/1*lqEYxnx6E74jMoaAmYkglw.png)

**Sinusoidal Positional Encoding** — The encodings are created using sine and cosine functions of varying frequencies, with each dimension corresponding to a sinusoid as:

![](https://miro.medium.com/v2/resize:fit:812/1*HUHFvPf9N92lumcg26sXBw.png)

Where _pos_ is the position of the token in the sequence, _d_ is the position embedding dimension, and i is the dimension index (_0<=i<d_).

Sinusoidal positional encoding, which employs sine and cosine functions, has a profound connection to the Fourier transform, enabling models to capture both local and global positional information. The **high-frequency** components, corresponding to lower values of i in the encoding formula, allow the model to understand local relationships between neighboring tokens, such as word pairs in a sentence. Conversely, **low-frequency** components, associated with higher values of i, enable the model to grasp more global patterns across the entire sequence. This dual capability is crucial for comprehending both immediate contextual relationships and long-range dependencies, such as connections between words in different sentences. By incorporating both high and low-frequency elements, the encoding provides a rich representation of position that supports the model’s ability to process complex sequential data effectively.

**Learned Positional Encoding** — Learned Positional Encoding is an alternative approach to representing token positions in sequence models. In this method, each position in the sequence is assigned a unique embedding vector that is learned during the training process alongside other model parameters. For instance, in a model with a context length of 512 and a token embedding size of 768, a learnable tensor of size 512x768 is added to the model’s trainable parameters.

## Get Sulbha Jain’s stories in your inbox

Join Medium for free to get updates from this writer.

Subscribe

This approach allows the model to adaptively learn the most effective way to encode positional information for its specific task, such as text classification or translation. Learned positional embeddings offer greater expressiveness compared to sinusoidal encodings, as they can be tailored to the particular requirements of the task at hand. However, this flexibility comes at the cost of increased model size and computational complexity due to the additional trainable parameters. Despite this trade-off, learned positional encodings have been successfully implemented in prominent models like BERT and GPT, demonstrating their effectiveness in various natural language processing tasks.

## 2. Relative Positional Embeddings

Relative positional embeddings represent a sophisticated approach to encoding token positions in sequence models. Unlike absolute positional encodings, they focus on the relationships between tokens in terms of their relative distances, rather than their exact positions in the sequence.

![](https://miro.medium.com/v2/resize:fit:942/1*W1lUzeOFuUeUQ9jB_6VIvg.png)

This approach aligns well with the attention mechanism’s operation, which computes the importance of other tokens for each specific token based on their relative positions. The relative distance between positions m and n is typically represented as r = clip(m-n, Rmin, Rmax), where the maximum relative position is clipped. This clipping assumes that precise relative positioning becomes less crucial beyond a certain distance and enables the model to generalize to sequence lengths not encountered during training.

Relative positional embeddings offer several advantages:

1. They handle longer sequences more effectively.  
2. They generalize better to sequence lengths not seen during training.  
3. They capture relational information between tokens more directly.

However, this approach may miss some useful information from the absolute position of tokens, such as the position of the first token in a sequence.

Relative positional embeddings have been successfully implemented in various models, including Transformer-XL and T5, demonstrating their effectiveness in improving model performance on tasks involving long sequences or requiring generalization to different sequence lengths.

## 3. Rotatory Positional Embedding (RoPE)

Rotary Position Embedding (RoPE) is an innovative approach to incorporating positional information in transformer models. It utilizes a rotation matrix to encode absolute positions of tokens while preserving relative positional relationships in self-attention formulations. The key idea behind RoPE is to rotate word vectors in a high-dimensional space based on their positions in the sequence, combining benefits of both absolute and relative embeddings.RoPE modifies the attention weight computation at every layer instead of adding position information to token embeddings as:

![](https://miro.medium.com/v2/resize:fit:950/1*PSDxdpjSY9XScBTSgsYgug.png)

This approach offers several advantages:

1. Effective handling of longer sequences compared to absolute positional embeddings  
2. Natural incorporation of both absolute and relative position information  
3. Computational efficiency and ease of implementation

It’s worth noting that RoPE doesn’t add position information to the values in the attention module. The output of the attention module is a weighted sum of the value vector, which means the outputs of each transformer layer don’t have explicit position details.

Due to its effectiveness, RoPE has been adopted by popular language models such as LLaMA and GPT-NeoX. Its ability to combine the strengths of absolute and relative positional embeddings while addressing their respective limitations has made it a significant advancement in transformer architecture.

## 4. Attention with Linear Biases (ALiBi)

Attention with Linear Biases (ALiBi) modifies the way attention scores are computed in the attention sublayer of the network. ALiBi introduces a static, non-learned bias after the query-key dot product during the computation of attention scores. This bias is added in the form of a head-specific slope that is determined before training, creating a geometric sequence of slopes for the different heads in the model. The method has an inductive bias towards recency, penalizing attention scores between distant query-key pairs with the penalty increasing as the distance grows, and it leverages different rates of penalty increase across different heads based on the slope magnitude. It adds a penalty to attention weight scores that is proportional to the distance between tokens. Therefore, the attention score between two tokens i and j at every layer is calculated as:

_Attention score = query_i . key_j — m.(i-j)_

Where _-m.(i-j)_ is a penalty which is proportional to the distance between token _i_ and _j_. The scalar _m_ is a head-specific slope fixed before training and its values for different heads are chosen as a geometric sequence. For example, for 8 head, _m_ might be:

![](https://miro.medium.com/v2/resize:fit:256/0*xfzB3E7SCtnPCIBk.png)

This means, the first head has a relatively large _m_ so it penalizes far apart tokens more and focuses on recent tokens, while the 8th head has the smallest _m_, allowing it to attend to more distant tokens.

## 5. Kernelized Relative Positional Embedding (Kerple)

Kernelized Relative Positional Embedding for Length Extrapolation (KERPLE) is an innovative framework that enhances relative positional embeddings (RPE) in transformer models. Kernelized Relative Positional Embedding for Length Extrapolation (KERPLE) generalizes relative positional embeddings (RPE) by kernelizing positional differences using conditionally positive definite (CPD) kernels known for generalizing distance metrics. They transform CPD kernels into positive definite (PD) kernels by adding a constant offset, which is absorbed during softmax normalization in the self-attention mechanism of transformers. This approach allows for a variety of RPEs that facilitate length extrapolation in a principled manner. KERPLE represents a significant advancement in positional embedding techniques, offering improved performance and flexibility for transformer models in handling variable-length sequences.

# Limitations of Positional Embeddings

Positional embeddings in transformer models face several limitations, particularly when it comes to extrapolation — the ability to handle sequences longer than those seen during training. While transformers are theoretically agnostic to input length, their performance often degrades with longer sequences due to the constraints of their positional encoding methods.Different positional embedding techniques show varying capabilities for extrapolation:

1. Learned position embeddings have no extrapolation ability beyond their training length.
2. Sinusoidal embeddings show very limited practical extrapolation capabilities.
3. Rotary Position Embedding (RoPE) outperforms sinusoidal embeddings but still struggles with longer sequences.
4. T5 bias method (a form of relative position embedding) offers better extrapolation but is computationally expensive.
5. Attention with Linear Biases (ALiBi) demonstrates superior extrapolation performance with minimal memory increase.

To address these limitations, new methods like Position Interpolation (PI) have been introduced. PI aims to extend the context window sizes of RoPE-based models by reducing position indices to align with the initial context window size through interpolation.

The authors of [ALiBi](https://arxiv.org/pdf/2108.12409) demonstrated that the bottleneck for transformer extrapolation is its position embedding method. As shown in Fig. 3, they compared the extrapolation capabilities of different position embedding methods. Since learned position embedding does not have a capability to encode positions greater than the training length, it has no extrapolation ability.

![](https://miro.medium.com/v2/resize:fit:934/0*huEYvt6eiKX-mq1N.png)

Fig 3: Extrapolation: as the input sequence gets longer (x-axis), [sinusoidal](https://user.phil.hhu.de/~cwurm/wp-content/uploads/2020/01/7181-attention-is-all-you-need.pdf), [RoPE](https://arxiv.org/pdf/2104.09864), and [T5](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf) position encodings show degraded perplexity (y-axis, lower is better), while [ALiBi](https://arxiv.org/pdf/2108.12409) does not (image from [paper](https://arxiv.org/pdf/2108.12409)).

Fig. 3 shows that the sinusoidal position embedding in practice has very limited extrapolation capabilities. While [RoPE](https://arxiv.org/pdf/2104.09864) outperforms the sinusoidal one, it still does not achieve satisfactory results. The [T5](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf) bias method (a version of relative position embedding) leads to better extrapolation than both sinusoidal and [RoPE](https://arxiv.org/pdf/2104.09864) embedding. Unfortunately, the [T5](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf) bias is computationally expensive (Fig. 4). [ALiBi](https://arxiv.org/pdf/2108.12409) outperforms all these position embeddings with negligible (0–0.7%) memory increase.

![](https://miro.medium.com/v2/resize:fit:934/0*DpGFTzeb-Q5a5JU1.png)

Fig. 4: comparison of batched training, inference speed and memory use of [sinusoidal](https://user.phil.hhu.de/~cwurm/wp-content/uploads/2020/01/7181-attention-is-all-you-need.pdf), [RoPE](https://arxiv.org/pdf/2104.09864), [T5](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf), and [ALiBi](https://arxiv.org/pdf/2108.12409) position encodings (image from [paper](https://arxiv.org/pdf/2108.12409)).

The choice of positional embedding method significantly impacts a model’s ability to handle longer sequences at inference time, with ALiBi currently showing the most promising results in terms of balancing performance and computational efficiency.

# Conclusion

The method of encoding positional information in Transformer architectures plays a crucial role in their ability to process sequential data and extrapolate to longer sequences during inference. Traditional absolute positional embedding techniques, while providing positional awareness, often face challenges when dealing with sequences longer than those encountered during training. This limitation has spurred the development of more advanced positional encoding methods. Newer approaches such as relative position encoding, Rotary Position Embedding (RoPE), and Attention with Linear Biases (ALiBi) have demonstrated improved capabilities in handling longer sequences at inference time. As Transformers become increasingly prevalent across various applications, ongoing refinement of positional encoding techniques remains essential for pushing the boundaries of their performance and adaptability to diverse sequence lengths and tasks.

## Appendix

1. MME Source: Bhavishya Pandit
2. [https://towardsdatascience.com/beyond-attention-how-advanced-positional-embedding-methods-improve-upon-the-original-transformers-90380b74d324](https://towardsdatascience.com/beyond-attention-how-advanced-positional-embedding-methods-improve-upon-the-original-transformers-90380b74d324)
3. [https://medium.com/towards-data-science/understanding-positional-embeddings-in-transformers-from-absolute-to-rotary-31c082e16b26](https://medium.com/towards-data-science/understanding-positional-embeddings-in-transformers-from-absolute-to-rotary-31c082e16b26)
4. [https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/nlp/nemo_megatron/positional_embeddings.html](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/nlp/nemo_megatron/positional_embeddings.html)
5. [https://medium.com/towards-data-science/understanding-positional-embeddings-in-transformers-from-absolute-to-rotary-31c082e16b26](https://medium.com/towards-data-science/understanding-positional-embeddings-in-transformers-from-absolute-to-rotary-31c082e16b26)