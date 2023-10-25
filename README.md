# Language Models From Dummies

This is a series of introductory tutorials about language models (such as GPT).

### Description

A technical introduction to the topic of language models (LLMs), explaining the basic concepts, and creating sample code.

In this tutorials we fully implement a language model from scratch; including all the tools to train and use the model.

### Audience

The audience is assumed to be technical, but non-experts on neural networks.

We assume coding skills (Python), basic understanding of what a neural network is, what “back-propagation” is.
If you don’t remember the concepts, a quick brush up before the tutorials should be enough.

You are NOT assumed to be an expert practitioner on neural nets.

## Tutorial

- LLMs from dummies - part 1
  - [Video]()
  - [Slides](./slides/LLMs_from_dummies_part_1.pdf)
  - [Notebook](./notebooks/LLMs_from_dummies_Part_1.ipynb)
- LLMs from dummies - part 2
  - [Video]()
  - [Slides](./slides/LLMs_from_dummies_part_2.pdf)
  - [Notebook]()

## Topics

**Part 1:**
- A simple language translation program
  - Tokens, dictionaries
  - Approximating keys: Levengstein distance
- A "Machine Learning" approach
  - Encoding: One-Hot encoding,
  - Vocabularies
  - HashMap as a matrix multiplication
  - Relaxing "one hot"
  - Cosine similarity, SoftMax
  - Attention Mechanism
- Tokenization: BPE (analogy to compression)
- Embeddings (Word2vect, Autoencoder)

**Part 2:**
- Brief recap of Part 1
- Language model (moving away from the "translation" example)
- How the model works 
  - Predicting the next token
  - Blocks and data flow
  - Nomenclature: `block_size`, `n_embed`, etc.
- Language Model 1: Putting it all toghether
  - Model training: Boilerplate code
  - PyTorch implementation
  - Trivial tokenizer
  - Creating a Dataset
  - Trainig loop
  - Predicting
  - Model performance: What went wrong?
- Attention Revisited
  - Problem: "The model is cheating"
  - Masked attention
  - Softmax masking trick
- Masked Attention: Impreoved model
- Multi-Headed Attention
- Possitional Encoding
- Language Model 2
- Feed-Forward layer: ReLU layer
- Transformer Blocks
- Trainign deep networks
  - Vanishing & Exploding Gradients
  - Residual connections
  - Layer Normalization
  - Dropout
- Transformer
   - Encoders stack / Decoder stack
   - Contextual embeddings
   - Transformers

**Part 3:**
- Pre-training
- Training
- Fine-tunning
- Adapting: LORA, PERF methodologies
- Instruct models
- Prompts all the way down
   - `Chat`: Just isolated requests with "memory" of conversations
   - `Context`: Just add a sentence to the prompt
   - `Prompt engineering`: Similar to adding the right words in a Google search

**Part 4:**
- Frameworks
- LangChain
  - Key API concepts: `Models`,  `LLM`,  `Prompts`,  `Agents`
  - Simple examples
- Vector databases
