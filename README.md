# Language Models From Dummies

This is a series of introductory talks about language models (such as GPT).

**What:**
These are basic, but technical introduction to the topic of language models, explaining the basics concepts and going over sample code.
 
**Who:**
The audience is assumed to be technical, but non-experts on neural networks.  We assume basic coding skills and basic understanding of what a neural network is / what “back-propagation” is (if you don’t remember, a quick brush up these concepts before the talks should be enough). You are NOT assumed to be an expert practitioner on neural nets.
 
### Presentataions

- LLMs from dummies - part 1
  - [Video]()
  - [Slides](./LLMs_from_dummies_part_1.pdf)
- LLMs from dummies - part 2
  - [Video](./LLMs_from_dummies_part_2.pdf)
  - 

### Topics / Agenda:

- Part 1:
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

- Part 2:
  - Brief recap of Part 1
  - Language model (moving away from the "translation" example)
  - How the model works 
    - Predicting the next token
    - Blocks and data flow
    - Nomenclature: `block_size`, `n_embed`, etc.
  - 
- Stacking Blocks
   - Encoders stack / Decoder stack
   - Contextual embeddings
   - Transformers
- Pre-training, Training, Fine-tunning, Adapting, Instruct
  - LORA / PERF: Basic concepts
  - "Instruct" models
- Prompts all the way down
   - "Chat": Just isolated requests with "memory" of conversations
   - "Context": Just add a sentence to the prompt
   - "Prompt engineering": Similar to adding the right words in a Google search
- Frameworks:
   - LangChain (API, Models, LLM, Prompts, Agents). Simple examples
   - Vector databases
