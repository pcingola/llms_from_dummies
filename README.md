# Large Language Models (LLMs) From Dummies: Tutorial

This is a series of introductory tutorials about language models (such as GPT).

### Description

A technical introduction to the topic of language models (LLMs), explaining the basic concepts, and creating sample code.

In these tutorials we fully implement a language model from scratch; including all the tools to train and use the model.

### Audience

The audience is assumed to be technical, but non-experts on neural networks.

We assume coding skills (Python), basic understanding of what a neural network is, what “back-propagation” is.
If you don’t remember the concepts, a quick brush up before the tutorials should be enough.

You are NOT assumed to be an expert practitioner on neural nets.

## Tutorial

- LLMs from dummies - part 1
  - [Video](https://youtu.be/tMr08LaMDtA)
  - [Slides](./slides/LLMs_from_dummies_part_1.pdf)
  - [Notebook](./notebooks/LLMs_from_dummies_Part_1.ipynb)
- LLMs from dummies - part 2
  - [Video](https://youtu.be/SGLJHH2Wzns)
  - [Slides](./slides/LLMs_from_dummies_part_2.pdf)
  - [Notebook](./notebooks/LLMs_from_dummies_Part_2.ipynb)
- LLMs from dummies - part 3
  - [Video](https://youtu.be/uYmV0UTn9bE)
  - [Slides](./slides/LLMs_from_dummies_part_3.pdf)
- LLMs from dummies - part 4
  - [Video]()
  - [Slides](./slides/LLMs_from_dummies_part_4.pdf)

## Topics

### Part 1: Attention
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

### Part 2: Transformers & GPT
- Brief recap of Part 1
- Language model (moving away from the "translation" example)
- How the model works 
  - Predicting the next token
  - Blocks and data flow
  - Nomenclature: `block_size`, `n_embed`, etc.
- Language Model 1: Putting it all together
  - Model training: Boilerplate code
  - PyTorch implementation
  - Trivial tokenizer
  - Creating a Dataset
  - Training loop
  - Predicting
  - Model performance: What went wrong?
- Attention Revisited
  - Problem: "The model is cheating"
  - Masked attention
  - Softmax masking trick
- Masked Attention: Improved model
- Multi-Headed Attention
- Positional Encoding
- Language Model 2
- Feed-Forward layer: ReLU layer
- Language Model 3
- Transformer Blocks
- Language Model 4
- Training deep networks
  - Vanishing & Exploding Gradients
  - Residual connections
  - Layer Normalization
  - Dropout
- Language Model 5
  - Scaling the model
- Transformer architecture
   - Encoders stack / Decoder stack
   - Contextual embeddings
   - Transformers

### Part 3: Train, Human feedback, and Model scales
- Recap of previous tutorials
- Running "Lit-Lllama" on EC2. See [`install_llm_in_ec2.md`](./install_llm_in_ec2.md)
  - Select an EC2 instance
  - Installing GPU drivers
  - Install "Lit-Llama" model
- Transformer: Encode & Decoder
  - Translation: How it operates
- Contextual Embeddings
  - Encoder-Decoder link
- Vector databases
  - Key Concepts
  - Document summarizations
  - Embedding similarity 
- Pre-training vs Training
  - Reinforcement Learning
  - Proximal Policy Optimization (PPO)
  - Instruct models
- Fine-tunning vs Adapting
  - LORA: Low Rank Adaptation
  - PERF
  - Frozen models + multiple adaptations
- Instruct models
- GPT: Generative Pre-trained Transformer
- Model Scales
  - GPT-1, GPT-2, GPT-3, GPT-4, PaLM
  - Emergent behaviours
  - Model training costs

### Part 4
- Recap of previous tutorials
- Prompt engineering
  - The prompt is the LLM's "API"
  - Prompt length / information
  - Why do oyu need "prompt engineering"
  - Roles: system, user, assistant, function
  - Prompt enginering techniques
    - Zero-shot
    - Few-Shot
    - Chain of thought
    - Tree of thought
- Open API example (using `curl`)
- Tools: Combining LLMs with external tools/functions
  - Example from "Chain of thought"
  - Function call (from OpenAI)
  - Code example
- Frameworks: LangChain
  - Framework's abstractions
  - Simple Code example
  - LLMs, Prompt templates
  - Output parsers
  - Memory
  - Chains
  - Tools
  - Agents / Code example
  - OpenAI functions / Code example
- Vector databases
  - Why do we need them?
  - Basic concepts
  - Embedding vectors for document search
  - Retreival-Augmented Generation
  - Using LLMs with Vector Databases
    - Refinement
    - Map-Reduce
- Quick review of scaling models in GPUs
  - Single GPU
  - Multiple GPUs: DDP, ZeRO, Pipeline-Parallelism, Naive Parallelism
  - GPU usage Bubbles
  - Tensor parallelism


# LLama examples

### Description

There are examples for a simple implementation of an LLM, an API server, and using LangChain with this server:

1. Running an LLM on EC2: Example of installing a LLama model on an EC2 instance and using it via command line. See details in [`install_llm_in_ec2.md`](./install_llm_in_ec2.md)
2. LLM API Server: Example of creating an API server (FastAPI) that loads an LLM (the one in the previous example) and serves queries via a very simple API.
3. LangChain: Example of connecting LangChain to the API server created in the previous step.

### Install

```bash
# Clone repository
cd
git clone https://github.com/pcingola/llms_from_dummies.git

# Create virtual environment
# Note: You might need to install Python's "venv" by runnign something like
#   apt install -y python3.10-venv
cd llms_from_dummies/
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize git-lfs
# Note: You may need to install git-lfs:
#   apt install git-lfs
git-lfs install
# Download weights from HuggingFace
git clone https://huggingface.co/openlm-research/open_llama_7b data/checkpoints/open-llama/7B

# Convert weights
time python3 \
    lit-llama/scripts/convert_hf_checkpoint.py \
    --checkpoint_dir data/checkpoints/open-llama/7B/ \
    --model_size 7B
```


### Configuration

The configuration directories are in `scripts/config.sh`, please change the directories appropriately

- `LM_HOME`: This is supposed to be installed in `$HOME/llms_from_dummies`. You need to change this value if you are installing in another directory.
- `VENV_DIR`: We assume that a Python virtual environment is installed in the `$LM_HOME/.venv` directory
- `TEST_LLM`: If this variable is set to 'True', the API server will only load a "fake" LLM. This is usefull when debugging the API server (it is much faster than loading the LLama model)
- `SERVER`, `SERVER_PORT`: Server's IP address and port

### Install dependencies
```
pip install -r requirements.txt
```

### Runnign the API server

This server loads a 'lit-llama' model and serves queries via a simple API (FastAPI server implementation)

```
./scripts/run_server.sh
```

**WARNING::** If `TEST_LLM` in `scripts/config.sh` is set to 'True', the API server will only load a "fake" LLM (this is used for debugging)

### Test API queries

The script `./scripts/test_server.sh` runs a few simple queries against the API server and displays the JSON results (you need `jq` installed).

