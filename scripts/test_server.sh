#!/bin/bash -eu
set -o pipefail

# Examples of connecting to the server
# Query API and create an LLM request

DIR="$( cd "$( dirname "$0" )" && pwd )"
source "$DIR/config.sh"

echo
echo
echo "Info:"
curl "http://$SERVER:$SERVER_PORT/v1/info" | jq .

echo
echo
echo "Models:"
curl "http://$SERVER:$SERVER_PORT/v1/models" | jq .


# NOTE: Change the lines (e.g. 'fake' models is used for debugging)
#     "model": "fake",
#     "model": "llama_7b",

echo
echo
echo "Completion:"
curl -X 'POST' \
  "http://$SERVER:$SERVER_PORT/v1/completions/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "fake",
  "prompt": "Hello, my name is",
  "suffix": "string",
  "max_tokens": 50,
  "temperature": 0.1,
  "top_p": 0,
  "n": 1,
  "logprobs": 0,
  "echo": true,
  "stop": [
    "string"
  ],
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "best_of": 0,
  "logit_bias": {
    "additionalProp1": 0,
    "additionalProp2": 0,
    "additionalProp3": 0
  }
}' | jq .