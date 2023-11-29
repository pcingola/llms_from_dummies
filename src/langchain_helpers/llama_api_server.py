import os
from typing import Any, List, Mapping, Optional

import requests
from langchain.llms import OpenAI
from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.prompts import PromptTemplate

from langchain_helpers.debug_callback_handler import DebugCallbackHandler

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


from app.api import CompletionRequest


class OpenLlamaLLM(LLM):
    host = "localhost"
    port = 8000
    url = "http://localhost:8000/v1/completions/"
    """Server URL"""
    model_name = "llama_7b"
    """ Model's name """
    temperature: float = 0.7
    """What sampling temperature to use."""
    max_tokens: int = 256
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""
    top_p: float = 1
    """Penalizes repeated tokens."""
    n: int = 1
    """Adjust the probability of specific tokens being generated."""
    request_timeout = 120
    """The maximum number of seconds to wait for the server to respond."""
    max_retries: int = 3
    """The maximum number of retries before giving up."""

    def __init__(self, host:str = "127.0.0.1", port:int = 8000, model_name:str = "llama_7b", **kwargs) -> None:
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/v1/completions/"
        self.model_name = model_name

    @property
    def _llm_type(self) -> str:
        return "open_llama_fastapi_server"

    def _completion_request(self, prompt: str) -> CompletionRequest:
        """ Create a CompletionRequest object from the current parameters """
        return { 'model': self.model_name
                , 'prompt': prompt
                , 'suffix': ''
                , 'max_tokens': self.max_tokens
                , 'temperature': self.temperature
                , 'top_p': self.top_p
                , 'n': self.n
                , 'logprobs': 1
                , 'echo': False
                , 'stop': []
                , 'presence_penalty': 0
                , 'frequency_penalty': 0
                , 'best_of': 1
                , 'logit_bias': {}
                }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        print(f"OpenLlamaLLM prompt (len:{len(prompt)}): '{prompt}'")
        if stop is not None:
            print(f"OpenLlamaLLM, stop kwargs: {stop}")
        compl_req = self._completion_request(prompt)
        if run_manager:
            run_manager.on_llm_new_token(token=prompt)
        r = requests.post(url=self.url, json=compl_req, timeout=self.request_timeout)
        logger.debug(f"OpenLlamaLLM response: {r}")
        r = r.json()
        # Get first response
        response = r['choices'][0]['text']
        return response
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"url": self.url
                , 'model': self.model_name
                , 'max_tokens': self.max_tokens
                , 'temperature': self.temperature
                , 'top_p': self.top_p
                , 'n': self.n
                , 'max_retries': self.max_retries
                , 'request_timeout': self.request_timeout
                }
