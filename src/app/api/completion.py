from pydantic import BaseModel

class CompletionRequest(BaseModel):
    """ 
    Completion model
    Based on OpenAI's Completion
    Reference: https://platform.openai.com/docs/api-reference/completions/create
    """
    # TODO: Add `Field()` to show in the API documentation, see details https://fastapi.tiangolo.com/tutorial/body-fields/
    model: str
    prompt: str
    suffix: str
    max_tokens: int
    temperature: float
    top_p: float
    n: int
    logprobs: int
    echo: bool
    stop: list[str]
    presence_penalty: float
    frequency_penalty: float
    best_of: int
    logit_bias: dict[str, float]


class CompletionChoices(BaseModel):
    """ 
    Completion model
    Based on OpenAI's Completion
    Reference: https://platform.openai.com/docs/api-reference/completions/create
    """
    text: str
    index: int
    logprobs: int
    finish_reason: str

class CompletionUsage(BaseModel):
    """ 
    Completion model
    Based on OpenAI's Completion
    Reference: https://platform.openai.com/docs/api-reference/completions/create
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


