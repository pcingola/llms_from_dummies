import logging
import uuid
import time

from fastapi import APIRouter, HTTPException
from ..api import CompletionRequest, CompletionChoices, CompletionUsage

from ..models import get_model
# from ..dependencies import get_token_header


# Logging setup
logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/completions",
    tags=["completions"],
    # dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not found"}},
)


@router.post("/")
async def generate(req: CompletionRequest):
    """ Generate an answer to the question """
    model = get_model(req.model)

    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found")

    # Create a completion id and time stamp
    _id = "cmpl-" + str(uuid.uuid4())
    _time = time.time()

    # Calculate completion usage
    total_tokens = 0  # FIXME: Calculate real total tokens (e.g. from user's database?)
    prompt_tokens = len(req.prompt)  # FIXME: Calculate real tokens
    usage = CompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=total_tokens)

    choices = []    
    num_completions = 1 if req.n else req.n
    logger.info(f"Generating {num_completions} completions")
    for i in range(num_completions):
        text = model.generate(prompt=req.prompt
                                    , suffix=req.suffix
                                    , max_tokens=req.max_tokens
                                    , temperature=req.temperature
                                    , top_p=req.top_p
                                    , n=req.n
                                    , logprobs=req.logprobs
                                    , echo=req.echo
                                    , stop=req.stop
                                    , presence_penalty=req.presence_penalty
                                    , frequency_penalty=req.frequency_penalty
                                    , best_of=req.best_of
                                    , logit_bias=req.logit_bias
        )
        
        choices.append(CompletionChoices(text=text, index=0, logprobs=-1, finish_reason="max_tokens"))
        usage.completion_tokens += len(text)  # FIXME: Calculate real tokens
        usage.total_tokens += len(text)  # FIXME: Calculate real tokens

    return {"id": _id
            , "object": "text_completion"
            , "created": _time
            , "model": req.model
            , "prompt": req.prompt
            , "choices": choices
            , "usage": usage
            }
