import logging
import os
from pathlib import Path

from fastapi import FastAPI
from .models import load_models
from .routers import info, completions
from .config import API_VERSION_PREFIX, CHECKPOINTS_DIR, TEST_LLM, LM_HOME, RANDOM_SEED

# Logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info(f"LM_HOME         : {LM_HOME}")
logger.info(f"CHECKPOINTS_DIR : {CHECKPOINTS_DIR}")
logger.info(f"TEST_LLM        : {TEST_LLM}")
logger.info(f"RANDOM_SEED     : {RANDOM_SEED}")

# app = FastAPI(dependencies=[Depends(get_query_token)])
app = FastAPI()
app.include_router(info.router, prefix=API_VERSION_PREFIX)
app.include_router(completions.router, prefix=API_VERSION_PREFIX)

# Note: I've tried `lifespan` but it did not work.
# Maybe I need to update to FastAPI version
# Reference: https://fastapi.tiangolo.com/advanced/events/
@app.on_event("startup")
async def startup_event():
    """ Initialize on startup """
    load_models(test_models=TEST_LLM)


@app.get("/")
async def root():
    return {"message": "LLaMA server"}
