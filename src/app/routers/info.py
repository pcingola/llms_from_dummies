import logging

from fastapi import APIRouter
from ..dependencies import VERSION
from ..models import get_models_info, get_model_info

# Logging setup
logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["info"],
    # dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not found"}},
)

@router.get("/info")
async def info():
    """ Get the info about the server """
    return {"name": "Lit-LLaMA FastApi server"
            , "version": VERSION
            , "docs": "/docs"
            }


@router.get("/model{model}")
async def model(model: str):
    """ Models list """
    return get_model_info(model)


@router.get("/models")
async def models():
    """ Models list """
    return {"object": "list"
            , "data": get_models_info()
            }


