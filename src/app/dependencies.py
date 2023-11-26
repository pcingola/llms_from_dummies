from typing import Annotated

from fastapi import Header, HTTPException

VERSION = "0.1.0"


async def get_token_header(x_token: Annotated[str, Header()]):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


async def get_query_token(token: str):
    if token != "my_token":
        raise HTTPException(status_code=400, detail="No my_token token provided")
    