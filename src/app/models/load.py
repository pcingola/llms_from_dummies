import logging
from pathlib import Path

import lightning as L
from .model import Model, ModelInfo
from .fake import FakeModel
from .llama import LLamaModel


# Store models in this dictionary
_models = {}

logger = logging.getLogger(__name__)


def get_model(model_id) -> Model:
    """ Return a model """
    return _models.get(model_id)


def get_model_info(model_id) -> ModelInfo:
    """ A model information for 'model_id' """
    m = _models.get(model_id)
    if m:
        return m.model_info
    return None


def get_models_info() -> list[ModelInfo]:
    """ A list of model's information """
    return [_models[_id].model_info for _id in _models.keys()]


def load_model_fake():
    """ Load a fake model (used for testing the API) """
    logger.info("Load 'Fake' model")
    model_info_fake = ModelInfo(id="fake", owned_by="Nobody", permissions=[])
    model = FakeModel(model_info=model_info_fake)
    model.load()
    return model


def load_model_llama_7b():
    """ Load LLama '7B' model """
    logger.info("Load 'LLama 7B' model")
    model_info_llama_7b = ModelInfo(id="llama_7b", owned_by="Nobody", permissions=[])
    model = LLamaModel(model_info=model_info_llama_7b)
    model.load()
    return model


def load_models(test_models: bool = False):
    """ Load ALL models on server startup """
    logger.info("Load ML models")
    # Load fake model (this is used for testing the API)
    _models["fake"] = load_model_fake()
    if test_models:
        logger.info(f"Only loading test models")
        return
    # Load 'llama_7b' model
    _models["llama_7b"] = load_model_llama_7b()
