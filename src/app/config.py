import os
from pathlib import Path

# API version number
API_VERSION = "1"
API_VERSION_PREFIX = f"/v{API_VERSION}"

# Used for debugging: If set, random seed is applied on every request
RANDOM_SEED = None  # 1234

def get_env_var(name, default):
    """ Get an environment variable """
    if name in os.environ:
        return os.environ[name]
    return default


def to_bool(value):
    """ Convert a string to a boolean """
    return str(value).lower() in ['true', 'yes', 't', '1']


# Paramaters
# TEST_LLM : Use only 'test' LLMs?, i.e. don't load large models
#            This is used to speedup startup when developing the API
TEST_LLM = to_bool(get_env_var("TEST_LLM", "False"))

# LM_HOME: Project's home directory
LM_HOME_DEFAULT = os.environ.get("HOME") + "/llms_from_dummies"
LM_HOME = Path(get_env_var("LM_HOME", LM_HOME_DEFAULT))

# DATA_DIR: Data directory
CHECKPOINTS_DIR = LM_HOME / 'checkpoints'
