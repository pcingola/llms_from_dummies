
# IMPORTANT: This is supposed to be the directory where this repository is installed
#            Change to the appropriate location
#
# IMPORTANT: We assume that a Python virtual environment is installed in the '.venv' directory
#
export LM_HOME="$HOME/llms_from_dummies"
cd $LM_HOME

export SCRIPTS_DIR="$LM_HOME/scripts"
export SRC_DIR="$LM_HOME/src"
export LOGS_DIR="$LM_HOME/logs"

# Servers
export SERVER="127.0.0.1"
export SERVER_PORT=8000

# API Server
# If this variable is set to 'True', the API server will only load a "fake" LLM (this is used for debugging)
export TEST_LLM="True"

# Activate environment
export VENV_DIR="$LM_HOME/.venv/bin/activate"
source "$VENV_DIR"
