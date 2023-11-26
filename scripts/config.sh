
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
export VECTOR_DB_DIR="$LM_HOME/weaviate"

# Servers
export SERVER="127.0.0.1"
export SERVER_PORT=8000

# Activate environment
source "$LM_HOME/.venv/bin/activate"
