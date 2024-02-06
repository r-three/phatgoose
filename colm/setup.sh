export MM_ROOT=`pwd`
export PYTHONPATH=$MM_ROOT:$PYTHONPATH
export PYTHON_EXEC=python
mkdir -p ~/.cache/phatgoose/
export HUGGINGFACE_HUB_CACHE=~/.cache/phatgoose/
export TRANSFORMERS_CACHE=~/.cache/phatgoose/
export HF_HOME=~/.cache/phatgoose/
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=phatgoose
