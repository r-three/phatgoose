#!/bin/bash

# Default values
MODEL_TYPE=t5xl
# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -exp_name)
      EXP_NAME="$2"
      shift
      ;;
    -dataset)
      DATASET="$2"
      shift
      ;;
    -model_type)
      MODEL_TYPE="$2"
      shift
      ;;
    -extra_bindings)
      EXTRA_BINDINGS="$2"
      shift
      ;;
    *)
      # Unknown option, ignore
      ;;
  esac

  shift
done


if [ -z "$EXP_NAME" ]; then
  echo "Error: exp_name is not set."
  exit 1
fi

if [ -z "$DATASET" ]; then
  echo "Error: exp_name is not set."
  exit 1
fi

echo -e "\nEval ${DATASET}\n"

# Use the variables directly in the command
EXP_NAME=${EXP_NAME} python src/launch_single_process.py --gin_files colm/datasets/p3_${MODEL_TYPE}.gin colm/datasets/flanv2_${MODEL_TYPE}.gin colm/models/${MODEL_TYPE}/t5.gin colm/models/moe_lora_rank16.gin colm/experiments/eval.gin --gin_bindings P/EVALUATE/Evaluator.datasets=\"D/${DATASET}/EVAL\" P/EVALUATE/Evaluator.analysis_processors=[] 'M/MODEL/Model.init_moma_calls = [@M/MODEL/modify_with_lora, @M/MODEL/load_weights]' ${EXTRA_BINDINGS}
