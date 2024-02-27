#!/bin/bash

# Default values
SCORE_TYPE=original
SCALING_SCORES=True
ELEMENTWISE_AFFINE=False
EXTRA_BINDINGS='P/EVALUATE/Evaluator.datasets=["D/P3WIKIQA/EVAL", "D/P3PAWS/EVAL", "D/P3SOCIALIQA/EVAL", "D/P3QASC/EVAL", "D/P3ROPES/EVAL", "D/P3QUARTZ/EVAL", "D/P3COSMOSQA/EVAL", "D/P3QUAIL/EVAL", "D/P3AGNEWS/EVAL", "D/P3AMAZONPOLARITY/EVAL", "D/P3SAMSUM/EVAL", "D/P3WIKIBIO/EVAL", "D/P3DREAM/EVAL", "D/P3WIQA/EVAL", "D/P3QUAREL/EVAL", "D/P3SCIQ/EVAL", "D/P3QUOREF/EVAL", "D/P3DUORC/EVAL", "D/P3ROTTENTOMATOES/EVAL", "D/P3YELP/EVAL", "D/P3COMMONGEN/EVAL", "D/P3GIGAWORD/EVAL", "D/P3XSUM/EVAL", "D/P3MRPC/EVAL", "D/P3QQP/EVAL", "D/P3COMMONSENSEQA/EVAL", "D/P3COSE/EVAL", "D/P3WIKIHOP/EVAL", "D/P3HOTPOTQA/EVAL", "D/P3APPREVIEWS/EVAL", "D/P3TREC/EVAL", "D/P3MULTINEWS/EVAL", "D/P3IMDB/EVAL", "D/P3ADVERSARIALQA/EVAL", "D/P3CNNDAILYMAIL/EVAL", "D/P3DBPEDIA14/EVAL"] M/MODEL/FFNExperts.topk_value=2 M/MODEL/FFNExperts.normalize_topk=True'
ARCH=A2
MODEL_TYPE=t5xl
# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -score_type)
      SCORE_TYPE="$2"
      shift
      ;;
    -scaling_scores)
      SCALING_SCORES="$2"
      shift
      ;;
    -elementwise_affine)
      ELEMENTWISE_AFFINE="$2"
      shift
      ;;
    -exp_name)
      EXP_NAME="$2"
      shift
      ;;
    -arch)
      ARCH="$2"
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

if [[ "$ARCH" == "A1" ]]; then
  echo -e "Using A1 architecture\n"
  ARCH_GIN=colm/models/${MODEL_TYPE}/moe_lora_rank16_a1.gin
elif [[ "$ARCH" == "A2" ]]; then
  echo -e "Using A2 architecture\n"
  ARCH_GIN=colm/models/${MODEL_TYPE}/moe_lora_rank16_a2.gin
else
  echo -e "Error: ARCH is not set to A1 or A2\n"
  exit 1
fi

echo -e "\nRunning Eval\n"
echo -e "score_type: ${SCORE_TYPE}, scaling_scores: ${SCALING_SCORES} elementwise_affine: ${ELEMENTWISE_AFFINE}\n"

# Use the variables directly in the command
EXP_NAME=${EXP_NAME} python src/launch_single_process.py --gin_files colm/datasets/p3_${MODEL_TYPE}.gin colm/datasets/flanv2_${MODEL_TYPE}.gin colm/datasets/bigbench.gin colm/models/${MODEL_TYPE}/t5.gin ${ARCH_GIN} colm/experiments/eval.gin --gin_bindings M/MODEL/Router.score_type=\"${SCORE_TYPE}\" M/MODEL/Router.scaling_scores=${SCALING_SCORES} M/MODEL/Router.elementwise_affine=${ELEMENTWISE_AFFINE} ${EXTRA_BINDINGS}
