#!/bin/bash

# Default values
SCORE_TYPE=original
SCALING_SCORES=True
ELEMENTWISE_AFFINE=False
ARCH=A2
COMPONENT=None
COMPUTE_HIDDENS=False
HIDDENS_SUFFIX="_a2"
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
    -component)
      COMPONENT="$2"
      shift
      ;;
    -arch)
      ARCH="$2"
      shift
      ;;
    -exp_name)
      EXP_NAME="$2"
      shift
      ;;
    -dataset)
      DATASET="$2"
      shift
      ;;
    -compute_hiddens)
      COMPUTE_HIDDENS="$2"
      shift
      ;;
    -hiddens_suffix)
      HIDDENS_SUFFIX="$2"
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

if [[ "$ARCH" == "A1" ]]; then
  echo -e "Using A1 architecture\n"
  ARCH_GIN=colm/models/${MODEL_TYPE}/moe_lora_rank16_a1.gin
  REGEX_GIN='create_router_embeddings.regex_pattern_list=["encoder_linear","decoder"]'
elif [[ "$ARCH" == "A2" ]]; then
  echo -e "Using A2 architecture\n"
  ARCH_GIN=colm/models/${MODEL_TYPE}/moe_lora_rank16_a2.gin
  REGEX_GIN='create_router_embeddings.regex_pattern_list=["encoder_linear","decoder_linear"]'
  WEIGHTED_HIDDENS='M/MODEL/FFNExperts.replace_with_weighted_hiddens=True'
else
  echo -e "Error: ARCH is not set to A1 or A2\n"
  exit 1
fi


echo -e "score_type: ${SCORE_TYPE}, scaling_scores: ${SCALING_SCORES} elementwise_affine: ${ELEMENTWISE_AFFINE}\n"

# echo -e "\nInsert required keys into the model"
# python scripts/manipulations.py --gin_bindings create_router_embeddings.path=\"${EXP_NAME}\" 'func_caller.func=@create_router_embeddings' ${REGEX_GIN}

if [[ "${COMPUTE_HIDDENS}" == "True" ]]; then
  echo -e "\ncomputing averaged hiddens"

  EXP_NAME=${EXP_NAME} python src/launch_single_process.py --gin_files colm/datasets/p3_${MODEL_TYPE}.gin colm/datasets/flanv2_${MODEL_TYPE}.gin colm/models/${MODEL_TYPE}/t5.gin ${ARCH_GIN} colm/experiments/eval.gin --gin_bindings P/EVALUATE/Evaluator.datasets=\"D/${DATASET}/TRAIN\" D/${DATASET}/TRAIN/P3Dataset.metrics=\[\"custom\"\] D/${DATASET}/TRAIN/FlanDataset.metrics=\[\"custom\"\] D/${DATASET}/TRAIN/InterfaceInfo.interface=\"gen\"  'P/EVALUATE/Evaluator.analysis_processors=[@SaveAveragedHiddens()]' SaveAveragedHiddens.save_dir=\"exp_out/${EXP_NAME}/averaged_hiddens${HIDDENS_SUFFIX}\" D/${DATASET}/TRAIN/P3Dataset.max_examples_per_dataset=1000 D/${DATASET}/TRAIN/FlanDataset.max_examples_per_dataset=1000 M/MODEL/Router.score_type=\"${SCORE_TYPE}\" M/MODEL/Router.scaling_scores=${SCALING_SCORES} 'main.procedure_exec_order=["P/EVALUATE"]' M/MODEL/Router.elementwise_affine=${ELEMENTWISE_AFFINE} ${WEIGHTED_HIDDENS} ${EXTRA_BINDINGS}
fi

if [[ "${COMPONENT}" == "pretrained" ]]; then
  echo -e "\nUsing pretrained component from model"
  python scripts/insert_averaged_hiddens.py --exp_name ${EXP_NAME} --dataset D/${DATASET}/TRAIN --index 0 --component pretrained --hiddens_suffix ${HIDDENS_SUFFIX}
elif [[ "${COMPONENT}" == "task" ]]; then
  echo -e "\nUsing task component from model"
  python scripts/insert_averaged_hiddens.py --exp_name ${EXP_NAME} --dataset D/${DATASET}/TRAIN --index 0 --component task --hiddens_suffix ${HIDDENS_SUFFIX}
elif [[ "${COMPONENT}" == "specific_task" ]]; then
  if [[ "${COMPUTE_HIDDENS}" == "True" ]]; then
    echo -e "\n computing averaged init hiddens"
    EXP_NAME=P3C4_lora python src/launch_single_process.py --gin_files colm/datasets/p3.gin colm/models/t5_large/t5.gin ${ARCH_GIN} colm/experiments/eval.gin --gin_bindings P/EVALUATE/Evaluator.datasets=\"D/${DATASET}/TRAIN\" D/${DATASET}/TRAIN/P3Dataset.metrics=\[\"custom\"\] D/${DATASET}/TRAIN/InterfaceInfo.interface=\"gen\"  'P/EVALUATE/Evaluator.analysis_processors=[@SaveAveragedHiddens()]' SaveAveragedHiddens.save_dir=\"exp_out/${EXP_NAME}/averaged_init_hiddens${HIDDENS_SUFFIX}\" D/${DATASET}/TRAIN/P3Dataset.max_pretemplate_examples_per_dataset=1000 M/MODEL/Router.score_type=\"${SCORE_TYPE}\" M/MODEL/Router.scaling_scores=${SCALING_SCORES} M/MODEL/Router.elementwise_affine=${ELEMENTWISE_AFFINE} ${EXTRA_BINDINGS}
  fi
  echo -e "\nUsing specific task component from model"
  python scripts/insert_averaged_hiddens.py --exp_name ${EXP_NAME} --dataset D/${DATASET}/TRAIN --index 0 --component specific_task --hiddens_suffix ${HIDDENS_SUFFIX}
else
  python scripts/insert_averaged_hiddens.py --exp_name ${EXP_NAME} --dataset D/${DATASET}/TRAIN --index 0 --hiddens_suffix ${HIDDENS_SUFFIX}
fi
