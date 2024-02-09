# PHATGOOSE Repository

## Introduction
PHATGOOSE, which stands for Post-Hoc Adaptive Gating Over an Ocean of Specialized Experts, enables zero-shot generalization from specialized experts (eg PEFT modules) trained on diverse datasets by adaptively routing among them. It requires an additional, inexpensive training step of a gate in front of a frozen PEFT module for its corresponding task.

## Setup
Follow these steps to set up the PHATGOOSE environment:

1. **Create a Conda Environment**:
   ```shell
   conda create -n phatgoose python==3.9
   conda activate phatgoose
   ```

2. **Install Required Packages**:
   ```shell
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   source colm/setup.sh
   ```

## Training Procedure
Below are the steps for required for PHATGOOSE and other baselines:

### Train a Single LoRA on a Dataset
Use the example command below to train:
```shell
bash colm/experiments/bash_scripts/train_single_task_loralinear.sh -exp_name Flanv2zsWmt16translateroen_t5xl_lora -dataset FLAN2021WMT16TRANSLATEROEN/ZS -extra_bindings 'P/TRAIN/Trainer.gradient_accumulation_factor=256';
```
*Note: Ensure the `gradient_accumulation_factor` is set according to the batch_size in `colm/datasets/<file>.gin` files.*

### Convert into MoE Style with a Single Expert
```shell
python scripts/manipulations.py --gin_bindings 'put_index_to_lora.path="Flanv2zsWmt16translateroen_t5xl_lora"' 'put_index_to_lora.out_path="datasets_concatenated/Flanv2zsWmt16translateroen_t5xl_lora"' 'func_caller.func=@put_index_to_lora'
```
*The modified checkpoints are saved to the `datasets_concatenated` sub-directory in the `exp_out` directory to double-check if manipulation worked as intended and to retain the old checkpoint of lora_linear if needed.*

### Train the Corresponding Gate
```shell
bash colm/experiments/bash_scripts/train_gate.sh -exp_name datasets_concatenated/Flanv2zsWmt16translateroen_t5xl_lora_inpgatetrainnogumbel -dataset FLAN2021WMT16TRANSLATEROEN/ZS -old_exp_name datasets_concatenated/Flanv2zsWmt16translateroen_t5xl_lora -extra_bindings 'main.logging_backend=None P/TRAIN/Trainer.gradient_accumulation_factor=512';
```
*Note: We don't perform any logging while gate training, but it can be added by setting `main.logging_backend="wandb"` if needed.*

### Avoid Saving to GCP Unintentionally
Training a model always saves to GCP. If this is not intended, you can add `MOMA/save_weights.should_save_to_gcp=False` in the extra_bindings of training commands. For example:
```shell
bash colm/experiments/bash_scripts/train_single_task_loralinear.sh -exp_name Flanv2zsWmt16translateroen_t5xl_lora -dataset FLAN2021WMT16TRANSLATEROEN/ZS -extra_bindings 'MOMA/save_weights.should_save_to_gcp=False P/TRAIN/Trainer.gradient_accumulation_factor=256';
```

### Make Trained Gate as the Routing Vector
Modify the checkpoint by running:
```shell
python scripts/manipulations.py --gin_bindings 'use_input_gate_as_router.path="datasets_concatenated/Flanv2zsWmt16translateroen_t5xl_lora_inpgatetrainnogumbel"' 'func_caller.func=@use_input_gate_as_router';
```

### Concatenate All Experts with Gates to form an MoE
```shell
python scripts/concatenate.py --gin_bindings 'run_concatenate.print_commands=False' 'run_concatenate.out_path="FullCompleteA2inpgatetrainnogumbel_t5xl_lora_concatenated"' 'func_caller.func=@run_concatenate' 'run_concatenate.suffix="t5xl_lora_inpgatetrainnogumbel"' 'run_concatenate.datasets="Full"'
```

## Baseline Methods 

### Compute Average Hiddens for Average Activation Baseline
```shell
python scripts/concatenate.py --gin_bindings 'run_concatenate.print_commands=True' 'run_concatenate.out_path="FullCompleteA2_t5xl_lora_concatenated"' 'func_caller.func=@run_concatenate' 'run_concatenate.suffix="t5xl_lora"' 'run_concatenate.compute_hiddens=True' 'run_concatenate.extra_bindings="M/MODEL/ENCODER/ExposeHidden.reduction_method=\"masked_mean\" M/MODEL/DECODER/ExposeHidden.reduction_method=\"mean\""' 'run_concatenate.datasets="Full"'

... *continue with steps from above command* ...

python scripts/concatenate.py --gin_bindings 'run_concatenate.print_commands=False' 'run_concatenate.out_path="FullCompleteA2_t5xl_lora_concatenated"' 'func_caller.func=@run_concatenate' 'run_concatenate.suffix="t5xl_lora"' 'run_concatenate.compute_hiddens=False' 'run_concatenate.datasets="Full"'
```

## Create Expert Library  and checkpoint for Retrieval 
```shell
bash colm/experiments/bash_scripts/retriever.sh -make_expert_library True -dataset_setting Full
bash colm/experiments/bash_scripts/retriever.sh -create_checkpoint True -dataset_setting All
```

## Create Merged Experts checkpoint 
```shell
python scripts/manipulations.py --gin_bindings 'average_outer_product_lora_weights.path="FullCompleteA2_t5xl_lora_concatenated"' 'average_outer_product_lora_weights.out_path="FullParameteravg_t5xl_lora_outerproduct"' 'func_caller.func=@average_outer_product_lora_weights'
```


## Models and Datasets
We provide checkpoints for PHATGOOSE, along with baselines such as Average Activation, Merged Experts, and Retrieval, accessible at our [Hugging Face repository](https://huggingface.co/r-three).

For individual experts, we recommend splitting any checkpoint other than Merged Experts. Each checkpoint contains keys for an expert ending with `layer1__i`, `layer2__i`, indicating the LoRA parameters of the expert `i` trained on dataset `i`. The sequence of datasets is detailed in the [`scripts/concatenate.py` file](scripts/concatenate.py). 

Datasets including T0 Held-in and BIG-bench are available through Hugging Face. For the FLAN dataset, we will provide a processed version soon, sourced from the [FLAN dataset on Hugging Face](https://huggingface.co/datasets/Open-Orca/FLAN).

## Evaluation
Here are the scripts for evaluating different methods:

### Multitask
```shell
bash colm/experiments/bash_scripts/eval_multitask.sh -exp_name flan_t5_xl -extra_bindings 'P/EVALUATE/Evaluator.datasets=["D/BBBOOLEANEXPRESSIONS/EVAL", "D/BBCAUSALJUDGEMENT/EVAL", "D/BBDATEUNDERSTANDING/EVAL", "D/BBDISAMBIGUATIONQA/EVAL", "D/BBFORMALFALLACIES/EVAL", "D/BBGEOMETRICSHAPES/EVAL", "D/BBHYPERBATON/EVAL", "D/BBLOGICALDEDUCTION/EVAL", "D/BBMOVIERECOMMENDATION/EVAL", "D/BBMULTISTEPARITHMETICTWO/EVAL", "D/BBNAVIGATE/EVAL", "D/BBOBJECTCOUNTING/EVAL", "D/BBPENGUINSINATABLE/EVAL", "D/BBREASONINGABOUTCOLOREDOBJECTS/EVAL", "D/BBRUINNAMES/EVAL", "D/BBSALIENTTRANSLATIONERRORDETECTION/EVAL", "D/BBSNARKS/EVAL", "D/BBSPORTSUNDERSTANDING/EVAL", "D/BBTEMPORALSEQUENCES/EVAL", "D/BBTRACKINGSHUFFLEDOBJECTS/EVAL", "D/BBWEBOFLIES/EVAL", "D/BBWORDSORTING/EVAL"] P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText()] WriteOutputText.save_dir="exp_out/flan_t5_xl/output_text" M/MODEL/hf_torch_model.model_name_or_path="google/flan-t5-xl" M/MODEL/Model.init_moma_calls=[]'
```

### Single Expert
```shell
bash colm/experiments/bash_scripts/eval_multitask.sh -exp_name datasets_concatenated/P3Socialiqa_t5xl_lora -extra_bindings 'P/EVALUATE/Evaluator.datasets=["D/BBBOOLEANEXPRESSIONS/EVAL", "D/BBCAUSALJUDGEMENT/EVAL", "D/BBDATEUNDERSTANDING/EVAL", "D/BBDISAMBIGUATIONQA/EVAL", "D/BBFORMALFALLACIES/EVAL", "D/BBGEOMETRICSHAPES/EVAL", "D/BBHYPERBATON/EVAL", "D/BBLOGICALDEDUCTION/EVAL", "D/BBMOVIERECOMMENDATION/EVAL", "D/BBMULTISTEPARITHMETICTWO/EVAL", "D/BBNAVIGATE/EVAL", "D/BBOBJECTCOUNTING/EVAL", "D/BBPENGUINSINATABLE/EVAL", "D/BBREASONINGABOUTCOLOREDOBJECTS/EVAL", "D/BBRUINNAMES/EVAL", "D/BBSALIENTTRANSLATIONERRORDETECTION/EVAL", "D/BBSNARKS/EVAL", "D/BBSPORTSUNDERSTANDING/EVAL", "D/BBTEMPORALSEQUENCES/EVAL", "D/BBTRACKINGSHUFFLEDOBJECTS/EVAL", "D/BBWEBOFLIES/EVAL", "D/BBWORDSORTING/EVAL"] M/MODEL/ENCODER/ExposeHidden.reduction_method="masked_mean" M/MODEL/DECODER/ExposeHidden.reduction_method="mean" P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText()] WriteOutputText.save_dir="exp_out/datasets_concatenated/P3Socialiqa_t5xl_lora/output_text"'
```

###  Retrieval
```shell
bash colm/experiments/bash_scripts/retriever.sh -dataset_setting Full -extra_bindings 'main.procedure_exec_order=["P/EVALUATE/BBH"] P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText()] WriteOutputText.save_dir="exp_out/FullCompleteansretrieval_t5xl_lora_concatenated/output_text"'
```

### Merged Experts
```shell
bash colm/experiments/bash_scripts/eval_multitask.sh -exp_name FullParameteravg_t5xl_lora_concatenated -extra_bindings 'P/EVALUATE/Evaluator.datasets=["D/BBBOOLEANEXPRESSIONS/EVAL", "D/BBCAUSALJUDGEMENT/EVAL", "D/BBDATEUNDERSTANDING/EVAL", "D/BBDISAMBIGUATIONQA/EVAL", "D/BBFORMALFALLACIES/EVAL", "D/BBGEOMETRICSHAPES/EVAL", "D/BBHYPERBATON/EVAL", "D/BBLOGICALDEDUCTION/EVAL", "D/BBMOVIERECOMMENDATION/EVAL", "D/BBMULTISTEPARITHMETICTWO/EVAL", "D/BBNAVIGATE/EVAL", "D/BBOBJECTCOUNTING/EVAL", "D/BBPENGUINSINATABLE/EVAL", "D/BBREASONINGABOUTCOLOREDOBJECTS/EVAL", "D/BBRUINNAMES/EVAL", "D/BBSALIENTTRANSLATIONERRORDETECTION/EVAL", "D/BBSNARKS/EVAL", "D/BBSPORTSUNDERSTANDING/EVAL", "D/BBTEMPORALSEQUENCES/EVAL", "D/BBTRACKINGSHUFFLEDOBJECTS/EVAL", "D/BBWEBOFLIES/EVAL", "D/BBWORDSORTING/EVAL"] M/MODEL/ENCODER/ExposeHidden.reduction_method="masked_mean" M/MODEL/DECODER/ExposeHidden.reduction_method="mean" P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText()] WriteOutputText.save_dir="exp_out/FullParameteravg_t5xl_lora_concatenated/output_text"' 
```

### Average Activation
```shell
bash colm/experiments/bash_scripts/eval_multitask.sh -exp_name FullCompleteA2_t5xl_lora_concatenated -extra_bindings 'P/EVALUATE/Evaluator.datasets=["D/BBBOOLEANEXPRESSIONS/EVAL", "D/BBCAUSALJUDGEMENT/EVAL", "D/BBDATEUNDERSTANDING/EVAL", "D/BBDISAMBIGUATIONQA/EVAL", "D/BBFORMALFALLACIES/EVAL", "D/BBGEOMETRICSHAPES/EVAL", "D/BBHYPERBATON/EVAL", "D/BBLOGICALDEDUCTION/EVAL", "D/BBMOVIERECOMMENDATION/EVAL", "D/BBMULTISTEPARITHMETICTWO/EVAL", "D/BBNAVIGATE/EVAL", "D/BBOBJECTCOUNTING/EVAL", "D/BBPENGUINSINATABLE/EVAL", "D/BBREASONINGABOUTCOLOREDOBJECTS/EVAL", "D/BBRUINNAMES/EVAL", "D/BBSALIENTTRANSLATIONERRORDETECTION/EVAL", "D/BBSNARKS/EVAL", "D/BBSPORTSUNDERSTANDING/EVAL", "D/BBTEMPORALSEQUENCES/EVAL", "D/BBTRACKINGSHUFFLEDOBJECTS/EVAL", "D/BBWEBOFLIES/EVAL", "D/BBWORDSORTING/EVAL"] M/MODEL/FFNExperts.topk_value=2 M/MODEL/FFNExperts.normalize_topk=True M/MODEL/ENCODER/ExposeHidden.reduction_method=None M/MODEL/DECODER/ExposeHidden.reduction_method=None P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText(), @RoutingDistribution()] WriteOutputText.save_dir="exp_out/FullCompleteA2_t5xl_lora_concatenated/output_text" RoutingDistribution.save_dir="exp_out/FullCompleteA2_t5xl_lora_concatenated/routing_distribution"'
```

### PHATGOOSE
```shell
bash colm/experiments/bash_scripts/eval_multitask.sh -exp_name FullCompleteA2inpgatetrainnogumbel_t5xl_lora_concatenated -extra_bindings 'P/EVALUATE/Evaluator.datasets=["D/BBBOOLEANEXPRESSIONS/EVAL", "D/BBCAUSALJUDGEMENT/EVAL", "D/BBDATEUNDERSTANDING/EVAL", "D/BBDISAMBIGUATIONQA/EVAL", "D/BBFORMALFALLACIES/EVAL", "D/BBGEOMETRICSHAPES/EVAL", "D/BBHYPERBATON/EVAL", "D/BBLOGICALDEDUCTION/EVAL", "D/BBMOVIERECOMMENDATION/EVAL", "D/BBMULTISTEPARITHMETICTWO/EVAL", "D/BBNAVIGATE/EVAL", "D/BBOBJECTCOUNTING/EVAL", "D/BBPENGUINSINATABLE/EVAL", "D/BBREASONINGABOUTCOLOREDOBJECTS/EVAL", "D/BBRUINNAMES/EVAL", "D/BBSALIENTTRANSLATIONERRORDETECTION/EVAL", "D/BBSNARKS/EVAL", "D/BBSPORTSUNDERSTANDING/EVAL", "D/BBTEMPORALSEQUENCES/EVAL", "D/BBTRACKINGSHUFFLEDOBJECTS/EVAL", "D/BBWEBOFLIES/EVAL", "D/BBWORDSORTING/EVAL"] M/MODEL/FFNExperts.topk_value=2 M/MODEL/FFNExperts.normalize_topk=True M/MODEL/ENCODER/ExposeHidden.reduction_method=None M/MODEL/DECODER/ExposeHidden.reduction_method=None P/EVALUATE/Evaluator.analysis_processors=[@WriteOutputText(), @RoutingDistribution()] WriteOutputText.save_dir="exp_out/FullCompleteA2inpgatetrainnogumbel_t5xl_lora_concatenated/output_text" RoutingDistribution.save_dir="exp_out/FullCompleteA2inpgatetrainnogumbel_t5xl_lora_concatenated/routing_distribution"'
```

*Change the datasets and the checkpoint accordingly to run for BIG-bench Lite and T0 Held-out datasets.*