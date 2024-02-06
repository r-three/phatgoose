import os
import re
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

import gin
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

p3_dataset_dict = {
    "P3Agnews": "P3AGNEWS",
    "P3Amazonpolarity": "P3AMAZONPOLARITY",
    "P3Cosmosqa": "P3COSMOSQA",
    "P3Samsum": "P3SAMSUM",
    "P3Quartz": "P3QUARTZ",
    "P3Ropes": "P3ROPES",
    "P3Wikibio": "P3WIKIBIO",
    "P3Paws": "P3PAWS",
    "P3Wikiqa": "P3WIKIQA",
    "P3Socialiqa": "P3SOCIALIQA",
    "P3Qasc": "P3QASC",
    "P3Quail": "P3QUAIL",
    "P3Dream": "P3DREAM",
    "P3Wiqa": "P3WIQA",
    "P3Quarel": "P3QUAREL",
    "P3Sciq": "P3SCIQ",
    "P3Quoref": "P3QUOREF",
    "P3Duorc": "P3DUORC",
    "P3Rottentomatoes": "P3ROTTENTOMATOES",
    "P3Yelp": "P3YELP",
    "P3Commongen": "P3COMMONGEN",
    "P3Gigaword": "P3GIGAWORD",
    "P3Xsum": "P3XSUM",
    "P3Mrpc": "P3MRPC",
    "P3Qqp": "P3QQP",
    "P3Commonsenseqa": "P3COMMONSENSEQA",
    "P3Cose": "P3COSE",
    "P3Wikihop": "P3WIKIHOP",
    "P3Hotpotqa": "P3HOTPOTQA",
    "P3Appreviews": "P3APPREVIEWS",
    "P3Trec": "P3TREC",
    "P3Multinews": "P3MULTINEWS",
    "P3Imdb": "P3IMDB",
    "P3Adversarialqa": "P3ADVERSARIALQA",
    "P3Cnndailymail": "P3CNNDAILYMAIL",
    "P3Dbpedia14": "P3DBPEDIA14",
}

flanv2_dataset_dict = {
    "Flanv2zsAi2arceasy": "FLAN2021AI2ARCEASY/ZS",
    "Flanv2zsAi2arcchallenge": "FLAN2021AI2ARCCHALLENGE/ZS",
    "Flanv2zsAlgebralinear1d": "FLAN2021ALGEBRALINEAR1D/ZS",
    "Flanv2zsBoolq": "FLAN2021BOOLQ/ZS",
    "Flanv2zsCoqa": "FLAN2021COQA/ZS",
    "Flanv2zsDefpronounresolution": "FLAN2021DEFPRONOUNRESOLUTION/ZS",
    "Flanv2zsDrop": "FLAN2021DROP/ZS",
    "Flanv2zsFixpunct": "FLAN2021FIXPUNCT/ZS",
    "Flanv2zsGemDart": "FLAN2021GEMDART/ZS",
    "Flanv2zsGeme2enlg": "FLAN2021GEME2ENLG/ZS",
    "Flanv2zsGemwebnlgen": "FLAN2021GEMWEBNLGEN/ZS",
    "Flanv2zsGemwikilinguaen": "FLAN2021GEMWIKILINGUAEN/ZS",
    "Flanv2zsGluesst2": "FLAN2021GLUESST2/ZS",
    "Flanv2zsGluecola": "FLAN2021GLUECOLA/ZS",
    "Flanv2zsGluemnli": "FLAN2021GLUEMNLI/ZS",
    "Flanv2zsGlueqnli": "FLAN2021GLUEQNLI/ZS",
    "Flanv2zsGluestsb": "FLAN2021GLUESTSB/ZS",
    "Flanv2zsGluewnli": "FLAN2021GLUEWNLI/ZS",
    "Flanv2zsLambada": "FLAN2021LAMBADA/ZS",
    "Flanv2zsNaturalquestionsopen": "FLAN2021NATURALQUESTIONSOPEN/ZS",
    "Flanv2zsNewsroom": "FLAN2021NEWSROOM/ZS",
    "Flanv2zsOpenbookqa": "FLAN2021OPENBOOKQA/ZS",
    "Flanv2zsOpinionabstractidebate": "FLAN2021OPINIONABSTRACTSIDEBATE/ZS",
    "Flanv2zsOpinionabstractrottentomatoes": "FLAN2021OPINIONABSTRACTSROTTENTOMATOES/ZS",
    "Flanv2zsParacrawlenes": "FLAN2021PARACRAWLENES/ZS",
    "Flanv2zsPiqa": "FLAN2021PIQA/ZS",
    "Flanv2zsQuac": "FLAN2021QUAC/ZS",
    "Flanv2zsSentiment140": "FLAN2021SENTIMENT140/ZS",
    "Flanv2zsSnli": "FLAN2021SNLI/ZS",
    "Flanv2zsSquad": "FLAN2021SQUAD/ZS",
    "Flanv2zsSupergluemultirc": "FLAN2021SUPERGLUEMULTIRC/ZS",
    "Flanv2zsSupergluerecord": "FLAN2021SUPERGLUERECORD/ZS",
    "Flanv2zsTriviaqa": "FLAN2021TRIVIAQA/ZS",
    "Flanv2zsTruecase": "FLAN2021TRUECASE/ZS",
    "Flanv2zsUnifiedqascienceinst": "FLAN2021UNIFIEDQASCIENCEINST/ZS",
    "Flanv2zsWordsegment": "FLAN2021WORDSEGMENT/ZS",
}

niv2_dataset_dict = {
    "Niv2zsTranslation": "NIV2TRANSLATION/ZS",
    "Niv2zsProgramexecution": "NIV2PROGRAMEXECUTION/ZS",
    "Niv2zsQuestiongeneration": "NIV2QUESTIONGENERATION/ZS",
    "Niv2zsSentimentanalysis": "NIV2SENTIMENTANALYSIS/ZS",
    "Niv2zsTextcategorization": "NIV2TEXTCATEGORIZATION/ZS",
    "Niv2zsTextmatching": "NIV2TEXTMATCHING/ZS",
    "Niv2zsToxiclanguagedetection": "NIV2TOXICLANGUAGEDETECTION/ZS",
    "Niv2zsCauseeffectclassification": "NIV2CAUSEEFFECTCLASSIFICATION/ZS",
    "Niv2zsInformationextraction": "NIV2INFORMATIONEXTRACTION/ZS",
    "Niv2zsTextualentailment": "NIV2TEXTUALENTAILMENT/ZS",
    "Niv2zsWrongcandidategeneration": "NIV2WRONGCANDIDATEGENERATION/ZS",
    "Niv2zsNamedentityrecognition": "NIV2NAMEDENTITYRECOGNITION/ZS",
    "Niv2zsCommonsenseclassification": "NIV2COMMONSENSECLASSIFICATION/ZS",
    "Niv2zsFillintheblank": "NIV2FILLINTHEBLANK/ZS",
    "Niv2zsTextcompletion": "NIV2TEXTCOMPLETION/ZS",
    "Niv2zsSentencecomposition": "NIV2SENTENCECOMPOSITION/ZS",
    "Niv2zsTitlegeneration": "NIV2TITLEGENERATION/ZS",
    "Niv2zsLanguageidentification": "NIV2LANGUAGEIDENTIFICATION/ZS",
    "Niv2zsQuestionunderstanding": "NIV2QUESTIONUNDERSTANDING/ZS",
    "Niv2zsSentenceperturbation": "NIV2SENTENCEPERTURBATION/ZS",
    "Niv2zsAnswerabilityclassification": "NIV2ANSWERABILITYCLASSIFICATION/ZS",
    "Niv2zsSummarization": "NIV2SUMMARIZATION/ZS",
    "Niv2zsCoreferenceresolution": "NIV2COREFERENCERESOLUTION/ZS",
    "Niv2zsTextqualityevaluation": "NIV2TEXTQUALITYEVALUATION/ZS",
    "Niv2zsTexttocode": "NIV2TEXTTOCODE/ZS",
    "Niv2zsParaphrasing": "NIV2PARAPHRASING/ZS",
    "Niv2zsDialoguegeneration": "NIV2DIALOGUEGENERATION/ZS",
    "Niv2zsQuestionrewriting": "NIV2QUESTIONREWRITING/ZS",
    "Niv2zsWordsemantics": "NIV2WORDSEMANTICS/ZS",
    "Niv2zsPostagging": "NIV2POSTAGGING/ZS",
    "Niv2zsLinguisticprobing": "NIV2LINGUISTICPROBING/ZS",
    "Niv2zsStorycomposition": "NIV2STORYCOMPOSITION/ZS",
    "Niv2zsSpeakeridentification": "NIV2SPEAKERIDENTIFICATION/ZS",
    "Niv2zsWordanalogy": "NIV2WORDANALOGY/ZS",
    "Niv2zsDatatotext": "NIV2DATATOTEXT/ZS",
    "Niv2zsStereotypedetection": "NIV2STEREOTYPEDETECTION/ZS",
    "Niv2zsNegotiationstrategydetection": "NIV2NEGOTIATIONSTRATEGYDETECTION/ZS",
    "Niv2zsDialogueactrecognition": "NIV2DIALOGUEACTRECOGNITION/ZS",
    "Niv2zsGenderclassification": "NIV2GENDERCLASSIFICATION/ZS",
    "Niv2zsCoherenceclassification": "NIV2COHERENCECLASSIFICATION/ZS",
    "Niv2zsExplanation": "NIV2EXPLANATION/ZS",
    "Niv2zsEthicsclassification": "NIV2ETHICSCLASSIFICATION/ZS",
    "Niv2zsWordrelationclassification": "NIV2WORDRELATIONCLASSIFICATION/ZS",
    "Niv2zsSentenceordering": "NIV2SENTENCEORDERING/ZS",
    "Niv2zsAnswerverification": "NIV2ANSWERVERIFICATION/ZS",
    "Niv2zsMathematics": "NIV2MATHEMATICS/ZS",
    "Niv2zsIntentidentification": "NIV2INTENTIDENTIFICATION/ZS",
    "Niv2zsKeywordtagging": "NIV2KEYWORDTAGGING/ZS",
    "Niv2zsCodetotext": "NIV2CODETOTEXT/ZS",
    "Niv2zsDialoguestatetracking": "NIV2DIALOGUESTATETRACKING/ZS",
    "Niv2zsTextsimplification": "NIV2TEXTSIMPLIFICATION/ZS",
    "Niv2zsStancedetection": "NIV2STANCEDETECTION/ZS",
    "Niv2zsFactverification": "NIV2FACTVERIFICATION/ZS",
    "Niv2zsGrammarerrordetection": "NIV2GRAMMARERRORDETECTION/ZS",
    "Niv2zsSectionclassification": "NIV2SECTIONCLASSIFICATION/ZS",
    "Niv2zsNumberconversion": "NIV2NUMBERCONVERSION/ZS",
    "Niv2zsStyletransfer": "NIV2STYLETRANSFER/ZS",
    "Niv2zsSpeakerrelationclassification": "NIV2SPEAKERRELATIONCLASSIFICATION/ZS",
    "Niv2zsIronydetection": "NIV2IRONYDETECTION/ZS",
    "Niv2zsQuestiondecomposition": "NIV2QUESTIONDECOMPOSITION/ZS",
    "Niv2zsOverlapextraction": "NIV2OVERLAPEXTRACTION/ZS",
    "Niv2zsGrammarerrorcorrection": "NIV2GRAMMARERRORCORRECTION/ZS",
    "Niv2zsSpellingerrordetection": "NIV2SPELLINGERRORDETECTION/ZS",
    "Niv2zsEntitygeneration": "NIV2ENTITYGENERATION/ZS",
    "Niv2zsSentenceexpansion": "NIV2SENTENCEEXPANSION/ZS",
    "Niv2zsDiscourseconnectiveidentification": "NIV2DISCOURSECONNECTIVEIDENTIFICATION/ZS",
    "Niv2zsDiscourserelationclassification": "NIV2DISCOURSERELATIONCLASSIFICATION/ZS",
    "Niv2zsPoemgeneration": "NIV2POEMGENERATION/ZS",
    "Niv2zsEntityrelationclassification": "NIV2ENTITYRELATIONCLASSIFICATION/ZS",
    "Niv2zsPunctuationerrordetection": "NIV2PUNCTUATIONERRORDETECTION/ZS",
    "Niv2zsSpamclassification": "NIV2SPAMCLASSIFICATION/ZS",
    "Niv2zsPaperreview": "NIV2PAPERREVIEW/ZS",
    "Niv2zsSentencecompression": "NIV2SENTENCECOMPRESSION/ZS",
    "Niv2zsPrepositionprediction": "NIV2PREPOSITIONPREDICTION/ZS",
    "Niv2zsMisc": "NIV2MISC/ZS",
}

add1_dataset_dict = {
    "P3Wscfixed": "P3WSCFIXED",
    "P3Copa": "P3COPA",
    "P3Hswag": "P3HSWAG",
    "P3Wic": "P3WIC",
    "P3Racehigh": "P3RACEHIGH",
    "P3Racemiddle": "P3RACEMIDDLE",
    "P3Webquestions": "P3WEBQUESTIONS",
    "Flanv2zsQrecc": "DIALOGQRECC/ZS",
    "Flanv2zsWikidialog": "DIALOGWIKIDIALOG/ZS",
    "Flanv2zsQreccii": "DIALOGQRECCII/ZS",
    "Flanv2zsWikidialogii": "DIALOGWIKIDIALOGII/ZS",
    "Flanv2zsAeslc": "FLAN2021AESLC/ZS",
    "Flanv2zsWmt16translatecsen": "FLAN2021WMT16TRANSLATECSEN/ZS",
    "Flanv2zsWmt16translatedeen": "FLAN2021WMT16TRANSLATEDEEN/ZS",
    "Flanv2zsWmt16translateruen": "FLAN2021WMT16TRANSLATERUEN/ZS",
    "Flanv2zsWmt16translatefien": "FLAN2021WMT16TRANSLATEFIEN/ZS",
    "Flanv2zsWmt16translateroen": "FLAN2021WMT16TRANSLATEROEN/ZS",
    "Flanv2zsWmt16translatetren": "FLAN2021WMT16TRANSLATETREN/ZS",
    "Flanv2zsWmt14translatefren": "FLAN2021WMT14TRANSLATEFREN/ZS",
}

all_dataset_dict = {**p3_dataset_dict, **flanv2_dataset_dict, **niv2_dataset_dict}
full_dataset_dict = {
    **p3_dataset_dict,
    **flanv2_dataset_dict,
    **niv2_dataset_dict,
    **add1_dataset_dict,
}


@gin.configurable
def average_lora_weights(path, out_path):
    exp_path = f"exp_out/{path}"
    checkpoint = torch.load(f"{exp_path}/best.pt")
    out_path = f"exp_out/{out_path}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    new_checkpoint = checkpoint.copy()
    # delete router_embeddings if present
    for key in checkpoint.keys():
        if "expert_embeddings" in key:
            if key.endswith("__0"):
                print(f"Modifying {key} to zeros")
                new_checkpoint[key] = torch.zeros_like(checkpoint[key])
            else:
                print(f"Deleting {key}")
                del new_checkpoint[key]
    count = {}
    for key in checkpoint.keys():
        if "expert_lora" in key:
            key_prefix, index = key.split("__")
            if index == "0":
                count[key] = 1
                continue
            del new_checkpoint[key]
            new_key = f"{key_prefix}__0"
            new_checkpoint[new_key] = new_checkpoint[new_key] + checkpoint[key]
            count[new_key] += 1
    print(f"Count dict is {count}")
    for key in new_checkpoint.keys():
        if "expert_lora" in key:
            new_checkpoint[key] = new_checkpoint[key] / count[key]

    print(f"Saving to {out_path}")
    torch.save(new_checkpoint, f"{out_path}/best.pt")


def process_key_eigen(key, checkpoint, method_type, backbone_weight_dict=None):
    lora_a = checkpoint[key]
    lora_b = checkpoint[key.replace("layer1", "layer2")]
    lora_weight = torch.matmul(lora_a, lora_b).float()
    routing_key = key.replace("expert_lora.layer1", "router.expert_embeddings")
    U_lora, S_lora, V_lora = torch.svd(lora_weight)
    if method_type == "eigen_weighted_vector":
        S_weights = S_lora / torch.sum(S_lora)
    elif method_type == "top1_eigen_weighted_vector":
        S_weights = torch.zeros_like(S_lora)
        S_weights[0] = 1
    elif method_type == "backbone_eigen_weighted_vector":
        pass
    new_value = torch.sum(torch.matmul(U_lora, torch.diag(S_weights)), dim=1)
    return routing_key, new_value


def process_keys_batch(keys_batch, checkpoint, method_type, backbone_weight_dict=None):
    results = []
    for key in keys_batch:
        result = process_key_eigen(key, checkpoint, method_type, backbone_weight_dict)
        if result[0]:  # Only add if routing_key is not None
            results.append(result)
        else:
            raise ValueError(f"Routing key is None for {key}")
    return results


@gin.configurable
def create_router_embeddings_from_weights(path, out_path, method_type="simple"):
    # method_types are simple, eigen_weighted_vector, backbone_eigen_weighted_vector
    exp_path = f"exp_out/{path}"
    checkpoint = torch.load(f"{exp_path}/best.pt")
    if method_type == "backbone_eigen_weighted_vector":
        backbone_model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-xl-lm-adapt")
        backbone_weight_dict = backbone_model.state_dict()
    else:
        backbone_weight_dict = None
    out_path = f"exp_out/{out_path}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    new_checkpoint = checkpoint.copy()
    # delete router_embeddings if present
    for key in checkpoint.keys():
        if "expert_embeddings" in key:
            print(f"Deleting {key}")
            del new_checkpoint[key]
    if method_type == "simple":
        for key in checkpoint.keys():
            if "layer1" in key:
                routing_key = key.replace(
                    "expert_lora.layer1", "router.expert_embeddings"
                )
                print(f"Adding {routing_key} of shape {new_checkpoint[key].shape[0]}")
                new_checkpoint[routing_key] = torch.mean(new_checkpoint[key], dim=1)
    elif (
        method_type == "eigen_weighted_vector"
        or method_type == "backbone_eigen_weighted_vector"
        or method_type == "top1_eigen_weighted_vector"
    ):
        # combine layer 1 and layer 2 to create a matrix and take weighted eigen vector
        keys = [key for key in checkpoint.keys() if "layer1" in key]
        batch_size = 10
        batches = [keys[i : i + batch_size] for i in range(0, len(keys), batch_size)]
        with ThreadPoolExecutor() as executor:
            # Submit batch tasks to the executor
            future_to_batch = {
                executor.submit(
                    process_keys_batch,
                    batch,
                    checkpoint,
                    method_type,
                    backbone_weight_dict,
                ): batch
                for batch in batches
            }

            # Process results as they are completed
            for future in as_completed(future_to_batch):
                results = future.result()
                for routing_key, new_value in results:
                    print(f"Adding {routing_key} of shape {new_value.shape[0]}")
                    new_checkpoint[routing_key] = new_value
        # for key in checkpoint.keys():
        #     if "layer1" in key:
        #         lora_a = checkpoint[key]
        #         lora_b = checkpoint[key.replace("layer1", "layer2")]
        #         lora_weight = torch.matmul(lora_a, lora_b).float()
        #         routing_key = key.replace(
        #             "expert_lora.layer1", "router.expert_embeddings"
        #         )
        #         print(f"Adding {routing_key} of shape {new_checkpoint[key].shape[0]}")
        #         U_lora, S_lora, V_lora = torch.svd(lora_weight)
        #         if method_type == "eigen_weighted_vector":
        #             S_weights = S_lora / torch.sum(S_lora)
        #         elif method_type == "top1_eigen_weighted_vector":
        #             S_weights = torch.zeros_like(S_lora)
        #             S_weights[0] = 1
        #         elif method_type == "backbone_eigen_weighted_vector":
        #             backbone_weight = backbone_weight_dict[
        #                 f"{key.split('._addons')[0]}.weight"
        #             ]
        #             U_backbone, S_backbone, V_backbone = torch.svd(backbone_weight)
        #             # need to do matching of eigen vector of U_lora and U_backbone
        #             matching_weights = torch.matmul(U_lora.T, U_backbone)
        #             matching_weights[matching_weights < 0] = 0
        #             S_matched_backbone = (
        #                 torch.matmul(matching_weights, S_backbone) + 1e-6
        #             )
        #             if (
        #                 torch.sort(S_matched_backbone, descending=True)
        #                 != S_matched_backbone
        #             ):
        #                 print(f"backbone weighting is useful")
        #                 import ipdb

        #                 ipdb.set_trace()
        #             S_weights = S_lora / S_matched_backbone
        #             S_weights = S_weights / torch.sum(S_weights)
        #         new_checkpoint[routing_key] = torch.sum(
        #             torch.matmul(U_lora, torch.diag(S_weights)), dim=1
        #         )

    print(f"Saving to {out_path}")
    torch.save(new_checkpoint, f"{out_path}/best.pt")


@gin.configurable
def make_one_router_decoder(
    path, out_path, matching_strings=["EncDecAttention.k", "EncDecAttention.v"]
):
    out_path = f"exp_out/{out_path}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    checkpoint = torch.load(f"exp_out/{path}/best.pt")
    new_checkpoint = checkpoint.copy()
    router_embeddings = {}
    for key in checkpoint:
        for matching_string in matching_strings:
            if matching_string in key and "expert_embeddings" in key:
                print(f"Counting {key}")
                key_index = key.split("__")[1]
                if key_index in router_embeddings:
                    router_embeddings[key_index][0] += checkpoint[key]
                    router_embeddings[key_index][1] += 1
                else:
                    router_embeddings[key_index] = [checkpoint[key], 1]
    for index in router_embeddings:
        print(f"Count for {index} is {router_embeddings[index][1]}")
        router_embeddings[index] = (
            router_embeddings[index][0] / router_embeddings[index][1]
        )

    for key in checkpoint:
        if "decoder" in key and "expert_embeddings" in key:
            print(f"Deleting {key}")
            del new_checkpoint[key]

    for index in router_embeddings:
        new_key = f"decoder._addons.router.expert_embeddings__{index}"
        print(f"Adding {new_key}")
        new_checkpoint[new_key] = router_embeddings[index]

    print(f"Saving to {out_path}")
    torch.save(new_checkpoint, f"{out_path}/best.pt")


@gin.configurable
def make_a_specific_checkpoint(path1, path2, out_path):
    out_path = f"exp_out/{out_path}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    checkpoint1 = torch.load(f"exp_out/{path1}/best.pt")
    checkpoint2 = torch.load(f"exp_out/{path2}/best.pt")
    new_checkpoint = checkpoint1.copy()
    print(f"deleting decoder expert embeddings from {path1}")
    for key in checkpoint1.keys():
        if "decoder" in key and "expert_embeddings" in key:
            del new_checkpoint[key]
    for key in checkpoint2.keys():
        if "decoder" in key and "expert_embeddings" in key:
            print(f"adding {key} from {path2}")
            new_checkpoint[key] = checkpoint2[key]
    print(f"Saving to {out_path}")
    torch.save(new_checkpoint, f"{out_path}/best.pt")


@gin.configurable
def use_input_gate_as_router(path):
    exp_path = f"exp_out/{path}"
    checkpoint = torch.load(f"{exp_path}/best.pt")
    new_checkpoint = checkpoint.copy()
    for key in checkpoint.keys():
        if "input_gate" in key:
            new_key = key.replace(
                "expert_lora.expert_input_gate", "router.expert_embeddings__0"
            )
            print(f"Using {key} as {new_key}")
            new_checkpoint[new_key] = checkpoint[key]
    print(f"Saving to {exp_path}")
    torch.save(new_checkpoint, f"{exp_path}/best.pt")


@gin.configurable
def use_input_gate_in_router_and_expert(path):
    exp_path = f"exp_out/{path}"
    checkpoint = torch.load(f"{exp_path}/best.pt")
    new_checkpoint = checkpoint.copy()
    for key in checkpoint.keys():
        if "input_gate" in key:
            routing_key = key.replace(
                "expert_lora.expert_input_gate", "router.expert_embeddings__0"
            )
            print(f"Adding {routing_key}")
            new_checkpoint[routing_key] = checkpoint[key]
            expert_key1 = key.replace(
                "expert_lora.expert_input_gate", "expert_lora.layer1__0"
            )
            new_expert_layer1 = torch.cat(
                (checkpoint[key].unsqueeze(1), checkpoint[expert_key1]), dim=1
            )
            print(f"Adding {expert_key1}")
            new_checkpoint[expert_key1] = new_expert_layer1
            expert_key2 = key.replace(
                "expert_lora.expert_input_gate", "expert_lora.layer2__0"
            )
            new_expert_layer2 = torch.cat(
                (
                    torch.zeros_like(checkpoint[expert_key2][0].unsqueeze(0)),
                    checkpoint[expert_key2],
                ),
                dim=0,
            )
            print(f"Adding {expert_key2}")
            new_checkpoint[expert_key2] = new_expert_layer2
    print(f"Saving to {exp_path}")
    torch.save(new_checkpoint, f"{exp_path}/best.pt")


@gin.configurable
def delete_expert_input_gate(path, out_path):
    exp_path = f"exp_out/{path}"
    out_path = f"exp_out/{out_path}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    checkpoint = torch.load(f"{exp_path}/best.pt")
    new_checkpoint = checkpoint.copy()
    for key in checkpoint.keys():
        if "input_gate" in key:
            print(f"Deleting {key}")
            del new_checkpoint[key]
    print(f"Saving to {out_path}")
    torch.save(new_checkpoint, f"{out_path}/best.pt")


@gin.configurable
def put_index_to_lora(path, out_path):
    exp_path = f"exp_out/{path}"
    checkpoint = torch.load(f"{exp_path}/best.pt")
    out_path = f"exp_out/{out_path}"
    new_checkpoint = checkpoint.copy()
    # delete router_embeddings if present
    for key in checkpoint.keys():
        if "expert_embeddings" in key:
            print(f"Deleting {key}")
            del new_checkpoint[key]

    for key in checkpoint.keys():
        if "expert_lora_a" in key:
            new_key = key.replace("expert_lora_a", "_addons.expert_lora.layer1__0")
            print(f"Convert {key} as {new_key}")
            new_checkpoint[new_key] = checkpoint[key].T
            del new_checkpoint[key]
            # add router_embedding
            router_key = key.replace(
                "expert_lora_a", "_addons.router.expert_embeddings__0"
            )
            print(f"Adding {router_key} of shape {new_checkpoint[new_key].shape[0]}")
            new_checkpoint[router_key] = torch.zeros(new_checkpoint[new_key].shape[0])
        elif "expert_lora_b" in key:
            new_key = key.replace("expert_lora_b", "_addons.expert_lora.layer2__0")
            print(f"Convert {key} as {new_key}")
            new_checkpoint[new_key] = checkpoint[key].T
            del new_checkpoint[key]

    for key in checkpoint.keys():
        if "expert_lora.layer1__0" in key:
            router_key = key.replace(
                "expert_lora.layer1__0", "router.expert_embeddings__0"
            )
            print(f"Adding {router_key} of shape {new_checkpoint[key].shape[0]}")
            new_checkpoint[router_key] = torch.zeros(new_checkpoint[key].shape[0])

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(f"Saving to {out_path}")
    torch.save(new_checkpoint, f"{out_path}/best.pt")


@gin.configurable
def make_checkpoint_for_retrieval(
    checkpoint_path,
    embeddings_path,
    out_path,
    dataset_length=-1,
    dataset_dict_str="All",
):
    if dataset_dict_str == "P3":
        dataset_dict = p3_dataset_dict
    elif dataset_dict_str == "Flanv2":
        dataset_dict = flanv2_dataset_dict
    elif dataset_dict_str == "Niv2":
        dataset_dict = niv2_dataset_dict
    elif dataset_dict_str == "All":
        dataset_dict = all_dataset_dict
    elif dataset_dict_str == "Full":
        dataset_dict = full_dataset_dict
    exp_path = f"exp_out/{checkpoint_path}"
    checkpoint = torch.load(f"{exp_path}/best.pt")
    embeddings_path = f"exp_out/{embeddings_path}"
    out_path = f"exp_out/{out_path}"
    # delete any existing router embeddings
    new_checkpoint = checkpoint.copy()
    for key in checkpoint.keys():
        if "expert_embeddings" in key:
            del new_checkpoint[key]
            print(f"Deleting {key}")
    # load the embeddings
    train_dataset_embeddings = []
    for dataset in dataset_dict:
        dataset_name = dataset_dict[dataset].replace("/", "_")
        if dataset_length == -1:
            filename = f"D_{dataset_name}_TRAIN_embeddings.npy"
        else:
            filename = f"D_{dataset_name}_TRAIN_K{dataset_length}_embeddings.npy"
        with open(os.path.join(embeddings_path, filename), "rb") as f:
            train_dataset_embeddings.append(np.load(f))
    # pad each of train_dataset_embeddings according to the largest size of individual ones
    max_size = max([x.shape[0] for x in train_dataset_embeddings])
    for index, embedding in enumerate(train_dataset_embeddings):
        if embedding.shape[0] < max_size:
            train_dataset_embeddings[index] = np.pad(
                embedding, ((0, max_size - embedding.shape[0]), (0, 0)), "constant"
            )
    train_dataset_embeddings = np.stack(train_dataset_embeddings, axis=0)
    retriever_router = torch.tensor(train_dataset_embeddings, dtype=torch.float32)
    retriever_router = retriever_router.reshape(len(dataset_dict), -1)
    for i in range(len(dataset_dict)):
        new_key = f"encoder._addons.router.expert_embeddings__{i}"
        print(f"Adding {new_key}")
        new_checkpoint[new_key] = retriever_router[i]
    print(f"router dimension is {retriever_router.shape}")
    print(f"Saving to {out_path}")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    torch.save(new_checkpoint, f"{out_path}/best.pt")


@gin.configurable
def concatenate_two_checkpoints(path1, path2, out_path, index):
    out_path = f"exp_out/{out_path}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    checkpoint1 = torch.load(f"exp_out/{path1}/best.pt")
    checkpoint2 = torch.load(f"exp_out/{path2}/best.pt")
    new_checkpoint = checkpoint1.copy()
    for key in checkpoint2.keys():
        if "__" in key:
            prefix, key_index = key.split("__")
            new_index = index + int(key_index)
            new_key = f"{prefix}__{new_index}"
            print(f"Adding {new_key}")
            new_checkpoint[new_key] = checkpoint2[key]
    print(f"Saving to {out_path}")
    torch.save(new_checkpoint, f"{out_path}/best.pt")


@gin.configurable
def create_router_embeddings(path, regex_pattern_list, out_path=None):
    all_model_shortcut_dict = {
        "encoder": r"^encoder",
        "encoder_block": r"^encoder\.block\.\d+",
        "encoder_sublayer": r"^encoder\.block\.\d+\.layer\.\d+\.(SelfAttention|DenseReluDense)",
        "encoder_linear": r"^encoder\.block\.\d+\.layer\.\d+\.(SelfAttention|DenseReluDense)\.(q|k|v|o|wi_\d|wo)",
        "encoder_final_ln": r"^encoder.final_layer_norm",
        "decoder": r"^decoder",
        "decoder_block": r"^decoder\.block\.\d+",
        "decoder_sublayer": r"decoder\.block\.\d+\.layer\.\d+\.(SelfAttention|EncDecAttention|DenseReluDense)",
        "decoder_linear": r"^decoder\.block\.\d+\.layer\.\d+\.(SelfAttention|EncDecAttention|DenseReluDense)\.(q|k|v|o|wi_\d|wo)",
        "decoder_final_ln": r"decoder.final_layer_norm",
    }
    for index, regex_pattern in enumerate(regex_pattern_list):
        if regex_pattern in all_model_shortcut_dict:
            regex_pattern_list[index] = all_model_shortcut_dict[regex_pattern]
    regex_pattern = "|".join(regex_pattern_list)
    exp_path = f"exp_out/{path}"
    if out_path:
        out_path = f"exp_out/{out_path}"
    else:
        out_path = exp_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    checkpoint = torch.load(f"{exp_path}/best.pt", map_location="cpu")
    new_checkpoint = checkpoint.copy()
    for key in checkpoint.keys():
        if "expert_embeddings" in key:
            print(f"Deleting {key}")
            del new_checkpoint[key]
    for key in checkpoint.keys():
        if "expert" in key and "gate" not in key:
            match = re.match(regex_pattern, key)
            if match:
                prefix = match.group(0)
                new_key = f"{prefix}._addons.router.expert_embeddings__0"
                if new_key not in new_checkpoint:
                    print(f"Adding {new_key} of size {checkpoint[key].shape[0]}")
                    new_checkpoint[new_key] = torch.zeros(checkpoint[key].shape[0])

    print(f"Saving to {out_path}")
    torch.save(new_checkpoint, f"{out_path}/best.pt")


@gin.configurable
def delete_router_embeddings(path, out_path=None):
    exp_path = f"exp_out/{path}"
    if out_path:
        out_path = f"exp_out/{out_path}"
    else:
        out_path = exp_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    checkpoint = torch.load(f"{exp_path}/best.pt")
    new_checkpoint = checkpoint.copy()
    for key in checkpoint.keys():
        if "expert_embeddings" in key:
            print(f"Deleting {key}")
            del new_checkpoint[key]
    print(f"Saving to {out_path}")
    torch.save(new_checkpoint, f"{out_path}/best.pt")


@gin.configurable
def normalize_expert_weights(
    path, out_path, index=None, norm_value=None, keys_to_ignore=[]
):
    out_path = f"exp_out/{out_path}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    checkpoint = torch.load(f"exp_out/{path}/best.pt")
    for key in checkpoint.keys():
        if "expert_lora.layer" in key:
            if index is not None and not key.endswith(f"__{index}"):
                # print(f"Skipping {key} since it does not match index {index}")
                continue
            skip = False
            for key_to_ignore in keys_to_ignore:
                if key_to_ignore in key:
                    # print(f"Skipping {key} since it matches {key_to_ignore}")
                    skip = True
                    break
            if skip:
                continue
            norm = torch.linalg.matrix_norm(checkpoint[key])
            if norm_value is not None and norm > norm_value:
                print(f"Normalizing {key} of norm {norm}")
                checkpoint[key] = checkpoint[key] / norm
                checkpoint[key] = norm_value * checkpoint[key]
    print(f"Saving to {out_path}")
    torch.save(checkpoint, f"{out_path}/best.pt")


@gin.configurable
def add_router_embeddings(list_of_paths, out_path):
    out_path = f"exp_out/{out_path}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    final_checkpoint = torch.load(f"exp_out/{list_of_paths[0]}/best.pt")
    for path in list_of_paths[1:]:
        exp_path = f"exp_out/{path}"
        print(f"loading from {path}")
        current_checkpoint = torch.load(f"{exp_path}/best.pt")
        for key in current_checkpoint.keys():
            if "expert_embeddings" in key:
                final_checkpoint[key] += current_checkpoint[key]
    torch.save(final_checkpoint, f"{out_path}/best.pt")


@gin.configurable
def skip_alternate(path, out_path, ffn_exp):
    exp_path = f"exp_out/{path}"
    out_path = f"exp_out/{out_path}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    checkpoint = torch.load(f"{exp_path}/best.pt")
    new_checkpoint = checkpoint.copy()
    for key in checkpoint.keys():
        if ffn_exp:
            if "DenseReluDense" not in key:
                print(f"Skipping {key}")
                del new_checkpoint[key]
        else:
            if "DenseReluDense" in key:
                print(f"Skipping {key}")
                del new_checkpoint[key]
    torch.save(new_checkpoint, f"{out_path}/best.pt")


@gin.configurable
def merge_into_blocks(path, out_path):
    exp_path = f"exp_out/{path}"
    out_path = f"exp_out/{out_path}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    checkpoint = torch.load(f"{exp_path}/best.pt")
    encoder_block_pattern = r"^encoder\.block\.\d+"
    decoder_block_pattern = r"^decoder\.block\.\d+"
    index_pattern = r"_addons.router.expert_embeddings__\d+"
    new_checkpoint = checkpoint.copy()
    new_dict = {}
    for key in checkpoint.keys():
        if "expert_embeddings" in key:
            print(f"Skipping {key}")
            del new_checkpoint[key]
            encoder_match = re.match(encoder_block_pattern, key)
            decoder_match = re.match(decoder_block_pattern, key)
            if encoder_match:
                block = encoder_match.group(0)
            elif decoder_match:
                block = decoder_match.group(0)
            else:
                raise ValueError(f"Key {key} does not match block pattern")
            # since matching in the end, we use re.search
            index_match = re.search(index_pattern, key)
            if not index_match:
                raise ValueError(f"Key {key} does not match index pattern")
            index = index_match.group(0)
            new_key = f"{block}.{index}"
            if new_key not in new_dict:
                new_dict[new_key] = [checkpoint[key], 1]
            else:
                new_dict[new_key][0] += checkpoint[key]
                new_dict[new_key][1] += 1

    print(f"Done merging keys, adding them to checkpoint....")
    for key, value in new_dict.items():
        print(f"Adding {key}")
        new_checkpoint[key] = value[0] / value[1]

    torch.save(new_checkpoint, f"{out_path}/best.pt")


@gin.configurable
def make_router_blockwise(path, out_path=None, index=None):
    exp_path = f"exp_out/{path}"
    if out_path:
        out_path = f"exp_out/{out_path}"
    else:
        out_path = exp_path.replace("lora", "lora_block")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    checkpoint = torch.load(f"{exp_path}/best.pt")
    encoder_block_pattern = r"^encoder\.block\.\d+"
    decoder_block_pattern = r"^decoder\.block\.\d+"
    if index:
        index_pattern = f"_addons.router.expert_embeddings__{index}"
    else:
        index_pattern = r"_addons.router.expert_embeddings__\d+"
    new_checkpoint = checkpoint.copy()
    new_dict = {}
    for key in checkpoint.keys():
        if "expert_embeddings" in key:
            encoder_match = re.match(encoder_block_pattern, key)
            decoder_match = re.match(decoder_block_pattern, key)
            if encoder_match:
                block = encoder_match.group(0)
            elif decoder_match:
                block = decoder_match.group(0)
            else:
                raise ValueError(f"Key {key} does not match block pattern")
            # since matching in the end, we use re.search
            index_match = re.search(index_pattern, key)
            if not index_match:
                raise ValueError(f"Key {key} does not match index pattern")
            index = index_match.group(0)
            new_key = f"{block}.{index}"
            if new_key not in new_dict:
                new_dict[new_key] = [checkpoint[key], 1]
            else:
                new_dict[new_key][0] += checkpoint[key]
                new_dict[new_key][1] += 1
    for key in checkpoint.keys():
        if "expert_embeddings" in key:
            encoder_match = re.match(encoder_block_pattern, key)
            decoder_match = re.match(decoder_block_pattern, key)
            if encoder_match:
                block = encoder_match.group(0)
            elif decoder_match:
                block = decoder_match.group(0)
            else:
                raise ValueError(f"Key {key} does not match block pattern")
            # since matching in the end, we use re.search
            index_match = re.search(index_pattern, key)
            if not index_match:
                raise ValueError(f"Key {key} does not match index pattern")
            index = index_match.group(0)
            new_key = f"{block}.{index}"
            checkpoint[key] = new_dict[new_key][0] / new_dict[new_key][1]

    torch.save(checkpoint, f"{out_path}/best.pt")


@gin.configurable
def skip_blocks(path, out_path, block_num):
    exp_path = f"exp_out/{path}"
    out_path = f"exp_out/{out_path}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    checkpoint = torch.load(f"{exp_path}/best.pt")
    # skip keys that have block num < block_num
    new_checkpoint = checkpoint.copy()
    for key in checkpoint.keys():
        pattern = r"encoder\.block\.(\d+)\.layer\.\d+"
        match = re.match(pattern, key)
        if match:
            block = match.group(1)
            if int(block) <= int(block_num):
                print(f"Skipping {key}")
                del new_checkpoint[key]
    torch.save(new_checkpoint, f"{out_path}/best.pt")


@gin.configurable
def func_caller(func):
    func()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gin_bindings", nargs="+", default=[])
    args = parser.parse_args()
    gin.parse_config(args.gin_bindings)
    func_caller()
