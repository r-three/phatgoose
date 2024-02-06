import os
from argparse import ArgumentParser
from collections import OrderedDict

import gin
import torch

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
p3_flanv2_dataset_dict = {**p3_dataset_dict, **flanv2_dataset_dict}
full_dataset_dict = {
    **p3_dataset_dict,
    **flanv2_dataset_dict,
    **niv2_dataset_dict,
    **add1_dataset_dict,
}


@gin.configurable
def commands_concatenate(
    dataset_dict,
    arch,
    suffix,
    compute_hiddens,
    datasets_path,
    hiddens_suffix,
    extra_bindings,
):
    commands = []
    if compute_hiddens:
        compute_hiddens = "True"
    else:
        compute_hiddens = "False"
    for dataset in dataset_dict:
        commands.append(
            f"bash colm/experiments/bash_scripts/reinit_router_embedding.sh -exp_name {datasets_path}/{dataset}_{suffix} -dataset {dataset_dict[dataset]} -arch {arch} -compute_hiddens {compute_hiddens} -hiddens_suffix {hiddens_suffix} -extra_bindings '{extra_bindings}'"
        )

    print(";".join(commands))


@gin.configurable
def run_concatenate(
    arch="A2",
    datasets=[],
    print_commands=False,
    suffix="lora",
    out_path="CompleteA2_lora_concatenated",
    compute_hiddens=False,
    hiddens_suffix="_a2",
    extra_bindings="",
):
    dataset_dict = {}
    if type(datasets) == str:
        if datasets == "P3":
            dataset_dict = p3_dataset_dict
        elif datasets == "Flanv2":
            dataset_dict = flanv2_dataset_dict
        elif datasets == "Niv2":
            dataset_dict = niv2_dataset_dict
        elif datasets == "All":
            dataset_dict = all_dataset_dict
        elif datasets == "P3flanv2":
            dataset_dict = p3_flanv2_dataset_dict
        elif datasets == "Add1":
            dataset_dict = add1_dataset_dict
        elif datasets == "Full":
            dataset_dict = full_dataset_dict
    else:
        for dataset in datasets:
            dataset_dict[dataset] = all_dataset_dict[dataset]
    datasets_path = "datasets_concatenated"
    if not os.path.exists(f"exp_out/{datasets_path}"):
        os.makedirs(f"exp_out/{datasets_path}")
    if print_commands:
        commands_concatenate(
            dataset_dict,
            arch,
            suffix,
            compute_hiddens,
            datasets_path,
            hiddens_suffix,
            extra_bindings,
        )
        return

    def update_checkpoint(checkpoint, index):
        new_checkpoint = OrderedDict()
        for key in checkpoint:
            if (
                key != "encoder.embed_tokens.weight"
                and key != "decoder.embed_tokens.weight"
            ):
                # replace __0 in key with __index
                if "__0" in key:
                    new_key = key.replace("__0", f"__{index}")
                elif "expert_output_gate" in key:
                    print(f"Ignoring {key}")
                    continue
                elif "expert_input_gate" in key:
                    print(f"Ignoring {key}")
                    continue
                else:
                    raise ValueError(f"Key {key} does not contain __0")
                new_checkpoint[new_key] = checkpoint[key]
            else:
                new_checkpoint[key] = checkpoint[key]
        return new_checkpoint

    checkpoints = []
    index = 0
    for dataset in dataset_dict:
        checkpoint_path = f"exp_out/{datasets_path}/{dataset}_{suffix}/best.pt"
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoints.append(
            update_checkpoint(torch.load(checkpoint_path, map_location="cpu"), index)
        )
        index += 1

    final_checkpoint = OrderedDict()
    final_checkpoint["encoder.embed_tokens.weight"] = checkpoints[0][
        "encoder.embed_tokens.weight"
    ]
    final_checkpoint["decoder.embed_tokens.weight"] = checkpoints[0][
        "decoder.embed_tokens.weight"
    ]

    # update all the other keys
    for checkpoint in checkpoints:
        for key in checkpoint:
            if (
                key != "encoder.embed_tokens.weight"
                and key != "decoder.embed_tokens.weight"
            ):
                final_checkpoint[key] = checkpoint[key]

    exp_name = out_path
    save_dir = f"exp_out/{exp_name}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = f"{save_dir}/best.pt"
    torch.save(final_checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


@gin.configurable
def func_caller(func):
    func()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gin_bindings", nargs="+", default=[])
    args = parser.parse_args()
    gin.parse_config(args.gin_bindings)
    func_caller()
