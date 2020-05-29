import csv
import os
from collections import defaultdict
from enum import Enum
from functools import partial
from typing import Callable, Tuple, Dict, Collection
import numpy as np

import console_arg_parser as arg_parser
import data_readers as dr
import adagram_grammar_extractor as gram_extractor
import baseline
import frequency_table_gen as freq
import grammar_file_generator as gram_gen
import grammar_train_file_generator as gram_train
import operation_revisor as rev
import rule_parser
import rulesynthesis.scan_search as scan_search
import rulesynthesis.synthTrain as synthTrain
from context_matrix import create_and_save_context_matrix, \
    rewrite_context_matrix_to_concept_lib_format, \
    create_grammar_context_matrix_for_concept_lib_format
from grammar_train_file_generator import SplitType
from main import process_data_file, write_alphabet, read_base_expected_words_and_morph_features
from mln_file_generator import Strictness
from precondition_interpreter import run_predictions


class PredType(Enum):
    NoOperation = 0
    AdaGramLeft = 1
    AdaGramRight = 2
    AdaGramBoth = 3
    RuleSynthHighCount = 4
    RuleSynthMediumCount = 5
    RuleSynthLowCount = 6
    RuleSynthLowCountSample = 7
    RuleSynthLowCountSampleM = 8


def _run_steps(directory_name: str, filename: str, generate_step_1: bool, generate_step_2: bool,
               generate_step_3: bool, generate_step_4: bool, generate_step_5: bool) -> None:
    file_data = filename.split("-")
    language = file_data[0]
    operation = file_data[1]
    if operation == "train" and len(file_data) > 2:
        if file_data[2] == "high":
            process_data_file(f"{directory_name}/{filename}", language, generate_step_1,
                              generate_step_2, generate_step_3, generate_step_4, generate_step_5)


def _iterate_directory(directory_name, operation: Callable[[str, str], None]) -> None:
    for filename in os.listdir(directory_name):
        operation(directory_name, filename)


def _write_alphabet_for_file(directory_name: str, filename: str):
    language = filename.split(".")[0]
    print(f"Writing alphabet for {language}")
    write_alphabet(f"data/processed/alphabet/{filename}", f"{directory_name}/{filename}")
    print(f"Finished alphabet for {language}")


def write_alphabets():
    dir_name = "data/processed/first_step"
    _iterate_directory(dir_name, _write_alphabet_for_file)


def write_steps(generate_step_1=False, generate_step_2=False, generate_step_3=False,
                generate_step_4=False, generate_step_5=False):
    directory_name = "data/latin_alphabet"
    op = partial(_run_steps, generate_step_1=generate_step_1, generate_step_2=generate_step_2,
                 generate_step_3=generate_step_3, generate_step_4=generate_step_4,
                 generate_step_5=generate_step_5)
    _iterate_directory(directory_name, op)


def write_context_matrices():
    directory_name = "data/processed"
    for filename in os.listdir(f"{directory_name}/first_step"):
        file_data = filename.split(r".")
        language = file_data[0]
        print(f"Starting work on {language}")
        first_step_file = f"{directory_name}/first_step_revised/{filename}"
        second_step_file = f"{directory_name}/second_step_revised/{filename}"
        output_path = f"{directory_name}/context_matrix/{filename}"
        create_and_save_context_matrix(output_path, first_step_file, second_step_file)

        print(f"Finished work on {language}")


def write_first_second_step_revision():
    directory_name = "data/processed"
    for filename in os.listdir(f"{directory_name}/subword"):
        language = filename.split(r".")[0]
        print(f"Starting work on {language}")
        subword_file = f"{directory_name}/subword/{filename}"
        first_step_file = f"{directory_name}/first_step/{filename}"
        second_step_file = f"{directory_name}/second_step/{filename}"
        revised_first_step_file = f"{directory_name}/first_step_revised/{filename}"
        revised_second_step_file = f"{directory_name}/second_step_revised/{filename}"
        rev.revise_steps(subword_file, first_step_file, second_step_file, revised_first_step_file,
                         revised_second_step_file)
        print(f"Finished work on {language}")


def rewrite_context_matrices():
    directory_name = "data/processed"
    for filename in os.listdir(f"{directory_name}/context_matrix"):
        language = filename.split(r".")[0]
        print(f"Starting work on {language}")
        context_matrix_file = f"{directory_name}/context_matrix/{filename}"
        output_file = f"{directory_name}/context_matrix_revised/{filename}"
        rewrite_context_matrix_to_concept_lib_format(context_matrix_file, output_file, 1)
        print(f"Finished work on {language}")


def create_grammar_context_matrices():
    directory_name = "data/processed"
    # context_matrix_file = f"{directory_name}/context_matrix_revised/danish.csv"
    # output_file = f"{directory_name}/grammar_context_matrix/danish.csv"
    # create_grammar_context_matrix_for_concept_lib_format(context_matrix_file, output_file)
    for filename in os.listdir(f"{directory_name}/context_matrix_revised"):
        language = filename.split(r".")[0]
        print(f"Starting work on {language}")
        context_matrix_file = f"{directory_name}/context_matrix_revised/{filename}"
        output_file = f"{directory_name}/grammar_context_matrix/{filename}"
        create_grammar_context_matrix_for_concept_lib_format(context_matrix_file, output_file)
        print(f"Finished work on {language}")


def reformat_sigmorphon_predictions():
    directory_name = "data/baseline_res"
    for filename in os.listdir(f"{directory_name}"):
        language, type, _ = filename.split(r"-")
        if type != "high":
            continue
        print(f"Starting work on {language}")
        sig_file = f"{directory_name}/{filename}"
        data_file = f"data/latin_alphabet/{language}-test"
        out_file = f"data/processed/predictions/sigmorphon/{language}.csv"
        baseline.format_and_save_sigmorphon_predictions(sig_file, data_file, out_file)
        print(f"Finished work on {language}")


def write_sigmorphon_baseline_cost():
    _write_baseline_cost("sigmorphon", "sigmorphon_cost")


def _write_baseline_cost(input_file_dir: str, output_file_dir: str):
    directory_name = "data/processed/predictions"
    for filename in os.listdir(f"{directory_name}/{input_file_dir}"):
        language = filename.split(r".")[0]
        print(f"Starting work on {language}")
        word_and_prediction_file = f"{directory_name}/{input_file_dir}/{filename}"
        output_file = f"{directory_name}/{output_file_dir}/{filename}"
        baseline.calculate_and_save_cost_baseline(word_and_prediction_file, output_file)
        print(f"Finished work on {language}")


def write_baseline_cost():
    _write_baseline_cost("base", "base_cost")


def _get_mean_and_standard_devs_for_languages() -> Dict[str, Tuple[float, float]]:
    directory_name = "data/latin_alphabet"
    res = {}
    for filename in os.listdir(f"{directory_name}"):
        sp = filename.split(r"-")
        language = sp[0]
        type = sp[1]
        if type != "test":
            continue
        data_file = f"{directory_name}/{filename}"
        mean, stdev = baseline.get_means_and_stdev_for_language(data_file)
        res[language] = (mean, stdev)
    return res


def _get_average_baseline_cost(cost_dir: str) -> Dict[str, float]:
    directory_name = f"data/processed/predictions/{cost_dir}"
    language_cost = {}
    for filename in os.listdir(f"{directory_name}"):
        language = filename.split(r".")[0]
        cost_file = f"{directory_name}/{filename}"
        cost = baseline.calculate_average_cost(cost_file)
        language_cost[language] = cost
    return language_cost


#
# def get_average_baseline_cost():
#     _get_average_baseline_cost("base_cost")
#
#
# def get_average_sigmorphon_cost():
#     _get_average_baseline_cost("sigmorphon_cost")


# def compare_base_to_sigmorphon():
#     base = _get_average_baseline_cost("base_cost")
#     sig = _get_average_baseline_cost("sigmorphon_cost")
#     means_stdev = _get_mean_and_standard_devs_for_languages()
#     zipped = zip(base, sig, means_stdev)
#     for item in zipped:
#         print(f"Language: {item[0][0]}")
#         print(f"Base: {item[0][1]} - Sig: {item[1][1]} ")
#         print(f"Mean: {item[2][1]}, Stdev: {item[2][2]}")

def calculate_average_prediction_costs(pred_type: PredType):
    base_dir = "data/processed/predictions"
    cost_file_name_dict = {
        PredType.AdaGramBoth: "adagram/both",
        PredType.AdaGramRight: "adagram/right",
        PredType.AdaGramLeft: "adagram/left",
        PredType.NoOperation: "no_op",
        PredType.RuleSynthLowCount: "rule_synth_low",
        PredType.RuleSynthLowCountSample: "rule_synth_low_sample",
        PredType.RuleSynthLowCountSampleM: "rule_synth_low_sample_m",
        PredType.RuleSynthMediumCount: "rule_synth_medium",
        PredType.RuleSynthHighCount: "rule_synth_high"
    }
    output_file_name_dict = {
        PredType.AdaGramBoth: "adagram_both",
        PredType.AdaGramRight: "adagram_right",
        PredType.AdaGramLeft: "adagram_left",
        PredType.NoOperation: "no_op",
        PredType.RuleSynthLowCount: "rule_synth_low",
        PredType.RuleSynthLowCountSample: "rule_synth_low_sample",
        PredType.RuleSynthLowCountSampleM: "rule_synth_low_sample_m",
        PredType.RuleSynthMediumCount: "rule_synth_medium",
        PredType.RuleSynthHighCount: "rule_synth_high"
    }
    output_file_path = f"{base_dir}/average_costs/{output_file_name_dict[pred_type]}.csv"
    languages_to_costs = _get_average_baseline_cost(f"{cost_file_name_dict[pred_type]}_cost")
    mean_and_stdev = _get_mean_and_standard_devs_for_languages()
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(output_file_path, mode="w+") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["Language", "Average cost", "Mean", "Stddev"])
        for language in languages_to_costs:
            mean, stddev = mean_and_stdev[language]
            writer.writerow([language, languages_to_costs[language], mean, stddev])


def compare_baselines(baseline_1_name: str, baseline_2_name: str):
    base_1 = _get_average_baseline_cost(baseline_1_name)
    base_2 = _get_average_baseline_cost(baseline_2_name)
    means_stdev = _get_mean_and_standard_devs_for_languages()
    for language in base_1:
        print(f"Language: {language}")
        print(f"{baseline_1_name}: {base_1[language]} - {baseline_2_name}: {base_2[language]} ")
        print(f"Mean: {means_stdev[language][0]}, Stdev: {means_stdev[language][1]}")


# def predict_language(language: str):
#     base_and_expected = read_base_and_expected_words(
#         f"data/processed/first_step_revised/{language}.csv")
#     mln_dir = "data/processed/mln"
#     weighted_preconditions = read_mln_file(
#         f"{mln_dir}/weighted/{language}.mln",
#         f"{mln_dir}/contexts/{language}.csv",
#         f"{mln_dir}/objects/{language}.csv",
#     )
#     predictions = run_predictions(base_and_expected, weighted_preconditions)
#     baseline.save_predictions(f"data/processed/predictions/mln_no_grammar/{language}.csv",
#                               predictions)
def calculate_average_costs_across_rule_counts_by_lang():
    pred_dir = "data/processed/prediction_costs"
    for item in os.listdir(pred_dir):
        cost_folder = os.path.join(pred_dir, item)
        if not os.path.isdir(cost_folder):
            continue
        output_file = f"data/processed/average_pred_costs/{item}.csv"
        _calculate_average_cost_across_rule_counts_folder(cost_folder, output_file)


def calculate_average_costs_across_rule_counts_combined():
    data = []
    header = []
    dir = "data/processed/average_pred_costs"
    for item in os.listdir(dir):
        if item == "combined.csv":
            continue
        with open(os.path.join(dir, item), mode="r") as file:
            reader = csv.reader(file)
            header = next(reader)
            for line in reader:
                line[0] = f"{line[0]}_{item[:-4]}"
                data.append(line)
    with open(os.path.join(dir, "combined.csv"), mode="w+") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)


def _calculate_average_cost_across_rule_counts_folder(cost_folder: str, output_file: str):
    languagecounts = defaultdict(lambda: defaultdict(lambda: 0.0))
    counts = set()
    for item in os.listdir(cost_folder):
        language, rule_count = item.split("_")
        rule_count = float(rule_count[:-4])
        counts.add(rule_count)
        item_path = os.path.join(cost_folder, item)
        cost = baseline.calculate_average_cost(item_path)
        languagecounts[language][str(rule_count)] = cost

    with open(output_file, mode="w+")as file:
        writer = csv.writer(file)
        ordered_counts = [str(c) for c in sorted(counts)]
        writer.writerow(["language"] + ordered_counts)
        for language in languagecounts:
            row = [language]
            for i in ordered_counts:
                row.append(languagecounts[language][i])
            writer.writerow(row)


def calculate_combined_average_cost():
    pass


# def calculate_average_costs():
#     output_file_path = f"{base_dir}/average_costs/{output_file_name_dict[pred_type]}.csv"
#     languages_to_costs = _get_average_baseline_cost(f"{cost_file_name_dict[pred_type]}_cost")
#     mean_and_stdev = _get_mean_and_standard_devs_for_languages()
#     output_dir = os.path.dirname(output_file_path)
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#     with open(output_file_path, mode="w+") as file:
#         writer = csv.writer(file, delimiter=",")
#         writer.writerow(["Language", "Average cost", "Mean", "Stddev"])
#         for language in languages_to_costs:
#             mean, stddev = mean_and_stdev[language]
#             writer.writerow([language, languages_to_costs[language], mean, stddev])


def calculate_costs():
    dir = "data/processed/predictions"
    for pred_folder in os.listdir(dir):
        pred_folder_path = os.path.join(dir, pred_folder)
        if not os.path.isdir(pred_folder_path):
            continue
        for file in os.listdir(pred_folder_path):
            input_path = os.path.join(pred_folder_path, file)
            output_path = f"data/processed/prediction_costs/{pred_folder}/{file}"
            baseline.calculate_and_save_cost_baseline(input_path, output_path)


def predict_rulesynth_results(
        top_quality_perc: float, morph_feature_comparison_strictness: Strictness
):
    dir = "data/processed/models/results"
    for item in os.listdir(dir):
        item_path = os.path.join(dir, item)
        if not os.path.isdir(item_path):
            continue
        output_dir = f"data/processed/predictions/{item}"
        print(f"Predicting {item}")
        _predict_rulesynth_folder(top_quality_perc, morph_feature_comparison_strictness, item_path,
                                  output_dir)


def _predict_rulesynth_folder(
        top_quality_perc: float,
        morph_feature_comparison_strictness: Strictness,
        input_dir: str,
        output_dir: str
):
    dr.mkdir_if_not_exists(output_dir)
    languages = set()
    for file in os.listdir(input_dir):
        language, _ = file.split("_")
        languages.add(language)
    for language in languages:
        _predict_rule_synth_language(
            top_quality_perc,
            morph_feature_comparison_strictness,
            language,
            input_dir,
            output_dir
        )


def _predict_rule_synth_language(
        top_quality_perc: float,
        morph_feature_comparison_strictness: Strictness,
        language: str,
        input_dir,
        output_dir: str
):
    base_expected_and_morphs = read_base_expected_words_and_morph_features(
        f"data/latin_alphabet/{language}-test"
    )
    weighted_ops = rule_parser.parse_combine_rules(input_dir, language, top_quality_perc)
    is_probabilistic = True
    predictions = run_predictions(base_expected_and_morphs, weighted_ops, is_probabilistic,
                                  morph_feature_comparison_strictness)
    baseline.save_predictions(os.path.join(output_dir, f"{language}_{top_quality_perc}.csv"),
                              predictions)


def predict_language(
        language: str,
        pred_type: PredType,
        morph_feature_comparison_strictness: Strictness
):
    top_quality_perc = 0.4
    base_expected_and_morphs = read_base_expected_words_and_morph_features(
        f"data/latin_alphabet/{language}-test")
    grammar_dir = "data/processed/grammar"
    if pred_type == PredType.AdaGramRight:
        pred_type_path_change = "adagram/right"
        grammar_file = f"{grammar_dir}/{pred_type_path_change}/{language}.grammar"
        weighted_ops = gram_extractor.process_grammar_file(grammar_file)
        is_probabilistic = True
    elif pred_type == PredType.AdaGramLeft:
        pred_type_path_change = "adagram/left"
        grammar_file = f"{grammar_dir}/{pred_type_path_change}/{language}.grammar"
        weighted_ops = gram_extractor.process_grammar_file(grammar_file)
        is_probabilistic = True
    elif pred_type == PredType.AdaGramBoth:
        pred_type_path_change = "adagram/both"
        grammar_file = f"{grammar_dir}/{pred_type_path_change}/{language}.grammar"
        weighted_ops = gram_extractor.process_grammar_file(grammar_file)
        is_probabilistic = True
    elif pred_type == PredType.NoOperation:
        pred_type_path_change = "no_op"
        weighted_ops = ()
        is_probabilistic = False
    elif pred_type == PredType.RuleSynthLowCount:
        pred_type_path_change = "rule_synth_low"
        rule_file = f"{language}"
        rule_dir = f"data/processed/models/results/low_rule"
        weighted_ops = rule_parser.parse_combine_rules(rule_dir, rule_file, top_quality_perc)
        is_probabilistic = True
    elif pred_type == PredType.RuleSynthLowCountSample:
        pred_type_path_change = "rule_synth_low_sample"
        rule_file = f"{language}"
        rule_dir = f"data/processed/models/results/low_rule_sample"
        weighted_ops = rule_parser.parse_combine_rules(rule_dir, rule_file, top_quality_perc)
        is_probabilistic = True
    elif pred_type == PredType.RuleSynthLowCountSampleM:
        pred_type_path_change = "rule_synth_low_sample_m"
        rule_file = f"{language}"
        rule_dir = f"data/processed/models/results/low_rule_sample_m"
        weighted_ops = rule_parser.parse_combine_rules(rule_dir, rule_file, top_quality_perc)
        is_probabilistic = True
    elif pred_type == PredType.RuleSynthMediumCount:
        pred_type_path_change = "rule_synth_medium"
        rule_file = f"{language}"
        rule_dir = f"data/processed/models/results/medium_rule"
        weighted_ops = rule_parser.parse_combine_rules(rule_dir, rule_file, top_quality_perc)
        is_probabilistic = True
    elif pred_type == PredType.RuleSynthHighCount:
        pred_type_path_change = "rule_synth_high"
        rule_file = f"{language}"
        rule_dir = f"data/processed/models/results/high_rule"
        weighted_ops = rule_parser.parse_combine_rules(rule_dir, rule_file, top_quality_perc)
        is_probabilistic = True
    else:
        raise NotImplementedError

    predictions = run_predictions(base_expected_and_morphs, weighted_ops, is_probabilistic,
                                  morph_feature_comparison_strictness)
    baseline.save_predictions(f"data/processed/predictions/{pred_type_path_change}/{language}.csv",
                              predictions)


# def calculate_mln_no_grammar_cost_language(language: str):
#     baseline.calculate_and_save_cost_baseline(
#         f"data/processed/predictions/mln_no_grammar/{language}.csv",
#         f"data/processed/predictions/mln_no_grammar_cost/{language}.csv"
#     )

def calculate_grammar_cost_for_language(language: str, pred_type: PredType):
    pred_files = {
        PredType.AdaGramLeft: "adagram/left",
        PredType.AdaGramBoth: "adagram/both",
        PredType.AdaGramRight: "adagram/right",
        PredType.NoOperation: "no_op",
        PredType.RuleSynthHighCount: "rule_synth_high",
        PredType.RuleSynthMediumCount: "rule_synth_medium",
        PredType.RuleSynthLowCount: "rule_synth_low",
        PredType.RuleSynthLowCountSample: "rule_synth_low_sample",
        PredType.RuleSynthLowCountSampleM: "rule_synth_low_sample_m"
    }
    pred_file = pred_files[pred_type]
    baseline.calculate_and_save_cost_baseline(
        f"data/processed/predictions/{pred_file}/{language}.csv",
        f"data/processed/predictions/{pred_file}_cost/{language}.csv"
    )


def write_context_morph_data() -> None:
    _iter_lang_dir("data/processed/first_step", _write_context_morph_data)


def write_frequency_tables():
    _iter_lang_dir(f"data/processed/context_morph_data", _write_frequency_tables)


def _iter_lang_dir(dir_path: str, op: Callable[[str], None]) -> None:
    for filename in os.listdir(dir_path):
        language, _ = filename.split('.')
        print(f"Started work on {language}")
        op(language)
        print(f"Finished work on {language}")


def _write_frequency_tables(language: str):
    data_path = f"data/processed/context_morph_data/{language}.csv"
    # freq.generate_context_op_table(
    #     data_path,
    #     f"data/processed/context_matrix/{language}.csv"
    # )
    # freq.generate_morph_op_table(
    #     data_path,
    #     f"data/processed/morph_matrix/{language}.csv"
    # )
    freq.generate_morph_context_op_table(
        data_path,
        f"data/processed/morph_context_matrix/{language}.csv"
    )


def _write_context_morph_data(language: str):
    freq.generate_basic_data_csv(
        f"data/processed/first_step/{language}.csv",
        f"data/processed/context_morph_data/{language}.csv"
    )


def run_adagram(language: str, split_type: SplitType, pred_type: PredType):
    pred_type_dict = {
        PredType.AdaGramBoth: "both",
        PredType.AdaGramRight: "right",
        PredType.AdaGramLeft: "left"
    }
    assert pred_type in pred_type_dict
    pred_type_name = pred_type_dict[pred_type]
    train_file_name = f"{language}.dat"
    split_type_str = str(split_type).replace("|", "_").replace(".", "_")
    input_file = f"data/processed/grammar/train_data/{split_type_str}/{train_file_name}"
    output_dir = f"data/processed/grammar/adagram/{pred_type_name}"
    grammar_file = f"data/processed/grammar/{pred_type_name}/{language}.unigram"
    number_of_entries = str(sum(1 for line in open(input_file)))
    args = [
        "--input_file", input_file,
        "--output_directory", output_dir,
        "--grammar_file", grammar_file,
        "--number_of_documents", number_of_entries,
        "--batch_size", "10"
    ]
    # adagram.main(args)
    output_grammar_path = f"data/processed/grammar/adagram/{pred_type_name}/{language}.grammar"
    _save_best_adagram_grammar(f"{output_dir}/{train_file_name}", output_grammar_path)


def run_rule_synthesis_search(args):
    language = args.language
    directory = f"/home/rafm/ProgramSynthNLP"
    model_output_dir = f"{directory}/data/processed/models"
    model_file = f"{language}{args.model_ending}"
    data_input_file = f"{directory}/data/processed/context_morph_data/{language}.csv"
    alphabet_file = f"{directory}/data/processed/alphabet/{language}.csv"
    grammar_file = f"{directory}/data/processed/grammar/adagram/both/{language}.csv"
    test_data_file = f"{directory}/data/processed/first_step/{language}.csv"
    result_file = f"{directory}/data/processed/models/results/{args.search_result_dir}/{language}"

    input_args = [
        "--dir_model", model_output_dir,
        "--fn_out_model", model_file,
        "--data_file_path", data_input_file,
        "--test_data_file_path", test_data_file,
        "--grammar_file_path", grammar_file,
        "--alphabet_file_path", alphabet_file,
        "--type", "NLP",
        "--new_test_ep", "NLP",
        "--timeout", "20",
        "--episode_type", "NLP",
        "--batchsize", "128",
        "--rule_count", str(args.rule_count),
        "--support_set_count", str(args.support_set_count),
        "--query_set_count", str(args.query_set_count),
        "--savefile", result_file,
        "--n_test", str(args.search_sample_count),
        "--n_runs", "1",
        "--max_decoder_output", str(args.max_decoder_output),
        "--max_searches", str(args.max_searches)
    ]
    if args.simple_search:
        input_args.append("--nosearch")

    scan_search.main(input_args)


def run_rule_synthesis(args):
    # --fn_out_model nlp.p --type NLP --batchsize 128 --episode_type NLP --num_pretrain_episodes 100000
    language = args.language
    directory = f"/home/rafm/ProgramSynthNLP"
    model_dir = f"{directory}/data/processed/models"
    model_output_file = f"{language}{args.model_ending}"
    data_input_file = f"{directory}/data/processed/context_morph_data/{language}.csv"
    alphabet_file = f"{directory}/data/processed/alphabet/{language}.csv"
    grammar_file = f"{directory}/data/processed/grammar/adagram/both/{language}.csv"
    test_data_file = f"{directory}/data/processed/first_step/{language}.csv"

    input_args = [
        "--dir_model", model_dir,
        "--fn_out_model", model_output_file,
        "--data_file_path", data_input_file,
        "--alphabet_file_path", alphabet_file,
        "--test_data_file_path", test_data_file,
        "--grammar_file_path", grammar_file,
        "--type", "NLP",
        "--episode_type", "NLP",
        "--num_pretrain_episodes", str(args.train_count),
        "--batchsize", "128",
        "--rule_count", str(args.rule_count),
        "--support_set_count", str(args.support_set_count),
        "--query_set_count", str(args.query_set_count),
        "--save_freq", "5"
    ]

    synthTrain.main(input_args)


def _save_best_adagram_grammar(input_dir: str, output_file_path: str):
    all_subdirs = [f"{input_dir}/{d}" for d in os.listdir(input_dir) if
                   os.path.isdir(f"{input_dir}/{d}")]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    all_results = [f for f in os.listdir(latest_subdir) if f.startswith("adagram")]
    best_result = max(all_results, key=lambda res: int(res.split("-")[1]))
    print(f"Best: {best_result}")
    os.replace(f"{latest_subdir}/{best_result}", output_file_path)


def generate_grammar_file_for_adagram(language: str, split_type: SplitType,
                                      train_type: gram_gen.TrainingType):
    out_names = {
        gram_gen.TrainingType.Left: "left",
        gram_gen.TrainingType.Right: "right",
        gram_gen.TrainingType.Both: "both"
    }

    output_dir = f"data/processed/grammar/{out_names[train_type]}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    gram_gen.generate_grammar_file(
        f"data/processed/alphabet/{language}.csv",
        f"data/processed/context_morph_data/{language}.csv",
        f"{output_dir}/{language}.unigram",
        split_type,
        train_type
    )


def generate_grammar_train_file_for_adagram(language: str, split_type: SplitType):
    split_type_str = str(split_type).replace("|", "_").replace(".", "_")
    output_dir = f"data/processed/grammar/train_data/{split_type_str}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    gram_train.generate_grammar_train_file(
        f"data/processed/context_morph_data/{language}.csv",
        f"{output_dir}/{language}.dat",
        split_type
    )


def get_all_languages_from_dir(directory: str) -> Collection[str]:
    return [file.split(".")[0] for file in os.listdir(directory)]


# write_steps(True, True, True, True, True)
# write_alphabets()
# write_first_second_step_revision()
# write_context_matrices()
# write_concepts()
# draw_trees()
# _test_memorizing_lattice()
# _test_decision_tree()
# predict_words()
# write_baseline_cost()
# predict_words()
# write_baseline_cost()
# reformat_sigmorphon_predictions()
# write_sigmorphon_baseline_cost()
# get_average_sigmorphon_cost()
# compare_base_to_sigmorphon()
# write_steps(True, True, True, True, True)
# write_first_second_step_revision()
# write_context_matrices()
# write_concepts()
# predict_words()
# write_baseline_cost()
# compare_base_to_sigmorphon()
# draw_trees()
# create_grammar_context_matrices()


# write_steps(True, True, True, True, True)
# write_first_second_step_revision()
# write_context_matrices()
# create_grammar_context_matrices()
# predict_language("asturian")
# _generate_no_op_baseline("asturian")
# calculate_mln_no_grammar_cost_language("asturian")
# compare_baselines("mln_no_grammar_cost", "no_op_cost")
# print("About to generate context morph data")
# input("Press ENTER to start")
# write_context_morph_data()
# print(f"About to generate frequency tables")
# input("Press ENTER to start")
# write_frequency_tables()
# _write_frequency_tables("latin")

# write_alphabets()

def make_adagrammar_for_languages():
    dir = "data/processed/first_step"
    languages = get_all_languages_from_dir(dir)
    calculated_dir = "data/processed/predictions/adagram/both_cost"
    calculated_languages = get_all_languages_from_dir(calculated_dir)
    languages = [lang for lang in languages if lang not in calculated_languages]
    split_type = SplitType.IncludeGrammar | SplitType.ContextLetters
    train_type = gram_gen.TrainingType.Both
    pred_type = PredType.AdaGramBoth
    strictness = Strictness.All
    for language in languages:
        try:
            print(f"Starting work on: {language}")
            generate_grammar_file_for_adagram(
                language,
                split_type,
                train_type
            )
            generate_grammar_train_file_for_adagram(language, split_type)
            run_adagram(language, split_type, PredType.AdaGramBoth)
            predict_language(language, pred_type, strictness)
            calculate_grammar_cost_for_language(language, pred_type)
            predict_language(language, PredType.NoOperation, strictness)
            calculate_grammar_cost_for_language(language, PredType.NoOperation)
            print(f"Finished work on {language}")
        except Exception as e:
            import traceback
            FAIL = '\033[91m'
            ENDC = '\033[0m'
            print(f"{FAIL}Language fail: {language}{ENDC}")
            print(f"{FAIL}{traceback.format_exc()}{ENDC}")
            print(e)
    calculate_average_prediction_costs(pred_type)


def pred_language_and_calculate_cost(language: str, pred_type: PredType, strictness: Strictness):
    predict_language(language, pred_type, strictness)
    calculate_grammar_cost_for_language(language, pred_type)
    calculate_average_prediction_costs(pred_type)


if __name__ == '__main__':
    pass
    # gram_path = "data/processed/grammar/adagram/both/asturian.grammar"
    # out_path = "data/processed/grammar/adagram/both/asturian.csv"
    # gram_extractor.save_grammar_file(gram_path, out_path)
    # make_adagrammar_for_languages()
    # calculate_average_prediction_costs(PredType.AdaGramBoth)
    # calculate_average_prediction_costs(PredType.NoOperation)
    # args = arg_parser.parse_args()
    # if args.search:
    #     run_rule_synthesis_search(args)
    # else:
    #     run_rule_synthesis(args)

    # for language in ["livonian", "asturian", "kurmanji"]:
    #     pred_language_and_calculate_cost(language, PredType.RuleSynthLowCount, Strictness.All)
    #     pred_language_and_calculate_cost(language, PredType.RuleSynthMediumCount, Strictness.All)
    # pred_language_and_calculate_cost("asturian", PredType.RuleSynthLowCountSample, Strictness.All)
    # pred_language_and_calculate_cost("asturian", PredType.RuleSynthLowCountSampleM, Strictness.All)
    # predict_rulesynth_results(0.5, Strictness.All)
    # for i in range(0, 11):
    #     predict_rulesynth_results(i / 10, Strictness.All)
    # calculate_costs()
    # calculate_average_costs_across_rule_counts_by_lang()
    # calculate_average_costs_across_rule_counts_combined()
    # reformat_sigmorphon_predictions()
    # write_sigmorphon_baseline_cost()
    # dir = "sigmorphon_cost"
    # l_m_s = _get_mean_and_standard_devs_for_languages()
    # sig_costs = _get_average_baseline_cost(dir)
    # out = "data/other_costs/sigmorphon.csv"
    # with open(out, mode="w+") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Language", "Sigmorphon"])
    #     for language in sig_costs:
    #         writer.writerow([language, sig_costs[language]])
