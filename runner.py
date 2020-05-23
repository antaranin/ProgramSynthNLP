import os
from enum import Enum
from typing import Callable, List, Tuple, Dict, Collection

from functools import partial
from context_matrix import create_and_save_context_matrix, load_context_matrix, \
    rewrite_context_matrix_to_concept_lib_format, \
    create_grammar_context_matrix_for_concept_lib_format
from grammar_train_file_generator import SplitType
from main import process_data_file, write_alphabet, read_base_expected_words_and_morph_features
from precondition_interpreter import run_predictions, Prediction
from mln_file_generator import read_mln_file, Strictness
import operation_revisor as rev
import baseline
import frequency_table_gen as freq
import adagram_grammar_extractor as gram_extractor
import grammar_train_file_generator as gram_train
import grammar_file_generator as gram_gen
import csv
import rulesynthesis.synthTrain as synthTrain
import rulesynthesis.scan_search as scan_search
import rule_parser


class PredType(Enum):
    NoOperation = 0
    AdaGramLeft = 1
    AdaGramRight = 2
    AdaGramBoth = 3
    RuleSynth = 4



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
        PredType.RuleSynth: "rule_synth"
    }
    output_file_name_dict = {
        PredType.AdaGramBoth: "adagram_both",
        PredType.AdaGramRight: "adagram_right",
        PredType.AdaGramLeft: "adagram_left",
        PredType.NoOperation: "no_op",
        PredType.RuleSynth: "rule_synth"
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


def predict_language(
        language: str,
        pred_type: PredType,
        morph_feature_comparison_strictness: Strictness
):
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
    elif pred_type == PredType.RuleSynth:
        pred_type_path_change = "rule_synth"
        rule_file = f"data/processed/models/results/{language}.p"
        weighted_ops = rule_parser.parse_rules(rule_file)
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
        PredType.RuleSynth: "rule_synth"
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
    #adagram.main(args)
    output_grammar_path = f"data/processed/grammar/adagram/{pred_type_name}/{language}.grammar"
    _save_best_adagram_grammar(f"{output_dir}/{train_file_name}", output_grammar_path)


def run_rule_synthesis_search(language: str):
    model_output_dir = "data/processed/models"
    model_file = f"{language}_proper_2.p"
    data_input_file = f"data/processed/context_morph_data/{language}.csv"
    alphabet_file = f"data/processed/alphabet/{language}.csv"
    grammar_file = f"data/processed/grammar/adagram/both/{language}.csv"
    test_data_file = f"data/processed/first_step/{language}.csv"
    result_file = f"data/processed/models/results/{language}"

    args = [
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
        "--rule_count", "100",
        "--support_set_count", "100",
        "--query_set_count", "100",
        "--savefile", result_file,
        "--n_runs", "5",
    ]

    scan_search.main(args)


def run_rule_synthesis(language: str):
    # --fn_out_model nlp.p --type NLP --batchsize 128 --episode_type NLP --num_pretrain_episodes 100000
    directory = f"/home/rafm/ProgramSynthNLP"
    model_dir=f"{directory}/data/processed/models"
    model_output_file = f"{language}_low_rule.p"
    data_input_file = f"{directory}/data/processed/context_morph_data/{language}.csv"
    alphabet_file = f"{directory}/data/processed/alphabet/{language}.csv"
    grammar_file = f"{directory}/data/processed/grammar/adagram/both/{language}.csv"
    test_data_file = f"data/processed/first_step/{language}.csv"

    args = [
	"--dir_model", model_dir,
        "--fn_out_model", model_output_file,
        "--data_file_path", data_input_file,
        "--alphabet_file_path", alphabet_file,
        "--test_data_file_path", test_data_file,
        "--grammar_file_path", grammar_file,
        "--type", "NLP",
        "--episode_type", "NLP",
        "--num_pretrain_episodes", "20000",
        "--batchsize", "128",
        "--rule_count", "10",
        "--support_set_count", "40",
        "--query_set_count", "10",
	"--save_freq", "100"
    ]

    synthTrain.main(args)


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


if __name__ == '__main__':
    # gram_path = "data/processed/grammar/adagram/both/asturian.grammar"
    # out_path = "data/processed/grammar/adagram/both/asturian.csv"
    # gram_extractor.save_grammar_file(gram_path, out_path)
    # make_adagrammar_for_languages()
    # calculate_average_prediction_costs(PredType.AdaGramBoth)
    # calculate_average_prediction_costs(PredType.NoOperation)
    # run_rule_synthesis("asturian")
    run_rule_synthesis_search("asturian")
    # predict_language("asturian", PredType.RuleSynth, Strictness.All)
    # calculate_grammar_cost_for_language("asturian", PredType.RuleSynth)
    # calculate_average_prediction_costs(PredType.RuleSynth)
