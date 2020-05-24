import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action='store_true')
    parser.add_argument('--train_count', type=int, default=100000,
                        help='number of episodes for training')
    parser.add_argument('--model_ending', type=str, default='',
                        help='the ending to be appended to the output model name, used for reading/writing model')
    parser.add_argument('--search_result_dir', type=str, default='',
                        help='the directory for the results (not the full path, just the name of this particular)')
    parser.add_argument('--language', type=str, default='asturian',
                        help='the language to train/search on')
    parser.add_argument('--rule_count', type=int, default=100)
    parser.add_argument('--support_set_count', type=int, default=200)
    parser.add_argument('--query_set_count', type=int, default=100)
    parser.add_argument('--search_sample_count', type=int, default=20)
    parser.add_argument('--max_decoder_output', type=int, default=200)
    parser.add_argument('--max_searches', type=int, default=50)

    return parser.parse_args(args)
