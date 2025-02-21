import argparse
from collections import defaultdict
from inspect_ai.log import read_eval_log

def process_log_file(log_file):
    log = read_eval_log(log_file)
    print(f"Read log from {log_file}")
    scores = defaultdict(list)
    if log.samples is not None:
        for sample in log.samples:
            if sample.scores is not None:
                for key, score in sample.scores.items():
                    scores[key].append(score.value)
    return scores

def compute_average_scores(scores, log_file):
    for key, value in scores.items():
        if len(value) > 0 and value[0] in {'C', 'I'}:
            value = [1 if v == 'C' else 0 for v in value]
        value_sum = sum(value)
        value_count = len(value)
        print(f"Average score for {key} from {log_file}: {value_sum} / {value_count} = {value_sum/value_count}")

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-files", type=str, nargs='+', required=True)
    args = parser.parse_args()
    log_files = args.log_files
    
    for log_file in log_files:
        scores = process_log_file(log_file)
        compute_average_scores(scores, log_file)
