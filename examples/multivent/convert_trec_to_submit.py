import json
from argparse import ArgumentParser


def trec_to_dict(trec_file_path):
    result = {}
    with open(trec_file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 6:
                continue  # skip malformed lines
            query_id, _, doc_id, _, score, _ = parts
            if query_id not in result:
                result[query_id] = {}
            result[query_id][doc_id] = float(score)
    return result


parser = ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()


trec_path = args.input
output_path = args.output

converted = trec_to_dict(trec_path)

with open(output_path, "w") as f:
    json.dump(converted, f, indent=2)

print("Converted format saved to:", output_path)
