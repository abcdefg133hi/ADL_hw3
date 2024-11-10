import json
import argparse

def change(fin, fout):
    data = None
    with open(fin, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        data[i]["output"] = ""
    with open(args.output_file, 'w', encoding='utf-8', errors='ignore') as file:
        json.dump(data, file, ensure_ascii=False)





parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--input_file",
    type=str,
    default=None,
    help="The path of the input.",
)
parser.add_argument(
    "--output_file",
    type=str,
    default=None,
    help="The path of the output.",
)
args = parser.parse_args()

change(args.input_file, args.output_file)
