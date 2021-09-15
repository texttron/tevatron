from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
from tqdm import tqdm
from tevatron.preprocessor import MarcoPassageCollectionPreProcessor as CollectionPreProcessor

parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--truncate', type=int, default=32)
parser.add_argument('--query_file', required=True)
parser.add_argument('--save_to', required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
processor = CollectionPreProcessor(tokenizer=tokenizer, max_length=args.truncate)

with open(args.query_file, 'r') as f:
    lines = f.readlines()

os.makedirs(os.path.split(args.save_to)[0], exist_ok=True)
with open(args.save_to, 'w') as jfile:
    for x in tqdm(lines):
        q = processor.process_line(x)
        jfile.write(q + '\n')
