from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

with open(args.input) as f_in, open(args.output, 'w') as f_out:
    cur_qid = None
    rank = 0
    for line in f_in:
        qid, docid, score = line.split()
        if cur_qid != qid:
            cur_qid = qid
            rank = 0
        rank += 1
        f_out.write(f'{qid}\t{docid}\t{rank}\n')
