import argparse
from .processor import SimpleTrainProcessor, SimpleDevProcessor, \
    SimpleTestProcessor, SimpleCorpusProcessor, MsMarcoDevProcessor

_PROCESSORS = {
    'Tevatron/wikipedia-nq': {
        'train': SimpleTrainProcessor,
        'dev': SimpleDevProcessor,
        'test': SimpleTestProcessor,
        'corpus': SimpleCorpusProcessor,
    },
    'Tevatron/wikipedia-trivia': {
        'train': SimpleTrainProcessor,
        'dev': SimpleDevProcessor,
        'test': SimpleTestProcessor,
        'corpus': SimpleCorpusProcessor,
    },
    'Tevatron/msmarco-passage': {
        'train': SimpleTrainProcessor,
        'dev': MsMarcoDevProcessor,
        'corpus': SimpleCorpusProcessor,
    }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, required=False, default='bert-base-uncased')
    parser.add_argument('--threads', type=int, required=False, default=12)
    parser.add_argument('--with-title', action='store_true')
    args = parser.parse_args()

    processor = _PROCESSORS[args.dataset][args.split]()

    print("Loading dataset")
    processor.load(args.dataset)

    print("Processing dataset")
    processor.process(args.tokenizer, args.with_title, threads=args.threads)

    print("Saving dataset")
    processor.save(args.output)


if __name__ == '__main__':
    main()
