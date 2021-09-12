from .processor import TrainProcessor, TestProcessor, CorpusProcessor

PROCESSOR_INFO = {
    'Tevatron/wikipedia-nq': {
        'train': TrainProcessor,
        'dev': TrainProcessor,
        'test': TestProcessor,
        'corpus': CorpusProcessor,
    },
    'Tevatron/wikipedia-trivia': {
        'train': TrainProcessor,
        'dev': TrainProcessor,
        'test': TestProcessor,
        'corpus': CorpusProcessor,
    },
    'Tevatron/msmarco-passage': {
        'train': TrainProcessor,
        'dev': TestProcessor,
        'corpus': CorpusProcessor,
    },
    'Tevatron/scifact': {
        'train': TrainProcessor,
        'dev': TestProcessor,
        'test': TestProcessor,
        'corpus': CorpusProcessor,
    },
}
