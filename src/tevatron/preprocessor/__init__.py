from .preprocessor_tsv import SimpleTrainPreProcessor as MarcoPassageTrainPreProcessor,  \
    SimpleCollectionPreProcessor as MarcoPassageCollectionPreProcessor

from .preprocessor_dict import TrainPreProcessor as HFTrainPreProcessor, TestPreProcessor as HFTestPreProcessor, \
    CorpusPreProcessor as HFCorpusPreProcessor
