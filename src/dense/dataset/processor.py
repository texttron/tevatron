class Processor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class TrainProcessor(Processor):

    def __call__(self, example):
        query = self.tokenizer.encode(example['query'], add_special_tokens=False)
        positives = []
        for pos in example['positive_passages']:
            text = pos['title'] + " " + pos['text'] if 'title' in pos else pos['text']
            positives.append(self.tokenizer.encode(text, add_special_tokens=False))
        negatives = []
        for neg in example['negative_passages']:
            text = neg['title'] + " " + neg['text'] if 'title' in neg else neg['text']
            negatives.append(self.tokenizer.encode(text, add_special_tokens=False))
        return {'query': query, 'positives': positives, 'negatives': negatives}


class TestProcessor(Processor):
    def __call__(self, example):
        query_id = example['query_id']
        query = self.tokenizer.encode(example['query'], add_special_tokens=False)
        return {'text_id': query_id, 'text': query}


class CorpusProcessor(Processor):
    def __call__(self, example):
        docid = example['docid']
        text = example['title'] + " " + example['text'] if 'title' in example else example['text']
        text = self.tokenizer.encode(text, add_special_tokens=False)
        return {'text_id': docid, 'text': text}
