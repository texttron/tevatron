import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import ProcessorMixin
from arguments import DataArguments


logger = logging.getLogger(__name__)


@dataclass
class TrainCollator:
    data_args: DataArguments
    processor: ProcessorMixin


    def __call__(self, features: List[Tuple[str, List[str]]]):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """
        all_queries = [f[0] for f in features]
        all_passages = []
        for f in features:
            all_passages.extend(f[1])
        query_prompts = []
        for query in all_queries:
            prompt = f"query: {query}</s>"
            query_prompts.append(prompt)

        passages_prompts = []
        for idx in range(len(all_passages)):
            prompt = f"<|image_{idx+1}|>\nWhat is shown in this image?</s>"
            passages_prompts.append(prompt)
        query_inputs = self.processor(query_prompts, images=None, return_tensors="pt", padding="longest", max_length=self.data_args.query_max_len, truncation=True)
        passage_inputs = self.processor(passages_prompts, images=all_passages, return_tensors="pt", padding="longest", max_length=self.data_args.passage_max_len, truncation=True)
        # remove the first dimension of size 1
        passage_inputs['input_ids'] = passage_inputs['input_ids'].squeeze(0)
        passage_inputs['attention_mask'] = passage_inputs['attention_mask'].squeeze(0)
        passage_inputs['image_sizes'] = passage_inputs['image_sizes'].squeeze(0)
        return query_inputs, passage_inputs
    


@dataclass
class EncodeCollator:
    data_args: DataArguments
    processor: ProcessorMixin

    def __call__(self, features: List[Tuple[str, str]]):
        """
        Collate function for encoding.
        :param features: list of (id, text) tuples
        """
        text_ids = [x[0] for x in features]
        passages = [x[1] for x in features]
        max_length = self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len
        if self.data_args.encode_is_query:
            prompts = [f"query: {text}</s>" for text in passages]
            images = None
            inputs = self.processor(prompts, images=images, return_tensors="pt", padding="longest", max_length=max_length, truncation=True)
        else:
            prompts = []
            for idx in range(len(passages)):
                prompt = f"<|image_{idx+1}|>\nWhat is shown in this image?</s>"
                prompts.append(prompt)
            inputs = self.processor(prompts, images=passages, return_tensors="pt", padding="longest", max_length=max_length, truncation=True)
            inputs['input_ids'] = inputs['input_ids'].squeeze(0)
            inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
            inputs['image_sizes'] = inputs['image_sizes'].squeeze(0)
        
        return text_ids, inputs
