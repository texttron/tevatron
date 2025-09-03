from typing import Dict, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from tevatron.retriever.modeling import MultiModalDenseModel
from transformers import AutoProcessor

from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorQA
from vidore_benchmark.retrievers import BaseVisionRetriever, VisionRetriever
from qwen_vl_utils import process_vision_info
from vidore_benchmark.utils.iter_utils import batched
import math
from vidore_benchmark.utils.data_utils import get_datasets_from_collection
from argparse import ArgumentParser


class TevatronVisionRetriever(BaseVisionRetriever):
    """
    Dummy retriever that generates random dense embeddings.
    """

    def __init__(
        self,
        model: MultiModalDenseModel,
        processor,
        query_prefix: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(use_visual_embedding=True)
        self.model = model
        self.processor = processor
        self.query_prefix = query_prefix

    def forward_queries(
        self,
        queries,
        batch_size: int,
        **kwargs,
    ) -> torch.Tensor:
        qs = []
        queries = [f"{self.query_prefix}{q}" for q in queries]
        for batch_query in tqdm(
            batched(queries, batch_size),
            desc="Forwarding query batches",
            total=math.ceil(len(queries) / batch_size),
            leave=False,
        ):
            messages = []
            for idx in range(len(batch_query)):
                text = batch_query[idx]
                content = []
                if text:
                    text = self.processor.tokenizer.decode(
                        self.processor.tokenizer.encode(
                            text, max_length=512, truncation=True
                        )
                    )
                    content.append({"type": "text", "text": text})
                message = [{"role": "user", "content": content}]
                messages.append(message)

            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False
                )
                for msg in messages
            ]
            texts = [x + "<|endoftext|>" for x in texts]

            image_inputs, video_inputs = process_vision_info(messages)

            collated_inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding="longest",
            )
            with torch.no_grad():
                query_embeddings = self.model.encode_query(collated_inputs.to("cuda"))
            qs.extend(list(torch.unbind(query_embeddings.to("cpu"))))
        return qs

    def forward_passages(
        self,
        passages,
        batch_size: int,
        **kwargs,
    ) -> torch.Tensor:

        ps = []

        for batch_passage in tqdm(
            batched(passages, batch_size),
            desc="Forwarding passage batches",
            total=math.ceil(len(passages) / batch_size),
            leave=False,
        ):
            messages = []
            for idx in range(len(batch_passage)):
                image = batch_passage[idx]
                content = []
                if image:
                    content.append(
                        {
                            "type": "image",
                            "image": image,
                            "resized_height": 784,
                            "resized_width": 784,
                        }
                    )
                    # content.append({'type': 'text', 'text': 'What is shown in this image?'})
                message = [{"role": "user", "content": content}]
                messages.append(message)

            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False
                )
                for msg in messages
            ]

            texts = [x + "<|endoftext|>" for x in texts]

            image_inputs, video_inputs = process_vision_info(messages)

            collated_inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding="longest",
            )
            with torch.no_grad():
                passage_embeddings = self.model.encode_passage(
                    collated_inputs.to("cuda")
                )
            ps.extend(list(torch.unbind(passage_embeddings.to("cpu"))))
        return ps

    def get_scores(
        self,
        query_embeddings,
        passage_embeddings,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Dot-product similarity between queries and passages.
        """
        if isinstance(query_embeddings, list):
            query_embeddings = torch.stack(query_embeddings)
        if isinstance(passage_embeddings, list):
            passage_embeddings = torch.stack(passage_embeddings)

        scores = torch.einsum("bd,cd->bc", query_embeddings, passage_embeddings)

        return scores


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lora_name_or_path", default=None)
    parser.add_argument("--pooling", default="last")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--query_prefix", default="")
    args = parser.parse_args()

    # Load model and processor
    model = (
        MultiModalDenseModel.load(
            args.model_name_or_path,
            pooling=args.pooling,
            normalize=args.normalize,
            lora_name_or_path=args.lora_name_or_path,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
        )
        .eval()
        .to("cuda")
    )

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = "left"

    # Get retriever instance
    vision_retriever = TevatronVisionRetriever(
        model=model,
        processor=processor,
        query_prefix=args.query_prefix,
    )

    vidore_evaluator = ViDoReEvaluatorQA(vision_retriever)

    # dataset_names = [
    #     'vidore/arxivqa_test_subsampled',
    #     'vidore/docvqa_test_subsampled',
    #     'vidore/infovqa_test_subsampled',
    #     'vidore/tabfquad_test_subsampled',
    #     'vidore/tatdqa_test',
    #     'vidore/shiftproject_test',
    #     'vidore/syntheticDocQA_artificial_intelligence_test',
    #     'vidore/syntheticDocQA_energy_test',
    #     'vidore/syntheticDocQA_government_reports_test',
    #     'vidore/syntheticDocQA_healthcare_industry_test'
    # ]

    collection_name = (
        "vidore/vidore-benchmark-667173f98e70a1c0fa4db00d"  # ViDoRe Benchmark
    )
    dataset_names = get_datasets_from_collection(collection_name)

    res = []
    for dataset_name in dataset_names:
        print("Evaluating", dataset_name)
        ds = load_dataset(dataset_name, split="test")
        metrics_dataset = vidore_evaluator.evaluate_dataset(
            ds=ds,
            batch_query=args.batch_size,
            batch_passage=args.batch_size,
            batch_score=args.batch_size,
        )
        print(dataset_name, f"ndcg@5: {metrics_dataset['ndcg_at_5']}")
        res.append((dataset_name, f"ndcg@5: {metrics_dataset['ndcg_at_5']}"))

    print(res)
    # average
    print(sum([float(x[1].split(": ")[1]) for x in res]) / len(res))


if __name__ == "__main__":
    main()
