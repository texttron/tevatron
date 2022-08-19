# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.Wikipedia

# Lint as: python3
"""MsMarco Passage Reranking dataset."""

import json
import datasets

_CITATION = """
@misc{bajaj2018ms,
      title={MS MARCO: A Human Generated MAchine Reading COmprehension Dataset}, 
      author={Payal Bajaj and Daniel Campos and Nick Craswell and Li Deng and Jianfeng Gao and Xiaodong Liu 
      and Rangan Majumder and Andrew McNamara and Bhaskar Mitra and Tri Nguyen and Mir Rosenberg and Xia Song
       and Alina Stoica and Saurabh Tiwary and Tong Wang},
      year={2018},
      eprint={1611.09268},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = "dataset load script for MSMARCO Passage Reranking"

_DATASET_URLS = {
    # 'train': "https://huggingface.co/datasets/Tevatron/msmarco-passage/resolve/main/train.jsonl.gz",
}


class MsMarcoPassageRerank(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(version=VERSION,
                               description="MS MARCO passage rerank datasets"),
    ]

    def _info(self):
        features = datasets.Features({
            'query_id': datasets.Value('string'),
            'query': datasets.Value('string'),
            'docid': datasets.Value('string'), 
            'title': datasets.Value('string'), 
            'text': datasets.Value('string'),
            'score': datasets.Value('float32'),
        })
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="",
            # License for the dataset if available
            license="",
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.data_files:
            downloaded_files = self.config.data_files
        else:
            downloaded_files = dl_manager.download_and_extract(_DATASET_URLS)
        splits = [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "files": [downloaded_files[split]] if isinstance(downloaded_files[split], str) else downloaded_files[split],
                },
            ) for split in downloaded_files
        ]
        return splits
        
    def _generate_examples(self, files):
        """Yields examples."""
        for filepath in files:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    yield data['query_id']+data['docid'], data
