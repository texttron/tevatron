import os
import pandas as pd
import json
from collections import defaultdict
import argparse
from nirtools.ir import write_qrels
from tqdm import tqdm
from dataclasses import dataclass
from io import TextIOWrapper
import tarfile
import glob
import math

TRAIN_QUERY_FN = "multivent_2_train_queries.csv"
TRAIN_JUDGEMENT_FN = "multivent_2_train_judgments.jsonl"
TEST_QUERY_FN = "multivent_2_test_queries.csv"


def load_query(fn):
    df = pd.read_csv(fn, sep=",", header=0)
    qid2query = {}
    for _, row in df.iterrows():
        qid2query[row["Query_id"]] = row["query"]
    return qid2query


def load_judgement(fn):
    query2positive_doc = defaultdict(dict)
    with open(fn, "r") as f:
        for line in f:
            data = json.loads(line)
            query2positive_doc[data["query_id"]][data["doc_id"]] = data["relevance"]
    return query2positive_doc


####################
# Training Pairs
####################
def form_training_pairs(qid2query, query2positive_doc, output_fn):
    def _form_training_entry(qid, query, positive_doc_ids, negative_doc_ids):
        return {
            "positive_document_ids": positive_doc_ids,
            "negative_document_ids": negative_doc_ids,
            "query_id": qid,
            "query_image": None,
            "query_text": query,
            "source": "MultiVENT2.0",
        }
    
    output_dir = os.path.dirname(output_fn)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_fn, "w") as f:
        for qid, query in tqdm(qid2query.items(), desc="Forming training data"):
            positive_doc_ids = query2positive_doc[qid]
            negative_doc_ids = []
            entry = _form_training_entry(qid, query, positive_doc_ids, negative_doc_ids)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


####################
# Corpus
####################
@dataclass
class CorpusEntry:
    docid: str = None
    # image: str = None
    video: str = None
    audio: str = None
    audio_caption: str = None
    video_caption: str = None
    video_title: str = None
    video_description: str = None

    def __post_init__(self):
        def clean_text(text):
            if not text: return None
            if isinstance(text, float) and math.isnan(text): return None
            return text.replace("\n", " ").replace("\t", " ").strip()

        all_fields = [self.image, self.video, self.audio, self.audio_caption, self.video_caption, self.video_title, self.video_description]
        if all(x is None for x in all_fields):
            raise ValueError(f"All fields are None for document {self.docid}")

        self.video_caption = clean_text(self.video_caption)
        self.video_title = clean_text(self.video_title)
        self.video_description = clean_text(self.video_description)
        self.audio_caption = clean_text(self.audio_caption)

        text = [self.video_caption, self.video_title, self.video_description, self.audio_caption]
        text = [x for x in text if x]
        self.text = "\t".join(text)

    def to_dict(self):
        return {
            "docid": self.docid,
            "text": self.text,
            # "image": self.image,
            "video": self.video,
            "audio": self.audio,
            "audio_caption": self.audio_caption,
            "video_caption": self.video_caption,
            "video_title": self.video_title,
            "video_description": self.video_description,
        }


def get_file_obj_from_tar(tar_path, is_member_fn):
    # print(f"Processing archive: {tar_path}")
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            for member in tar.getmembers():
                if is_member_fn(member):
                    try:
                        file_obj = tar.extractfile(member)
                        if file_obj:
                            yield member.name, file_obj
                    except Exception as e:
                        print(f"[WARN] Skipping {member.name} in {os.path.basename(tar_path)}: {e}")
    except Exception as e:
        print(f"[ERROR] Could not open {tar_path}: {e}")
    return None


def load_video_text_from_tar(tar_path):
    def is_member_fn(member):
        return member.isfile() and member.name.endswith(".json")

    data = {}
    for file_name, file_obj in get_file_obj_from_tar(tar_path, is_member_fn):
        text_stream = TextIOWrapper(file_obj, encoding="utf-8")
        content = json.load(text_stream)

        caption = content.get("caption", "")
        yt_meta_dict = content.get("yt_meta_dict", {}).get("info", {})
        title = yt_meta_dict.get("title", "")
        description = yt_meta_dict.get("description", "")

        id = yt_meta_dict.get("id", "")
        assert id == os.path.splitext(os.path.basename(file_name))[0]

        data[id] = {
            "video_caption": caption,
            "video_title": title,
            "video_description": description,
        } 
    return data


def load_audio_text_from_tar(tar_path):
    def is_member_fn(member):
        return member.isfile() and member.name.endswith(".csv")

    data = {}
    for file_name, file_obj in get_file_obj_from_tar(tar_path, is_member_fn):
        text_stream = TextIOWrapper(file_obj, encoding="utf-8")
        df = pd.read_csv(text_stream, sep=",", header=0)
        assert df.columns[0] == "video_id"
        assert df.columns[1] == "text"
        for _, row in df.iterrows():
            id = row["video_id"].split("/")[-1].replace(".m4a", "") # remove the tar id (e.g., 000001)
            data[id] = {"audio_caption": row["text"]}
    return data


def form_corpus(multivent_path, audio_path_pattern, output_dir):
    """
    Args:
        multivent_path: path to the multivent data (a list of .tar files)
        audio_path: path to the audio data
    """
    os.makedirs(output_dir, exist_ok=True)

    for tar_path in tqdm(sorted(glob.glob(os.path.join(multivent_path, "*.tar"))), desc="Forming corpus"):
        tar_id = os.path.splitext(os.path.basename(tar_path))[0]

        output_fn = os.path.join(output_dir, f"{tar_id}.jsonl")
        with open(output_fn, "w") as f:
            audio_path = audio_path_pattern.format(tar_id=tar_id)

            id2video_text = load_video_text_from_tar(tar_path)
            id2audio_text = load_audio_text_from_tar(audio_path)
            for id, video_text in id2video_text.items():
                audio_text = id2audio_text.get(id, {})
                if not audio_text:
                    print(f"[WARN] No audio text found for {id}")

                corpus_entry = CorpusEntry(
                    docid=id,
                    video=f"{id}.mp4",
                    audio=f"{id}.m4a",
                    **video_text,
                    **audio_text,
                )
                f.write(json.dumps(corpus_entry.to_dict(), ensure_ascii=False) + "\n")



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multivent_path", "-path", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, default="data/MultiVent")
    return parser.parse_args()



def main():
    args = get_args()
    multivent_path = args.multivent_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(multivent_path, "train")
    test_path = os.path.join(multivent_path, "test")
    whisper_path = os.path.join(multivent_path, "features/train/whisper_asr/exp/scale24/data/video/baseline/train/speech_to_text")
    whisper_path_pattern = whisper_path + "/train-{tar_id}-whisperv3-large.tar.gz"

    ####### write judgement file #######
    train_qid2query = load_query(os.path.join(multivent_path, TRAIN_QUERY_FN))
    train_query2positive_doc = load_judgement(os.path.join(multivent_path, TRAIN_JUDGEMENT_FN))
    write_qrels(train_query2positive_doc, os.path.join(output_dir, "train.qrels"))

    ####### write training pairs #######
    form_training_pairs(train_qid2query, train_query2positive_doc, os.path.join(output_dir, "training_pairs.jsonl"))

    ####### write corpus #######
    form_corpus(train_path, whisper_path_pattern, os.path.join(output_dir, "corpus"))



def test_loading():
    path = "data/MultiVent/corpus/000001.jsonl"
    import datasets
    ds = datasets.load_dataset("json", data_files=path, split="train")
    import pdb; pdb.set_trace()



if __name__ == "__main__":
    # main()
    test_loading()