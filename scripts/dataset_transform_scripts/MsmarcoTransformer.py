from datasets import load_dataset, load_dataset_builder, Image, DatasetDict, Value, Sequence

def load_datasets():
    # available splits: ['train', 'dev', 'dl19', 'dl20']
    ds_dict = load_dataset("Tevatron/msmarco-passage-aug")
    return ds_dict


def transform_passages(entry):
    return {
        # transform positive/negative document to id
        "positive_document_ids": [[passage['docid'] for passage in passages] for passages in entry['positive_passages']],
        "negative_document_ids": [[passage['docid'] for passage in passages] for passages in entry['negative_passages']],
        # add query_image, source attributes, and answer
        "query_image": [None] * len(entry["query"]),
        "source": ["msmarco"] * len(entry["query"]),
        "answer": [None] * len(entry["query"]),
    }

def transform_dataset(ds):
    # convert document_ids to store a list of string docid
    trans_ds = ds.map(transform_passages, remove_columns=["positive_passages", "negative_passages"], batched=True, num_proc=8)
    
    # rename attributes
    trans_ds = trans_ds.rename_column("query", "query_text")
    # update old column attribute types
    trans_ds = trans_ds.cast_column("answer", Value("string")).cast_column("query_image", Image()).cast_column("positive_document_ids", Sequence(Value("string"))).cast_column("negative_document_ids", Sequence(Value("string")))

    # reorder columns
    return trans_ds.select_columns(['query_id', 'query_text', 'query_image', 'positive_document_ids', 'negative_document_ids', 'answer', 'source'])

def upload_dataset(new_ds_dict):
    new_ds_dict.push_to_hub("SamanthaZJQ/msmarco-passage-aug-2.0")

def main():
    ds_dict = load_datasets()
    print(ds_dict)
    ds_dict = {split: transform_dataset(ds_dict[split]) for split in ds_dict}
    # # perform dataset update
    upload_dataset(DatasetDict(ds_dict))
    # # verify feature
    print("-------------------")
    print(load_dataset_builder("SamanthaZJQ/msmarco-passage-aug-2.0").info.features)


if __name__=="__main__":
    main()