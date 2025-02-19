from datasets import load_dataset, load_dataset_builder, Image, DatasetDict, Value, Sequence


def load_datasets():
    # available splits: ['train', 'test']
    ds = load_dataset("vidore/colpali_train_set")
    return ds

def transform_passages(entry, indices):
    num_row = len(entry["query"])
    return {
        # transform positive/negative document to id
        "positive_document_ids": [[str(i)] for i in indices],
        "negative_document_ids": [[] for i in range(num_row)],
        # add attributes
        "query_id": [str(i) for i in indices],
        "query_image": [None] * num_row,
        "source": ["colpali: "+str(source) for source in entry['source']],
    }

def transform_dataset(ds):
    trans_ds = ds.map(transform_passages, remove_columns=["options", "page", "model", "prompt", "answer_type", "image", "image_filename"], batched=True, with_indices=True, num_proc=8)

    # rename attributes
    trans_ds = trans_ds.rename_column("query", "query_text")
    # update column attribute types
    trans_ds = trans_ds.cast_column("query_image", Image()).cast_column("positive_document_ids", Sequence(Value("string"))).cast_column("negative_document_ids", Sequence(Value("string")))

    # reorder columns
    return trans_ds.select_columns(['query_id', 'query_text', 'query_image', 'positive_document_ids', 'negative_document_ids', 'answer', 'source'])

def upload_dataset(new_ds_dict):
    new_ds_dict.push_to_hub("SamanthaZJQ/colpali-passage-2.0")

def main():
    ds_dict = load_datasets()
    print(ds_dict)
    ds_dict = {split: transform_dataset(ds_dict[split]) for split in ds_dict}
    print(ds_dict["train"][0])
    # perform dataset update
    upload_dataset(DatasetDict(ds_dict))
    # verify feature
    print("-------------------")
    print(load_dataset_builder("SamanthaZJQ/colpali-passage-2.0").info.features)


if __name__=="__main__":
    main()