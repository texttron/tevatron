from datasets import load_dataset, load_dataset_builder, Image, DatasetDict, Value, Sequence


def load_datasets():
    # available splits: ['train', 'test']
    ds = load_dataset("vidore/colpali_train_set")
    return ds

def transform_passages(entry, indices):
    num_row = len(entry["query"])
    return {
        "docid": [str(i) for i in indices],
        "text": [None] * num_row,
        "source": ["colpali:"+str(source) for source in entry['image_filename']],
    }

def transform_dataset(ds):
    trans_ds = ds.map(transform_passages, batched=True, with_indices=True, num_proc=8)

    # update column attribute types
    trans_ds = trans_ds.cast_column("text", Value("string"))

    # reorder columns
    return trans_ds.select_columns(['docid', 'image', 'text', 'source'])

def upload_dataset(new_ds_dict):
    new_ds_dict.push_to_hub("SamanthaZJQ/colpali-passage-corpus-2.0")

def main():
    ds_dict = load_datasets()
    print(ds_dict)
    ds_dict = {split: transform_dataset(ds_dict[split]) for split in ds_dict}
    print(ds_dict["train"][0])
    # perform dataset update
    upload_dataset(DatasetDict(ds_dict))
    # verify feature
    print("-------------------")
    print(load_dataset_builder("SamanthaZJQ/colpali-passage-corpus-2.0").info.features)


if __name__=="__main__":
    main()