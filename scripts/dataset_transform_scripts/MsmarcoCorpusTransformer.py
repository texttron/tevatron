from datasets import load_dataset, load_dataset_builder, Image, DatasetDict

def load_datasets():
    # available splits: ['train']
    ds_dict = load_dataset("Tevatron/msmarco-passage-corpus")
    return ds_dict

def transform_dataset(ds):
    new_image_column = [None] * len(ds)
    new_source_column = ["msmarco"] * len(ds)
    trans_ds = ds.add_column("image", new_image_column).add_column("source", new_source_column)
    return trans_ds.cast_column("image", Image())

def upload_dataset(new_ds_dict):
    new_ds_dict.push_to_hub("SamanthaZJQ/msmarco-passage-corpus-2.0")

def main():
    ds_dict = load_datasets()
    print(ds_dict)
    ds_dict = {split: transform_dataset(ds_dict[split]) for split in ds_dict}
    print(ds_dict["train"][0])
    # perform dataset update
    upload_dataset(DatasetDict(ds_dict))
    # verify feature
    print(load_dataset_builder("SamanthaZJQ/msmarco-passage-corpus-2.0").info.features)


if __name__=="__main__":
    main()
