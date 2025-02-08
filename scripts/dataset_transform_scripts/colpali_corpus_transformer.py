from datasets import load_dataset, load_dataset_builder, Image, DatasetDict, Value, Sequence


def loadDatasets():
    # available splits: ['train', 'test']
    ds = load_dataset("vidore/colpali_train_set")
    return ds

def transformPassages(entry, indices):
    num_row = len(entry["query"])
    return {
        "docid": [str(i) for i in indices],
        "text": [None] * num_row,
        "source": ["colpali:"+str(source) for source in entry['image_filename']],
    }

def transformDataset(ds):
    trans_ds = ds.map(transformPassages, batched=True, with_indices=True, num_proc=8)

    # update column attribute types
    trans_ds = trans_ds.cast_column("text", Value("string"))

    # reorder columns
    return trans_ds.select_columns(['docid', 'image', 'text', 'source'])

def uploadDataset(new_dsDict):
    new_dsDict.push_to_hub("SamanthaZJQ/colpali-passage-corpus-2.0")

def main():
    dsDict = loadDatasets()
    print(dsDict)
    dsDict = {split: transformDataset(dsDict[split]) for split in dsDict}
    print(dsDict["train"][0])
    # perform dataset update
    uploadDataset(DatasetDict(dsDict))
    # verify feature
    print("-------------------")
    print(load_dataset_builder("SamanthaZJQ/colpali-passage-corpus-2.0").info.features)


if __name__=="__main__":
    main()