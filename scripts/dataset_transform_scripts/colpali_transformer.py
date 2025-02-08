from datasets import load_dataset, load_dataset_builder, Image, DatasetDict, Value, Sequence


def loadDatasets():
    # available splits: ['train', 'test']
    ds = load_dataset("vidore/colpali_train_set")
    return ds

def transformPassages(entry, indices):
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

def transformDataset(ds):
    trans_ds = ds.map(transformPassages, remove_columns=["options", "page", "model", "prompt", "answer_type", "image", "image_filename"], batched=True, with_indices=True, num_proc=8)

    # rename attributes
    trans_ds = trans_ds.rename_column("query", "query_text")
    # update column attribute types
    trans_ds = trans_ds.cast_column("query_image", Image()).cast_column("positive_document_ids", Sequence(Value("string"))).cast_column("negative_document_ids", Sequence(Value("string")))

    # reorder columns
    return trans_ds.select_columns(['query_id', 'query_text', 'query_image', 'positive_document_ids', 'negative_document_ids', 'answer', 'source'])

def uploadDataset(new_dsDict):
    new_dsDict.push_to_hub("SamanthaZJQ/colpali-passage-2.0")

def main():
    dsDict = loadDatasets()
    print(dsDict)
    dsDict = {split: transformDataset(dsDict[split]) for split in dsDict}
    print(dsDict["train"][0])
    # perform dataset update
    uploadDataset(DatasetDict(dsDict))
    # verify feature
    print("-------------------")
    print(load_dataset_builder("SamanthaZJQ/colpali-passage-2.0").info.features)


if __name__=="__main__":
    main()