from datasets import load_dataset, load_dataset_builder, Image, DatasetDict, Value, Sequence

def loadDatasets():
    # available splits: ['train', 'dev', 'dl19', 'dl20']
    dsDict = load_dataset("Tevatron/msmarco-passage-aug")
    return dsDict


def transformPassages(entry):
    return {
        # transform positive/negative document to id
        "positive_document_ids": [[passage['docid'] for passage in passages] for passages in entry['positive_passages']],
        "negative_document_ids": [[passage['docid'] for passage in passages] for passages in entry['negative_passages']],
        # add query_image, source attributes, and answer
        "query_image": [None] * len(entry["query"]),
        "source": ["msmarco"] * len(entry["query"]),
        "answer": [None] * len(entry["query"]),
    }

def transformDataset(ds):
    # convert document_ids to store a list of string docid
    trans_ds = ds.map(transformPassages, remove_columns=["positive_passages", "negative_passages"], batched=True, num_proc=8)
    
    # rename attributes
    trans_ds = trans_ds.rename_column("query", "query_text")
    # update old column attribute types
    trans_ds = trans_ds.cast_column("answer", Value("string")).cast_column("query_image", Image()).cast_column("positive_document_ids", Sequence(Value("string"))).cast_column("negative_document_ids", Sequence(Value("string")))

    # reorder columns
    return trans_ds.select_columns(['query_id', 'query_text', 'query_image', 'positive_document_ids', 'negative_document_ids', 'answer', 'source'])

def uploadDataset(new_dsDict):
    new_dsDict.push_to_hub("SamanthaZJQ/msmarco-passage-aug-2.0")

def main():
    dsDict = loadDatasets()
    print(dsDict)
    dsDict = {split: transformDataset(dsDict[split]) for split in dsDict}
    # # perform dataset update
    uploadDataset(DatasetDict(dsDict))
    # # verify feature
    print("-------------------")
    print(load_dataset_builder("SamanthaZJQ/msmarco-passage-aug-2.0").info.features)


if __name__=="__main__":
    main()