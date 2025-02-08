from datasets import load_dataset, load_dataset_builder, Image, DatasetDict

def loadDatasets():
    # available splits: ['train']
    dsDict = load_dataset("Tevatron/msmarco-passage-corpus")
    return dsDict

def transformDataset(ds):
    new_image_column = [None] * len(ds)
    new_source_column = ["msmarco"] * len(ds)
    trans_ds = ds.add_column("image", new_image_column).add_column("source", new_source_column)
    return trans_ds.cast_column("image", Image())

def uploadDataset(new_dsDict):
    new_dsDict.push_to_hub("SamanthaZJQ/msmarco-passage-corpus-2.0")

def main():
    dsDict = loadDatasets()
    print(dsDict)
    dsDict = {split: transformDataset(dsDict[split]) for split in dsDict}
    print(dsDict["train"][0])
    # perform dataset update
    uploadDataset(DatasetDict(dsDict))
    # verify feature
    print(load_dataset_builder("SamanthaZJQ/msmarco-passage-corpus-2.0").info.features)


if __name__=="__main__":
    main()
