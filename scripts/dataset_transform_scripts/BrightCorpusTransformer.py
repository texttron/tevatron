from datasets import load_dataset, load_dataset_builder, Image


def load_datasets(task):
    # available splits: ['train']
    ds =  load_dataset("xlangai/BRIGHT", "documents")[task]
    return ds

def transform_dataset(ds, task):
    new_image_column = [None] * len(ds)
    new_source_column = [task] * len(ds)
    trans_ds = ds.add_column("image", new_image_column).add_column("source", new_source_column)
    trans_ds = trans_ds.rename_column('id', 'docid')
    trans_ds = trans_ds.rename_column('content', 'text')
    return trans_ds.cast_column("image", Image())

def upload_dataset(new_ds_dict, task):
    new_ds_dict.push_to_hub("ArvinZhuang/bright-corpus", config_name=task, split="train")

def main():
    for task in ['biology', 'earth_science', 'economics', 'psychology', 'robotics', 'stackoverflow',
                 'sustainable_living', 'pony', 'leetcode', 'aops', 'theoremqa_theorems', 'theoremqa_questions']:
        ds = load_datasets(task)
        print(task, ds)
        ds = transform_dataset(ds, task)
        upload_dataset(ds, task)

if __name__=="__main__":
    main()
