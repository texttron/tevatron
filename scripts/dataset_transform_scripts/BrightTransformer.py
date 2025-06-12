from datasets import load_dataset, load_dataset_builder, Image, DatasetDict, Value, Sequence


def load_datasets(task):
    # available splits: ['train']
    ds =  load_dataset("xlangai/BRIGHT", "examples")[task]
    return ds

def transform_dataset(ds, task):
    new_image_column = [None] * len(ds)
    new_source_column = [task] * len(ds)
    positive_document_ids = [[]] * len(ds)
    negative_document_ids = [[]] * len(ds)
    trans_ds = ds.add_column("query_image", new_image_column).add_column("source", new_source_column).add_column("positive_document_ids", positive_document_ids).add_column("negative_document_ids", negative_document_ids)
    trans_ds = trans_ds.rename_column('id', 'query_id')
    trans_ds = trans_ds.rename_column('query', 'query_text')
    trans_ds = trans_ds.rename_column('gold_answer', 'answer')
    trans_ds = trans_ds.remove_columns(['reasoning', 'excluded_ids', 'gold_ids_long', 'gold_ids'])
    return trans_ds.cast_column("query_image", Image()).cast_column("positive_document_ids", Sequence(Value("string"))).cast_column("negative_document_ids", Sequence(Value("string")))

def upload_dataset(new_ds_dict, task):
    new_ds_dict.push_to_hub("ArvinZhuang/bright", config_name=task, split="test")

def main():
    for task in ['biology', 'earth_science', 'economics', 'psychology', 'robotics', 'stackoverflow',
                 'sustainable_living', 'pony', 'leetcode', 'aops', 'theoremqa_theorems', 'theoremqa_questions']:
        ds = load_datasets(task)
        print(task, ds)
        ds = transform_dataset(ds, task)
        upload_dataset(ds, task)

if __name__=="__main__":
    main()
