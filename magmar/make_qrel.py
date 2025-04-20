from datasets import load_dataset


dataset = load_dataset("Tevatron/multivent-sample-train-test")['test']

with open("qrels_multivent-sample-train-test.txt", "w") as f:
    for i in range(len(dataset)):
        qid = dataset[i]['query_id']
        docid = dataset[i]['positive_document_ids'][0]
        f.write(f"{qid} 0 {docid} 1\n")

