# Evaluate model on BEIR dataset

BEIR is a heterogeneous benchmark for Zero-shot evaluation of information retrieval models.

Dense retrieval models trained with Tevatron can run BEIR evaluation by using/modifying the bash script we provided.

```
bash scripts/eval_beir.sh <model ckpt> <beir-dataset> 
```

The following `<beir-dataset>` are supported in Tevatron:

- arguana
- climate-fever
- cqadupstack-android
- cqadupstack-english
- cqadupstack-gaming
- cqadupstack-gis
- cqadupstack-wordpress
- cqadupstack-physics
- cqadupstack-programmers
- cqadupstack-stats
- cqadupstack-tex
- cqadupstack-unix
- cqadupstack-webmasters
- cqadupstack-wordpress
- dbpedia-entity
- fever
- fiqa
- hotpotqa
- nfcorpus
- quora
- scidocs
- trec-covid
- webis-touche2020
- nq

please see [Tevatron/beir](https://huggingface.co/datasets/Tevatron/beir) and [Tevatron/beir-corpus](https://huggingface.co/datasets/Tevatron/beir-corpus) for details.




 