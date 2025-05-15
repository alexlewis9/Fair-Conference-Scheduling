# Fair-Conference-Scheduling


## Process csv to PaperNodes: 
```
python -m src.data_processing.csv_to_paper_node ./data/test/papers.csv --emb-column  emb_v2
```

## Process data (.json) to embeddings:
```
 python -m src.data_processing.generate_embeddings ./data/test/test.json ./data/emb/ --model text-embedding-3-small
```