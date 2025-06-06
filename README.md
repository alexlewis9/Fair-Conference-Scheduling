# Fair-Conference-Scheduling


## Process data (.json) to embeddings:
```
 python -m src.cli.generator --config ./src/configs/embed.yaml
```

## Clustering and evaluate:
```
 python -m src.cli.clusterer --config ./src/configs/clusterer.yaml
```