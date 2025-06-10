# Fair-Conference-Scheduling


## Process data (.json) to embeddings:
```
 python -m src.cli.generator --config ./src/configs/embed.yaml
```

## Clustering and evaluate:
```
 python -m src.cli.clusterer --config ./src/configs/clusterer.yaml
```

## Embedding Quick Access:
### ICLR
- 2021, basic + text [data/emb/ICLR/2021/20250605_1513](data/emb/ICLR/2021/20250605_1513)
  - [20250609_1513](result/ICLR/2021/20250609_1513)
- 2022, basic + text [data/emb/ICLR/2022/20250605_1432](data/emb/ICLR/2022/20250605_1432)
  - [20250609_1512](result/ICLR/2022/20250609_1512)
- 2023, basic + text [data/emb/ICLR/2023/20250605_1153](data/emb/ICLR/2023/20250605_1153)
  - [20250609_1500](result/ICLR/2023/20250609_1500)
- 2024, basic + text [data/emb/ICLR/2024/20250602_1929](data/emb/ICLR/2024/20250602_1929)
  - [20250609_1459](result/ICLR/2024/20250609_1459)
- 2024, basic [data/emb/ICLR/2024/20250530_1457](data/emb/ICLR/2024/20250530_1457)
  - [20250609_1517](result/ICLR/2024/20250609_1517)
### NeurIPS
- 2023, basic [data/emb/NeurIPS/2023/20250602_2201](data/emb/NeurIPS/2023/20250602_2201)
  - result [20250609_1518](result/NeurIPS/2023/20250609_1518)
- 2023, basic + text [data/emb/NeurIPS/2023/20250602_2203](data/emb/NeurIPS/2023/20250602_2203)
  - result [20250609_1456](result/NeurIPS/2023/20250609_1456)
- 2024, basic [data/emb/NeurIPS/2024/20250602_2031](data/emb/NeurIPS/2024/20250602_2031)
  - result [20250609_1527](result/NeurIPS/2024/20250609_1527)
- 2024, basic + text [data/emb/NeurIPS/2024/20250602_2037](data/emb/NeurIPS/2024/20250602_2037)
  - result [20250609_1454](result/NeurIPS/2024/20250609_1454)

