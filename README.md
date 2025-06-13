# Fair-Conference-Scheduling


## Process data (.json) to embeddings:
```
 python -m src.cli.generator --config ./src/configs/embed.yaml
```

## Clustering and evaluate:
```
 python -m src.cli.clusterer --config ./src/configs/clusterer.yaml
```

## Embedding + Result Quick Access:
### ICLR
- 2021, basic + text [data/emb/ICLR/2021/20250605_1513](data/emb/ICLR/2021/20250605_1513)
  - [20250609_2118](result/ICLR/2021/20250609_2118)
- 2022, basic + text [data/emb/ICLR/2022/20250605_1432](data/emb/ICLR/2022/20250605_1432)
  - [20250609_2115](result/ICLR/2022/20250609_2115)
- 2023, basic + text [data/emb/ICLR/2023/20250605_1153](data/emb/ICLR/2023/20250605_1153)
  - [20250609_2112](result/ICLR/2023/20250609_2112)
- 2024, basic + text [data/emb/ICLR/2024/20250602_1929](data/emb/ICLR/2024/20250602_1929)
  - [20250609_2109](result/ICLR/2024/20250609_2109)
- 2024, basic [data/emb/ICLR/2024/20250530_1457](data/emb/ICLR/2024/20250530_1457)
  - [20250609_2102](result/ICLR/2024/20250609_2102)
- 2024, basic, gemini text-embedding-004 [20250611_0009](data/emb/ICLR/2024/20250611_0009)
  - [20250611_1647](result/ICLR/2024/20250611_1647)

### NeurIPS
- 2023, basic [data/emb/NeurIPS/2023/20250602_2201](data/emb/NeurIPS/2023/20250602_2201)
  - result [20250609_2106](result/NeurIPS/2023/20250609_2106)
- 2023, basic + text [data/emb/NeurIPS/2023/20250602_2203](data/emb/NeurIPS/2023/20250602_2203)
  - result [20250609_2105](result/NeurIPS/2023/20250609_2105)
- 2024, basic [data/emb/NeurIPS/2024/20250602_2031](data/emb/NeurIPS/2024/20250602_2031)
  - result [20250609_2100](result/NeurIPS/2024/20250609_2100)
- 2024, basic + text [data/emb/NeurIPS/2024/20250602_2037](data/emb/NeurIPS/2024/20250602_2037)
  - result [20250609_2102](result/NeurIPS/2024/20250609_2102)

