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
#### OpenAI large
- 2021, basic + text [data/emb/ICLR/2021/20250615_1154](data/emb/ICLR/2021/20250615_1154)
- 2022, basic + text [data/emb/ICLR/2022/20250605_1432](data/emb/ICLR/2022/20250605_1432)
  - [20250609_2115](result/ICLR/2022/20250609_2115)
- 2023, basic + text [data/emb/ICLR/2023/20250605_1153](data/emb/ICLR/2023/20250605_1153)
  - [20250609_2112](result/ICLR/2023/20250609_2112)
- 2024, basic + text [data/emb/ICLR/2024/20250602_1929](data/emb/ICLR/2024/20250602_1929)
  - [20250609_2109](result/ICLR/2024/20250609_2109)
- 2024, basic [data/emb/ICLR/2024/20250530_1457](data/emb/ICLR/2024/20250530_1457)
  - [20250609_2102](result/ICLR/2024/20250609_2102)
- (DO NOT USE) 2021, basic + text [data/emb/ICLR/2021/20250605_1513](data/emb/ICLR/2021/20250605_1513)
  - With outstanding paper sessions, different groups of agents
  - [20250609_2118](result/ICLR/2021/20250609_2118) 
#### Gemini text-embedding-004
- 2024, basic, gemini text-embedding-004 [20250611_0009](data/emb/ICLR/2024/20250611_0009)
  - [20250611_0012](result/ICLR/2024/20250611_0012)
- 2022, basic+text
#### gemini-embeddings-exp-03-07 
- 2024, basic+text [data/emb/ICLR/2024/20250614_1132](data/emb/ICLR/2024/20250614_1132)
- 2023, basic+text [data/emb/ICLR/2023/20250615_1159](data/emb/ICLR/2023/20250615_1159)
- 2022, basic+text [data/emb/ICLR/2022/20250615_1258](data/emb/ICLR/2022/20250615_1258)
- 2021, basic+text [data/emb/ICLR/2021/20250615_1304](data/emb/ICLR/2021/20250615_1304)
### NeurIPS
#### OpenAI large
- 2023, basic [data/emb/NeurIPS/2023/20250602_2201](data/emb/NeurIPS/2023/20250602_2201)
  - result [20250609_2106](result/NeurIPS/2023/20250609_2106)
- 2023, basic + text [data/emb/NeurIPS/2023/20250602_2203](data/emb/NeurIPS/2023/20250602_2203)
  - result [20250609_2105](result/NeurIPS/2023/20250609_2105)
- 2024, basic [data/emb/NeurIPS/2024/20250602_2031](data/emb/NeurIPS/2024/20250602_2031)
  - result [20250609_2100](result/NeurIPS/2024/20250609_2100)
- 2024, basic + text [data/emb/NeurIPS/2024/20250602_2037](data/emb/NeurIPS/2024/20250602_2037)
  - result [20250609_2102](result/NeurIPS/2024/20250609_2102)
#### gemini-embeddings-exp-03-07 
- 2023, basic+text [data/emb/NeurIPS/2023/20250615_1317](data/emb/NeurIPS/2023/20250615_1317)
- 2024, basic+text [data/emb/NeurIPS/2024/20250615_1325](data/emb/NeurIPS/2024/20250615_1325)
- 2023, basic [data/emb/NeurIPS/2023/20250615_1340](data/emb/NeurIPS/2023/20250615_1340)
- 2024, basic [data/emb/NeurIPS/2024/20250615_1336](data/emb/NeurIPS/2024/20250615_1336)


