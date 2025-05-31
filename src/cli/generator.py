import argparse
import json
import os
from datetime import datetime

from src.data_processing.generate_embeddings import generate_embeddings
from src.utils.io import load_yaml, save_json, save_yaml


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for text files.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    try:
        emb, raw_emb, log_lines = generate_embeddings(cfg['input_path'], cfg['output_path'], cfg['model'],
                                           include=cfg['include'],
                                           exclude=cfg['exclude'],
                                           stride=cfg['stride'],
                                           max_tokens=cfg['max_tokens'])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = os.path.join(cfg['output_path'], timestamp)
        os.makedirs(output_dir, exist_ok=True)

        output_raw_emb = os.path.join(output_dir, "raw_emb.json")
        save_json(raw_emb, output_raw_emb)
        log_lines.append("[OK] exported to raw_emb.json")

        output_emb = os.path.join(output_dir, "emb.json")
        save_json(emb, output_emb)
        log_lines.append("[OK] exported to emb.json")

        output_cfg = os.path.join(output_dir, "config.yaml")
        save_yaml(cfg, output_cfg)

        output_log = os.path.join(output_dir, "log.txt")
        with open(output_log, "w", encoding='utf-8') as f:
            for line in log_lines:
                f.write(line + "\n")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()