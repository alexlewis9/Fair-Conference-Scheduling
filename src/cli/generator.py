import argparse
import json
import os
from datetime import datetime

from src.data_processing.generate_embeddings import generate_embeddings
from src.utils.io import load_yaml, save_json, save_yaml
from src.utils.logger import setup_session_logger


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for text files.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = os.path.join(cfg['output_path'], timestamp)
        os.makedirs(output_dir, exist_ok=True)
        output_log = os.path.join(output_dir, "generator.log")
        logger, handler = setup_session_logger(output_log)
        logger.info(f"Generating embeddings for {cfg['input_path']}")

        emb, raw_emb = generate_embeddings(cfg['input_path'], cfg['model'], cfg['provider'],
                                           include=cfg['include'],
                                           exclude=cfg['exclude'],
                                           stride=cfg['stride'],
                                           max_tokens=cfg['max_tokens'],
                                           verbose=cfg['verbose'])

        output_raw_emb = os.path.join(output_dir, "raw_emb.json")
        save_json(raw_emb, output_raw_emb)
        logger.info(f"[OK] exported to raw_emb.json")

        output_emb = os.path.join(output_dir, "emb.json")
        save_json(emb, output_emb)
        logger.info(f"[OK] exported to emb.json")

        output_cfg = os.path.join(output_dir, "config.yaml")
        save_yaml(cfg, output_cfg)

        handler.close()
        logger.removeHandler(handler)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()