import argparse
from src.models.paper_node import PaperNode
import os
print("Current working directory:", os.getcwd())

def main():
    parser = argparse.ArgumentParser(description="Load paper nodes from CSV.")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument("--emb-column", default="emb", help="Name of the embedding column (default: emb)")
    args = parser.parse_args()

    try:
        paper_nodes = PaperNode.load_from_csv(args.csv_path, emb_column=args.emb_column)
        print(f"✅ Loaded {len(paper_nodes)} paper nodes.")
        for node in paper_nodes[:3]:  # preview
            print(node)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()