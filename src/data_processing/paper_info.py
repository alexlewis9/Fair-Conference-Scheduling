import json

# TODO: fine-grained control
def get_paper_info(paper_id, file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    for paper in data:
        if paper['id'] == paper_id:
            return paper
    return None
    