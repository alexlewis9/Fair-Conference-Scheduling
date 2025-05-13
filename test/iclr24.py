import os

import pandas as pd
from src import DATA_DIR


def test_iclr24():
    metadata_file = os.path.join(DATA_DIR, 'metadata', 'ICLR_2021_2024.csv')
    # Load the CSV file
    df = pd.read_csv('your_file.csv')

    # Filter rows where the 'year' column is 2024
    filtered_ids = df[df['year'] == 2024]['id']

    # Convert to list (optional)
    id_list = filtered_ids.tolist()

