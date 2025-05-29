import os


from src import DATA_DIR, generate_embeddings, Node, Graph, GreedyCohesiveClustering


def test_iclr24_abstract_basic():
    input_path = os.path.join(DATA_DIR, 'unified_text', 'ICLR', f'ICLR_2024.json')
    output_path = os.path.join(DATA_DIR, 'emb', 'ICLR')
    model = 'text-embedding-3-large'
    inclusion = ['abstract', 'title', 'authors']

    embeddings = generate_embeddings(input_path, output_path, model, include=inclusion)

    nodes = []
    for key, value in embeddings.items():
        nodes.append(Node(key, value))

    graph = Graph(nodes, 32)
    print(GreedyCohesiveClustering(graph, 32))

#result:
"""
[['P15CHILQlg', 'NSVtmmzeRB', 'Ouj6p4ca60'], ['ANvmVS2Yr0', 'WNkW0cOwiz', 'nHESwXvxWK'], ['LzPWWPAdY4', '6PmJoRfdaK', 'w4abltTZ2f'], ['VtmBAGCN7o', '9JQtrumvg8', 'H3UayAQWoE'], ['sFyTZEqmUY', 'jNR6s6OSBT', 'o2IEmeLL9r'], ['HhfcNgQn6p', 'HE9eUQlAvo', 'IYxDy2jDFL'], ['Ad87VjRqUw', 'UyNXMqnN3c', 'sllU8vvsFF'], ['Zsfiqpft6K', 'pzElnMrgSD', 'EanCFCwAjM'], ['aN4Jf6Cx69', 'ekeyCgeRfC', 'PdaPky8MUn'], ['hnrB5YHoYu', '84n3UwkH7b', 'jr03SfWsBS'], ['bNt7oajl2a', 'KUNzEQMWU7', 'VTF8yNQM66'], ['1vDArHJ68h', 'LjivA1SLZ6', 'agPpmEgf8C'], ['oO6FsMyDBt', 'HSKaGOi7Ar', 'IGzaH538fz'], ['hSyW5go0v8', 'WbWtOYIzIK', '1oijHJBRsT'], ['uNrFpDPMyo', 'aIok3ZD9to', 'FVhmnvqnsI'], ['hTEGyKf0dZ', 'KS8mIvetg2', 'd8w0pmvXbZ'], ['bTMMNT7IdW', 'tUtGjQEDd4', 'TpD2aG1h0D'], ['Yen1lGns2o', '9Cu8MRmhq2', 'c5pwL0Soay'], ['7VPTUWkiDQ', 'oTRwljRgiv', '5Ca9sSzuDp'], ['BV1PHbTJzd', 'zMPHKOmQNb', 'L0r0GphlIL'], ['gU58d5QeGv', 'WNzy9bRDvG', 'tqh1zdXIra'], ['4Ay23yeuz0', 'h922Qhkmx1', 'ze7DOLi394'], ['T7YV5UZKBc', 'AhizIPytk4', '2dnO3LLiJ1'], ['C61sk5LsK6', 'Fk5IzauJ7F', 'osoWxY8q2E'], ['dLrhRIMVmB', 'v7ZPwoHU1j', 'xuY33XhEGR'], ['mE52zURNGc', 'cc8h3I3V4E', 'g7ohDlTITL'], ['7Ttk3RzDeu', '3f5PALef5B', '9WD9KwssyT'], ['gFR4QwK53h', 'pOoKI3ouv1', '0BqyZSWfzo'], ['jKTUlxo5zy', 'yV6fD7LYkF'], [], [], []]
"""

if __name__ == '__main__':
    test_iclr24_abstract_basic()





