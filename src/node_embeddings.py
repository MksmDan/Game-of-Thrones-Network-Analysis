from node2vec import Node2Vec
import pandas as pd


def generate_node2vec_embeddings(
    graph,
    dimensions=32,
    walk_length=10,
    num_walks=20,
    workers=8,
):
    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        weight_key="weight",
        seed=42,
    )

    model = node2vec.fit(
        window=5,
        min_count=1,
    )

    rows = []

    for node in graph.nodes():

        vector = model.wv[str(node)]

        row = {"node": node}

        for i, value in enumerate(vector):
            row[f"emb_{i}"] = value

        rows.append(row)

    return pd.DataFrame(rows)