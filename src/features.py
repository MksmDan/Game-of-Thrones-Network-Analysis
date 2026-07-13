import pandas as pd
import networkx as nx


def extract_features(G, include_clustering: bool = True) -> pd.DataFrame:
    """
    Извлекает графовые признаки для каждого персонажа.

    Используемые признаки:
    - degree_centrality
    - weighted_degree
    - betweenness_centrality
    - pagerank
    - clustering_coefficient

    Parameters
    ----------
    G : networkx.Graph
        Граф персонажей.

    include_clustering : bool, default=True
        Добавлять ли коэффициент кластеризации.

    Returns
    -------
    pd.DataFrame
        Таблица признаков по персонажам.
    """
    nodes = list(G.nodes())
    features = pd.DataFrame(index=nodes)

    features["degree_centrality"] = pd.Series(nx.degree_centrality(G))

    features["weighted_degree"] = pd.Series(dict(G.degree(weight="weight")))

    features["betweenness"] = pd.Series(
        nx.betweenness_centrality(G, weight=None)
    )

    features["pagerank"] = pd.Series(
        nx.pagerank(G, weight="weight")
    )


    if include_clustering:
        features["clustering"] = pd.Series(
            nx.clustering(G, weight="weight")
        )

    features = features.fillna(0).reset_index()
    features = features.rename(columns={"index": "node"})

    return features

def extract_temporal_features(graphs_by_book):
    """
    Извлекает временные признаки персонажей
    отдельно для каждой книги.

    Для каждой книги сохраняются:
    - weighted_degree
    - pagerank
    - betweenness

    Parameters
    ----------
    graphs_by_book : dict[int, nx.Graph]
        Словарь графов по книгам.

    Returns
    -------
    pd.DataFrame
        Длинная таблица вида:

        node | book | weighted_degree |
        pagerank | betweenness
    """

    all_data = []

    for book, G in graphs_by_book.items():
        features = extract_features(G)
        features = features.set_index("node")

        for node in G.nodes():
            all_data.append({
                "node": node,
                "book": book,

                "weighted_degree":
                    features.loc[node, "weighted_degree"],

                "betweenness":
                    features.loc[node, "betweenness"],

                "pagerank":
                    features.loc[node, "pagerank"],
            })

    return pd.DataFrame(all_data)

def aggregate_temporal_features(df_temporal):
    agg = df_temporal.groupby("node").agg({
        "degree": ["mean", "max"],
        "betweenness": ["mean", "max"],
        "pagerank": ["mean", "max"],
        "weighted_degree" : ["mean", "max"]
    })

    # flatten columns
    agg.columns = ["_".join(col) for col in agg.columns]
    agg = agg.reset_index()

    return agg

def add_trend_features(df_temporal):
    df_sorted = df_temporal.sort_values(["node", "book"])

    trends = []

    for node, group in df_sorted.groupby("node"):
        group = group.sort_values("book")

        trends.append({
            "node": node,
            "degree_change": group["degree"].iloc[-1] - group["degree"].iloc[0],
            "pagerank_change": group["pagerank"].iloc[-1] - group["pagerank"].iloc[0],
            "weighted_degree" : group["weighted_degree"].iloc[-1] - group["weighted_degree"].iloc[0]
        })

    return pd.DataFrame(trends)

