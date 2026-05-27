import pandas as pd
import networkx as nx


def extract_features(G, include_clustering: bool = True) -> pd.DataFrame:
    """
    Извлекает базовые graph-features для каждого узла.

    Parameters
    ----------
    G : networkx.Graph
        Граф персонажей.
    include_clustering : bool, default=True
        Добавлять ли clustering coefficient.

    Returns
    -------
    pd.DataFrame
        Таблица признаков по узлам.
    """
    nodes = list(G.nodes())
    features = pd.DataFrame(index=nodes)

    # 1. Обычная центральность по числу связей
    features["degree_centrality"] = pd.Series(nx.degree_centrality(G))

    # 2. Взвешенная степень (сумма весов рёбер)
    features["weighted_degree"] = pd.Series(dict(G.degree(weight="weight")))

    # 3. Посредничество
    # Пока считаем без веса: для co-occurrence графа вес = сила связи,
    # а не "длина пути", поэтому напрямую weight сюда подставлять не стоит.
    features["betweenness"] = pd.Series(
        nx.betweenness_centrality(G, weight=None)
    )

    # 4. PageRank с учётом веса
    features["pagerank"] = pd.Series(
        nx.pagerank(G, weight="weight")
    )

    # 5. Локальная плотность окружения
    if include_clustering:
        features["clustering"] = pd.Series(
            nx.clustering(G, weight="weight")
        )

    features = features.fillna(0).reset_index()
    features = features.rename(columns={"index": "node"})

    return features

def extract_temporal_features(graphs_by_book):
    """
    graphs_by_book: dict {book: Graph}
    """

    all_data = []

    for book, G in graphs_by_book.items():
        features = extract_features(G)

        for node in G.nodes():
            all_data.append({
                "node": node,
                "book": book,
                  
                "betweenness": features["betweenness"].get(node, 0),
                "pagerank": features["pagerank"].get(node, 0),
            })

    df = pd.DataFrame(all_data)

    return df

def aggregate_temporal_features(df_temporal):
    agg = df_temporal.groupby("node").agg({
        "degree": ["mean", "max"],
        "betweenness": ["mean", "max"],
        "pagerank": ["mean", "max"]
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
        })

    return pd.DataFrame(trends)