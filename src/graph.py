import networkx as nx


def build_graph(df, min_weight=1):
    """
    Строит неориентированный граф взаимодействий персонажей.

    Parameters
    ----------
    df : pd.DataFrame
        Таблица взаимодействий персонажей.
        Ожидаются столбцы:
        - Source
        - Target
        - weight

    Returns
    -------
    networkx.Graph
        Граф персонажей с весами рёбер.
    """
    G = nx.Graph()

    for _, row in df.iterrows():
        weight = row.get("weight", 1)

        if weight >= min_weight:
            G.add_edge(
                row["Source"],
                row["Target"],
                weight=weight
            )

    return G

def build_graphs_by_book(df):
    graphs = {}

    for book in sorted(df["book"].unique()):
        df_book = df[df["book"] == book]
        graphs[book] = build_graph(df_book)

    return graphs

def basic_stats(G):
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G)
    }


def build_cumulative_graph(df, max_book, min_weight=1):
    df_cutoff = df[df["book"] <= max_book]
    return build_graph(df_cutoff, min_weight=min_weight)