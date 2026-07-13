from pathlib import Path
import pandas as pd


def load_all_edges() -> pd.DataFrame:
    """
    Загружает все файлы (по книгам) и объединяет их в один DataFrame
    """

    data_path = get_project_root() / "data"

    files = sorted(data_path.glob("book*.csv"))

    if not files:
        raise ValueError("No CSV files found in data folder")

    dfs = []

    for i, file in enumerate(files, start=1):
        df = pd.read_csv(file)

        df = validate_edges(df)

        # добавляем номер книги
        df["book"] = i

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    return df_all

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def validate_edges(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"Source", "Target"}

    if not required_columns.issubset(df.columns):
        raise ValueError("Missing required columns: Source, Target")

    df = df.dropna(subset=["Source", "Target"])

    if "weight" not in df.columns:
        df["weight"] = 1

    return df


def load_edges(filename: str = "book1.csv") -> pd.DataFrame:
    data_path = get_project_root() / "data" / filename

    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    df = pd.read_csv(data_path)
    df = validate_edges(df)


    return df

def load_node_labels(filename: str = "nodes.csv"):
    import pandas as pd
    from pathlib import Path

    data_path = get_project_root() / "data" / filename
    df = pd.read_csv(data_path)

    return df