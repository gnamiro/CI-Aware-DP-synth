# results_io.py
import os
import pandas as pd
from pathlib import Path
from Constraints import DEFAULT_COLS, EPS


def _epsilon_key(x):
    """Canonical string for epsilon used for matching/keys and CSV storage."""
    if x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and x.strip() == ""):
        return ""
    # preserve decimals when needed, drop trailing .0 for whole numbers
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        return str(int(x)) if x.is_integer() else repr(x)
    # strings: keep as-is (assume already formatted)
    return str(x)

def _format_eps_column(df):
    if "epsilon" in df.columns:
        df["epsilon"] = df["epsilon"].apply(_epsilon_key)
    return df


def ensure_results_table(csv_path, methods=None, cv=None, epsilons=None, cols=None):
    import pandas as pd
    from pathlib import Path
    cols = cols or ["method_name","fold_num","epsilon","CMI","CHI","W-dist","KL","TVD","AUC"]

    p = Path(csv_path); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        df = pd.read_csv(p)
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
    else:
        df = pd.DataFrame(columns=cols)

    # normalize existing epsilon
    if "epsilon" in df.columns:
        df["epsilon"] = df["epsilon"].apply(_epsilon_key)

    if methods is not None and cv is not None and epsilons is not None:
        norm_eps = [_epsilon_key(e) for e in epsilons]
        existing = set(zip(
            df.get("method_name", []).astype(str),
            df.get("fold_num", []).astype("Int64").fillna(-1).astype(int),
            df.get("epsilon", "").astype(str),
        ))

        new_rows = []
        for m in methods:
            for f in range(int(cv)):
                # Only Original has blank epsilon; others iterate over provided epsilons
                iter_eps = [""] if str(m) == "Original" else norm_eps
                for eps in iter_eps:
                    key = (str(m), int(f), eps)
                    if key not in existing:
                        new_rows.append({"method_name": str(m), "fold_num": int(f), "epsilon": eps})
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # order & save
    key_cols = ["method_name","fold_num","epsilon"]
    other_cols = [c for c in df.columns if c not in key_cols]
    df = df[key_cols + other_cols]
    df.to_csv(p, index=False)
    return df

def set_metric(csv_path, method_name, fold_num, metric_name, value, epsilon=None):
    """Set one metric for (method_name, fold_num, epsilon). Creates row/col if missing."""
    df = ensure_results_table(csv_path)

    # Ensure metric column exists
    if metric_name not in df.columns:
        df[metric_name] = pd.NA

    eps_key = _epsilon_key(epsilon)
    mask = (
        (df["method_name"].astype(str) == str(method_name)) &
        (df["fold_num"].astype(int) == int(fold_num)) &
        (df["epsilon"].astype(str) == eps_key)
    )

    if not mask.any():
        # create the row
        new_row = {"method_name": str(method_name), "fold_num": int(fold_num), "epsilon": eps_key}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        mask = (
            (df["method_name"].astype(str) == str(method_name)) &
            (df["fold_num"].astype(int) == int(fold_num)) &
            (df["epsilon"].astype(str) == eps_key)
        )

    df.loc[mask, metric_name] = value

    # Keep keys first and format epsilon nicely
    key_cols = ["method_name", "fold_num", "epsilon"]
    other_cols = [c for c in df.columns if c not in key_cols]
    df = df[key_cols + other_cols]
    df = _format_eps_column(df)
    df.to_csv(csv_path, index=False)
