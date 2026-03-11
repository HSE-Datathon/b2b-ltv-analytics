from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "b2b-ltv-service" / "app" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_DATE = pd.Timestamp("2023-06-30")

FEATURE_COLS = [
    "sector", "segment", "industry", "segment_2",
    "company_size", "revenue",
    "total_revenue", "recency", "num_orders", "avg_check",
    "customer_lifetime_days", "product_diversity", "unique_products",
    "customer_age", "revenue_per_day", "revenue_per_order",
    "avg_product_duration", "cluster",
]
CAT_FEATURES = ["sector", "segment", "industry", "segment_2", "cluster"]
CLUSTER_FEATURES = [
    "total_revenue", "num_orders", "avg_check",
    "recency", "unique_products", "revenue_per_day",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Загрузка данных...")
    df_companies = pd.read_excel(DATA_DIR / "companies_v2.xlsx")
    df_revenue = pd.read_excel(DATA_DIR / "revenue_v2.xlsx")

    df_companies.columns = df_companies.columns.str.strip().str.lower()
    df_revenue.columns = df_revenue.columns.str.strip().str.lower()

    if "company_size" in df_companies.columns:
        df_companies["company_size"] = pd.to_numeric(
            df_companies["company_size"], errors="coerce"
        )

    if "segment_2" in df_companies.columns:
        df_companies["segment_2"] = df_companies["segment_2"].astype(str)

    if "order_date" in df_revenue.columns:
        df_revenue["order_date"] = pd.to_datetime(df_revenue["order_date"], errors="coerce")

    print(f"  Компании: {len(df_companies)}, Транзакции: {len(df_revenue)}")
    return df_companies, df_revenue


def build_features(
    df_companies: pd.DataFrame, df_revenue: pd.DataFrame
) -> pd.DataFrame:
    print("Построение признаков...")

    df_hist = df_revenue[df_revenue["order_date"] <= CUTOFF_DATE].copy()
    df_future = df_revenue[df_revenue["order_date"] > CUTOFF_DATE].copy()

    future_rev = (
        df_future.groupby("company_id")["product_sum"]
        .sum()
        .reset_index()
        .rename(columns={"product_sum": "future_revenue"})
    )

    agg = df_hist.groupby("company_id").agg(
        total_revenue=("product_sum", "sum"),
        num_orders=("product_sum", "count"),
        avg_check=("product_sum", "mean"),
        avg_product_duration=("product_duration", "mean"),
        unique_products=("product_name", "nunique"),
        first_purchase=("order_date", "min"),
        last_purchase=("order_date", "max"),
    ).reset_index()

    agg["recency"] = (CUTOFF_DATE - agg["last_purchase"]).dt.days
    agg["customer_age"] = (agg["last_purchase"] - agg["first_purchase"]).dt.days + 1
    agg["customer_lifetime_days"] = (CUTOFF_DATE - agg["first_purchase"]).dt.days + 1
    agg["product_diversity"] = agg["unique_products"] / agg["num_orders"]
    agg["revenue_per_day"] = agg["total_revenue"] / agg["customer_lifetime_days"]
    agg["revenue_per_order"] = agg["total_revenue"] / agg["num_orders"]

    df = (
        df_companies
        .merge(agg, on="company_id", how="inner")
        .merge(future_rev, on="company_id", how="left")
    )
    df["future_revenue"] = df["future_revenue"].fillna(0)
    print(f"  Итоговый датасет: {len(df)} компаний")
    return df


def train_clustering(
    df: pd.DataFrame, n_clusters: int = 3
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    print("Обучение KMeans кластеризации...")

    X = np.log1p(df[CLUSTER_FEATURES].fillna(0).clip(lower=0))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(X_scaled).astype(str)

    for c in range(n_clusters):
        cnt = (df["cluster"] == str(c)).sum()
        print(f"  Кластер {c}: {cnt} компаний")

    return df, kmeans, scaler


def train_ltv_model(
    df: pd.DataFrame,
) -> tuple[CatBoostRegressor, dict, tuple[float, float]]:
    print("Обучение CatBoost LTV модели...")

    df_model = df[df["future_revenue"] > 0].copy()

    for col in CAT_FEATURES:
        if col in df_model.columns:
            df_model[col] = df_model[col].astype(str).fillna("unknown")

    X = df_model[FEATURE_COLS].copy()
    y = np.log1p(df_model["future_revenue"])

    num_cols = [c for c in FEATURE_COLS if c not in CAT_FEATURES]
    median_features: dict = {}
    for col in num_cols:
        col_numeric = pd.to_numeric(X[col], errors="coerce")
        median_features[col] = float(col_numeric.median() or 0.0)

    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(median_features[col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = CatBoostRegressor(
        iterations=3000,
        depth=6,
        learning_rate=0.01,
        l2_leaf_reg=5,
        loss_function="RMSE",
        eval_metric="MAE",
        early_stopping_rounds=200,
        cat_features=CAT_FEATURES,
        verbose=200,
        random_seed=42,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    from sklearn.metrics import mean_absolute_error, r2_score
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  Test MAE (log): {mae:.4f}  |  R2: {r2:.4f}")

    X_all = df_model[FEATURE_COLS].copy()
    for col in CAT_FEATURES:
        X_all[col] = X_all[col].astype(str).fillna("unknown")
    all_preds = np.expm1(model.predict(X_all))
    q33 = float(np.percentile(all_preds, 33))
    q66 = float(np.percentile(all_preds, 66))
    print(f"  Pороги сегментов: low<={q33:.0f} | mid<={q66:.0f} | high>{q66:.0f}")

    return model, median_features, (q33, q66)


def compute_nbo_scores(
    df: pd.DataFrame, df_revenue: pd.DataFrame
) -> dict[str, list[dict]]:
    print("Вычисление NBO-скоров...")

    df_hist = df_revenue[df_revenue["order_date"] <= CUTOFF_DATE].copy()
    df_rev_cluster = df_hist.merge(
        df[["company_id", "cluster"]], on="company_id", how="inner"
    )

    stats = (
        df_rev_cluster.groupby(["cluster", "product_name"])
        .agg(
            purchases=("product_sum", "count"),
            revenue=("product_sum", "sum"),
            avg_revenue=("product_sum", "mean"),
        )
        .reset_index()
    )

    cluster_totals = stats.groupby("cluster")["purchases"].sum().rename("total_purchases")
    stats = stats.join(cluster_totals, on="cluster")
    stats["popularity_score"] = stats["purchases"] / stats["total_purchases"]

    cluster_max = stats.groupby("cluster")["avg_revenue"].max().rename("max_avg_revenue")
    stats = stats.join(cluster_max, on="cluster")
    stats["value_score"] = stats["avg_revenue"] / stats["max_avg_revenue"]

    stats["nbo_score"] = (
        0.35 * stats["popularity_score"]
        + 0.25 * stats["value_score"]
        + 0.40 * stats["value_score"]
    )

    product_scores: dict[str, list[dict]] = {}
    for cluster, grp in stats.groupby("cluster"):
        top = grp.sort_values("nbo_score", ascending=False)
        product_scores[str(cluster)] = top[
            ["product_name", "nbo_score", "avg_revenue", "popularity_score"]
        ].to_dict("records")

    for cl, prods in product_scores.items():
        print(f"  Кластер {cl}: {len(prods)} продуктов, топ — {prods[0]['product_name']}")

    return product_scores


def precompute_company_results(
    df: pd.DataFrame,
    model: CatBoostRegressor,
    product_scores: dict,
    q33: float,
    q66: float,
    median_features: dict,
    top_nbo: int = 10,
) -> tuple[dict, dict]:
    print("Предрасчёт результатов по клиентам...")

    X = df[FEATURE_COLS].copy()
    for col in CAT_FEATURES:
        X[col] = X[col].astype(str).fillna("unknown")
    for col in median_features:
        if col in X.columns:
            X[col] = X[col].fillna(median_features[col])

    preds = np.expm1(model.predict(X))

    company_ltv: dict = {}
    company_nbo: dict = {}

    for idx, row in df.reset_index(drop=True).iterrows():
        cid = str(row["company_id"])
        ltv = float(preds[idx])
        segment = "high" if ltv > q66 else ("mid" if ltv > q33 else "low")
        company_ltv[cid] = {
            "predicted_ltv": ltv,
            "ltv_segment": segment,
            "cluster": str(row["cluster"]),
        }

        cluster = str(row["cluster"])
        prods = product_scores.get(cluster, [])[:top_nbo]
        company_nbo[cid] = prods

    print(f"  Рассчитано: {len(company_ltv)} компаний")
    return company_ltv, company_nbo


def compute_segment_stats(company_ltv: dict) -> list[dict]:
    records: dict = {"low": [], "mid": [], "high": []}
    for v in company_ltv.values():
        records[v["ltv_segment"]].append(v["predicted_ltv"])

    result = []
    for seg, vals in records.items():
        if vals:
            result.append({
                "ltv_segment": seg,
                "avg_ltv": float(np.mean(vals)),
                "clients_count": len(vals),
            })
    return sorted(result, key=lambda x: x["avg_ltv"])


def main() -> None:
    df_companies, df_revenue = load_data()
    df = build_features(df_companies, df_revenue)
    df, kmeans, scaler = train_clustering(df)
    model, median_features, (q33, q66) = train_ltv_model(df)
    product_scores = compute_nbo_scores(df, df_revenue)
    company_ltv, company_nbo = precompute_company_results(
        df, model, product_scores, q33, q66, median_features
    )
    segment_stats = compute_segment_stats(company_ltv)

    all_products: dict = {}
    for prods in product_scores.values():
        for p in prods:
            name = p["product_name"]
            if name not in all_products:
                all_products[name] = {"nbo_score": 0.0, "avg_revenue": p["avg_revenue"], "count": 0}
            all_products[name]["nbo_score"] += p["nbo_score"]
            all_products[name]["count"] += 1
    global_top_nbo = sorted(
        [{"product_name": k, **v} for k, v in all_products.items()],
        key=lambda x: x["nbo_score"],
        reverse=True,
    )

    print("\nСохранение артефактов...")

    model.save_model(str(MODELS_DIR / "catboost_ltv.cbm"))
    joblib.dump(kmeans, MODELS_DIR / "kmeans.joblib")
    joblib.dump(scaler, MODELS_DIR / "cluster_scaler.joblib")

    artifacts = {
        "feature_cols": FEATURE_COLS,
        "cat_features": CAT_FEATURES,
        "cluster_features": CLUSTER_FEATURES,
        "ltv_q33": q33,
        "ltv_q66": q66,
        "median_features": median_features,
        "company_ltv": company_ltv,
        "company_nbo": company_nbo,
        "product_scores": product_scores,
        "segment_stats": segment_stats,
        "global_top_nbo": global_top_nbo,
        "model_version": "1.0.0",
    }
    joblib.dump(artifacts, MODELS_DIR / "artifacts.joblib")

    print(f"\nГотово. Файлы в {MODELS_DIR}:")
    for f in MODELS_DIR.iterdir():
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
