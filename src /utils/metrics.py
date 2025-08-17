import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

# --- Метрики ---

def precision_at_k(y_true, y_pred, k=10):
    y_pred_k = y_pred[:k]
    return len(set(y_pred_k) & set(y_true)) / k if k > 0 else 0.0

def recall_at_k(y_true, y_pred, k=10):
    y_pred_k = y_pred[:k]
    return len(set(y_pred_k) & set(y_true)) / len(set(y_true)) if y_true else 0.0

def hit_rate_at_k(y_true, y_pred, k=10):
    y_pred_k = y_pred[:k]
    return int(len(set(y_pred_k) & set(y_true)) > 0)

def dcg_at_k(y_true, y_pred, k=10):
    y_true_set = set(y_true)
    dcg = 0.0
    for i, p in enumerate(y_pred[:k]):
        if p in y_true_set:
            dcg += 1.0 / np.log2(i + 2)
    return dcg

def ndcg_at_k(y_true, y_pred, k=10):
    ideal_dcg = dcg_at_k(y_true, y_true, min(k, len(y_true)))
    return dcg_at_k(y_true, y_pred, k) / ideal_dcg if ideal_dcg > 0 else 0.0

def average_precision_at_k(y_true, y_pred, k=10):
    """
    AP@k: учитывает порядок релевантных элементов.
    """
    if not y_true:
        return 0.0
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(y_pred[:k], start=1):
        if p in y_true:
            num_hits += 1.0
            score += num_hits / i
    return score / min(len(y_true), k)

def mean_average_precision(users_true, users_pred, k=10):
    """
    MAP@k: среднее AP@k по всем пользователям.
    users_true: dict {user: set(items)}
    users_pred: dict {user: list(pred_items)}
    """
    ap_scores = []
    for u in users_true:
        ap = average_precision_at_k(users_true[u], users_pred.get(u, []), k)
        ap_scores.append(ap)
    return float(np.mean(ap_scores)) if ap_scores else 0.0


# --- Обёртка для моделей ---

def evaluate_model(model, test_df, user_col="user_id", item_col="item_id", k=10) -> Dict[str, Any]:
    """
    Прогоняет модель по тестовому датасету и считает средние метрики.
    """
    results = defaultdict(list)
    users = test_df[user_col].unique()
    recs = model.recommend(users=users, N=k)

    for u, pred_items in zip(users, recs):
        true_items = set(test_df.loc[test_df[user_col] == u, item_col])

        results["precision"].append(precision_at_k(true_items, pred_items, k))
        results["recall"].append(recall_at_k(true_items, pred_items, k))
        results["hit_rate"].append(hit_rate_at_k(true_items, pred_items, k))
        results["ndcg"].append(ndcg_at_k(true_items, pred_items, k))
        results["ap"].append(average_precision_at_k(true_items, pred_items, k))

    # усредняем по пользователям
    return {metric: float(np.mean(vals)) for metric, vals in results.items()}
