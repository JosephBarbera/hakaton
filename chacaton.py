# -*- coding: utf-8 -*-
"""
"Поймай мошенника" — прототип системы выявления подозрительных банковских операций.
Математическая модель: логистическая регрессия на признаках, построенных ТОЛЬКО
по прошлой истории клиента (без утечки из будущего).
Запуск:  python pipeline.py
Результат: results_test.csv  (+ отчёт качества в консоли)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

FEATURES = [
    "log_amt_ratio", "region_mismatch", "is_new_device", "device_not_own",
    "hour_night", "balance_ratio", "income_ratio", "cnt_prev_60m",
    "known_recipient", "hist_size",
]
RU_NAMES = {
    "log_amt_ratio": "сумма сильно превышает привычную",
    "region_mismatch": "регион операции не совпадает с домашним",
    "is_new_device": "устройство не использовалось ранее",
    "device_not_own": "операция не с основного устройства клиента",
    "hour_night": "операция в ночные часы (0-5)",
    "balance_ratio": "сумма велика относительно остатка на счёте",
    "income_ratio": "сумма велика относительно месячного дохода",
    "cnt_prev_60m": "много операций за последний час",
    "known_recipient": "получатель встречался ранее (снижает риск)",
    "hist_size": "мало истории по клиенту (оценка менее надёжна)",
}

def load_data(folder="."):
    clients = pd.read_csv(f"{folder}/clients.csv")
    hist = pd.read_csv(f"{folder}/transactions_history.csv", parse_dates=["timestamp"])
    test = pd.read_csv(f"{folder}/transactions_test.csv", parse_dates=["timestamp"])
    for df in (hist, test):
        df["amount_rub"] = pd.to_numeric(df["amount_rub"], errors="coerce")
        df["balance_before_rub"] = pd.to_numeric(df["balance_before_rub"], errors="coerce")
        df["is_new_device"] = pd.to_numeric(df["is_new_device"], errors="coerce").fillna(0)
    return clients, hist, test

def build_features(df, hist, clients):
    """Признаки для каждой операции строим только по её ПРОШЛОЙ истории."""
    hist = hist.sort_values("timestamp").reset_index(drop=True)
    h = hist[["customer_id", "timestamp", "amount_rub", "device_id",
              "region", "recipient_account_id", "is_fraud"]].copy()

    # глобальная медиана сумм — запас для клиентов без истории
    global_median = h["amount_rub"].median()

    # устойчивые статистики истории клиента (медиана, не среднее — нечувствительна к выбросам)
    med_amount = h.groupby("customer_id")["amount_rub"].median()
    main_device = h.groupby("customer_id")["device_id"].agg(
        lambda s: s.mode().iat[0] if len(s) else None)
    known_rec = h.groupby("customer_id")["recipient_account_id"].agg(set)

    df = df.merge(clients[["client_id", "home_region", "monthly_income_rub"]],
                  left_on="customer_id", right_on="client_id", how="left")

    # количество операций клиента за предыдущий час (только прошлые!)
    h_indexed = h.set_index("timestamp")
    def cnt_60m(row):
        win = h_indexed.loc[row["timestamp"] - pd.Timedelta(minutes=60):
                            row["timestamp"] - pd.Timedelta(seconds=1)]
        return int((win["customer_id"] == row["customer_id"]).sum())
    df["cnt_prev_60m"] = df.apply(cnt_60m, axis=1)

    df["hist_size"] = df["customer_id"].map(h.groupby("customer_id").size()).fillna(0)
    df["prev_median"] = df["customer_id"].map(med_amount)
    df["main_device"] = df["customer_id"].map(main_device)
    df["known_rec"] = df["customer_id"].map(known_rec)

    df["amt_ratio"] = df["amount_rub"] / df["prev_median"].replace(0, np.nan)
    df["amt_ratio"] = df["amt_ratio"].fillna(df["amount_rub"] / global_median)
    df["amt_ratio"] = df["amt_ratio"].clip(0, 50)          # защита от взрыва объяснений
    df["log_amt_ratio"] = np.log1p(df["amt_ratio"])

    df["region_mismatch"] = (df["region"] != df["home_region"]).astype(float)
    df["is_new_device"] = df["is_new_device"].astype(float)

    df["device_not_own"] = (df["device_id"] != df["main_device"]).astype(float)
    df["known_recipient"] = df.apply(
        lambda r: float(r["recipient_account_id"] in r["known_rec"])
        if isinstance(r["known_rec"], set) else 0.0, axis=1)
    df["hour_night"] = ((df["timestamp"].dt.hour >= 0) &
                        (df["timestamp"].dt.hour <= 5)).astype(float)
    df["balance_ratio"] = (df["amount_rub"] / df["balance_before_rub"]).clip(0, 5).fillna(0.5)
    df["income_ratio"] = (df["amount_rub"] / df["monthly_income_rub"]).clip(0, 5).fillna(0.1)
    df[FEATURES] = df[FEATURES].fillna(0.0)
    return df

def train_and_evaluate(hist_feat):
    """Обучение только на ПОДТВЕРЖДЁННЫХ операциях; проверка на отложенном периоде по времени."""
    labeled = hist_feat[hist_feat["is_fraud"].notna()].copy()
    labeled = labeled.sort_values("timestamp")
    split_ts = labeled["timestamp"].quantile(0.8)          # разбиение по времени, без утечки
    train, valid = labeled[labeled["timestamp"] <= split_ts], labeled[labeled["timestamp"] > split_ts]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[FEATURES]);  y_tr = train["is_fraud"].astype(int).values
    X_va = scaler.transform(valid[FEATURES]);      y_va = valid["is_fraud"].astype(int).values

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_tr, y_tr)

    p = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, p)
    return model, scaler, (train, valid, p, auc)

def explain(row, model, scaler, k=3):
    """Вклады признаков в оценку риска: w_i * (x_i - среднее) / sd_i."""
    x = (row[FEATURES].values.astype(float) - scaler.mean_) / scaler.scale_
    contrib = pd.Series(model.coef_[0] * x, index=FEATURES).sort_values(ascending=False)
    top = [RU_NAMES[f] for f in contrib.index[:k] if contrib[f] > 0.05]
    return "; ".join(top) if top else "значимых отклонений от поведения клиента не выявлено"

def category(score):
    if score >= 0.90: return "высокий риск — обязательная проверка"
    if score >= 0.50: return "требует проверки"
    return "норма"

def rule_baseline(df):
    """Простой правиловый бейзлайн для сравнения (карточка 2 подсказок: ratio > 5)."""
    return ((df["amt_ratio"] > 5) |
            ((df["is_new_device"] == 1) & (df["region_mismatch"] == 1) & (df["amt_ratio"] > 3))
            ).astype(int)

if __name__ == "__main__":
    clients, hist, test = load_data()
    hist_feat = build_features(hist, hist, clients)
    model, scaler, (train, valid, p_val, auc) = train_and_evaluate(hist_feat)

    # --- качество на отложенном периоде (известные ответы) ---
    y_va = valid["is_fraud"].astype(int).values
    print(f"ROC-AUC на отложенном периоде: {auc:.3f}")
    for thr in (0.5, 0.75, 0.90):
        pred = (p_val >= thr)
        rec = pred[y_va == 1].mean()
        fp = int((pred & (y_va == 0)).sum())
        prec = pred[y_va == 1].sum() / max(pred.sum(), 1)
        print(f"порог {thr:.2f}: полнота {rec:.0%}, точность {prec:.0%}, ложных тревог {fp}")
    rule = rule_baseline(valid)
    print(f"правиловый бейзлайн: полнота {rule[y_va == 1].mean():.0%}, "
          f"ложных тревог {int((rule & (y_va == 0)).sum())}")

    # --- прогноз по тесту ---
    test_feat = build_features(test, hist, clients)
    p_test = model.predict_proba(scaler.transform(test_feat[FEATURES]))[:, 1]
    test_feat["risk_score"] = np.round(p_test, 4)
    test_feat["категория"] = [category(s) for s in p_test]
    test_feat["объяснение"] = test_feat.apply(lambda r: explain(r, model, scaler), axis=1)

    out = test_feat[["transaction_id", "customer_id", "timestamp", "amount_rub",
                     "risk_score", "категория", "объяснение"]]
    out = out.sort_values("risk_score", ascending=False)
    out.to_csv("results_test.csv", index=False, encoding="utf-8-sig")
    print(f"\nГотово: results_test.csv, строк {len(out)}, "



f"высокий риск {(p_test >= 0.9).sum()}, требует проверки {((p_test >= 0.5) & (p_test < 0.9)).sum()}")

