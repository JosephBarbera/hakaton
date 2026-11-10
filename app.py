# -*- coding: utf-8 -*-
"""
Веб-интерфейс прототипа "Поймай мошенника".
Запуск:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import pipeline as pl

st.set_page_config(page_title="Поймай мошенника", layout="wide")
st.title("🛡 Поймай мошенника")
st.caption("Прототип выявления подозрительных банковских операций — ИИМРТ, хакатон первокурсников")

CATEGORIES = ["Все", "Высокий риск", "Требует проверки", "Норма"]
CAT_PREFIX = {"Высокий риск": "высокий риск",
              "Требует проверки": "требует проверки",
              "Норма": "норма"}

HISTORY_COLS = ["timestamp", "transaction_type", "amount_rub", "region",
                "device_id", "recipient_account_id", "is_fraud"]


@st.cache_resource(show_spinner="Обучаем модель...")
def get_model():
    model, scaler, out = pl.run(verbose=False)
    return model, scaler, out


@st.cache_resource(show_spinner="Загружаем данные...")
def get_raw():
    clients, hist, test = pl.load_data()
    return clients, hist, test


try:
    model, scaler, results = get_model()
    clients, hist, test = get_raw()
except FileNotFoundError as e:
    st.error(f"Не найден файл данных: {e}. Положите все CSV рядом с app.py.")
    st.stop()

# ---------- сводка ----------
hi = results["категория"].str.startswith("высокий").sum()
mid = results["категория"].str.startswith("требует").sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Операций проанализировано", len(results))
c2.metric("Высокий риск", int(hi))
c3.metric("Требует проверки", int(mid))
c4.metric("Норма", int(len(results) - hi - mid))

st.divider()

# ---------- таблица результатов ----------
st.subheader("Список операций (отсортирован по риску)")

cat_filter = st.selectbox("Категория:", CATEGORIES)
if cat_filter == "Все":
    df_view = results
else:
    df_view = results[results["категория"].str.startswith(CAT_PREFIX[cat_filter])]

top_n = st.slider("Показать операций:", 50, 10000, 100, step=50)
st.dataframe(df_view.head(top_n), use_container_width=True, hide_index=True)
st.caption(f"Показано {min(top_n, len(df_view))} из {len(df_view)} операций выбранной категории")

st.download_button("Выгрузить полный отчёт (CSV)",
                   results.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                   "results_test.csv", "text/csv")

st.divider()

# ---------- разбор отдельной операции ----------
st.subheader("Разбор отдельной операции")
tx_id = st.text_input("ID операции (например, TX0030037):",
                      placeholder="TX......").strip().upper()

if tx_id:
    match = results[results["transaction_id"] == tx_id]
    if match.empty:
        st.warning(f"Операция {tx_id} не найдена — проверьте номер.")
    else:
        row = match.iloc[0]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Оценка риска", f"{row['risk_score']:.2f}")
            st.write(f"**Категория:** {row['категория']}")
            st.write(f"**Клиент:** {row['customer_id']}")
            st.write(f"**Сумма:** {row['amount_rub']:,.0f} руб.")
            st.write(f"**Время:** {row['timestamp']}")
        with col2:
            st.markdown("**Почему операция отмечена:**")
            st.info(row["объяснение"])

        # ---------- привычки клиента на момент операции ----------
        st.markdown("#### Привычки клиента на момент операции")
        h = hist[hist["customer_id"] == row["customer_id"]].copy()
        ts = pd.to_datetime(row["timestamp"])
        if "timestamp" in h.columns:
            h["timestamp"] = pd.to_datetime(h["timestamp"])
            past = h[h["timestamp"] < ts].sort_values("timestamp")
        else:
            past = h

        if past.empty:
            st.info("Прошлых операций нет — клиент новый, оценка по общим правилам.")
        else:
            med = float(past["amount_rub"].median())
            ratio = row["amount_rub"] / med if med > 0 else None
            if "timestamp" in past.columns:

                last_hour = past[past["timestamp"] >= ts - pd.Timedelta(hours=1)]
            else:
                last_hour = past.iloc[0:0]
            if "region" in past.columns and not past["region"].dropna().empty:
                usual_region = past["region"].mode().iloc[0]
            else:
                usual_region = "—"

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Медиана прошлых сумм", f"{med:,.0f} руб.")
            m2.metric("Во сколько раз выше медианы",
                      f"{ratio:.1f}x" if ratio else "—")
            m3.metric("Операций за последний час", str(len(last_hour)))
            m4.metric("Привычный регион", str(usual_region))

        # ---------- история клиента ----------
        st.divider()
        st.subheader(f"История клиента {row['customer_id']}")
        st.write(f"Всего операций в истории: {len(h)} — показаны последние 20 до момента операции")
        cols = [c for c in HISTORY_COLS if c in h.columns]
        st.dataframe(past.tail(20)[cols], use_container_width=True, hide_index=True)
else:
    st.info("Введите ID операции, чтобы открыть её разбор. Доступны все 10 000 операций.")

# ---------- распределение риска ----------
st.divider()
st.subheader("Распределение оценок риска по тестовым операциям")
fig = px.histogram(results, x="risk_score", nbins=40,
                   labels={"risk_score": "оценка риска"})
st.plotly_chart(fig, use_container_width=True)