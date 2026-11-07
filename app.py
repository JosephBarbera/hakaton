# -*- coding: utf-8 -*-
"""
Веб-интерфейс прототипа "Поймай мошенника".
Запуск:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

import pipeline as pl

st.set_page_config(page_title="Поймай мошенника", layout="wide")
st.title("🛡 Поймай мошенника")
st.caption("Прототип выявления подозрительных банковских операций — ИИМРТ, хакатон первокурсников")


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
hi = (results["категория"] == "высокий риск — обязательная проверка").sum()
mid = (results["категория"] == "требует проверки").sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Операций проанализировано", len(results))
c2.metric("Высокий риск", int(hi))
c3.metric("Требует проверки", int(mid))
c4.metric("Норма", int(len(results) - hi - mid))

st.divider()

# ---------- таблица результатов ----------
st.subheader("Список операций (отсортирован по риску)")
top_n = st.slider("Показать операций:", 50, 10000, 100)
st.dataframe(results.head(top_n), use_container_width=True, hide_index=True)

st.download_button("⬇ Выгрузить полный отчёт (CSV)",
                   results.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                   "results_test.csv", "text/csv")

st.divider()

# ---------- карточка операции ----------
st.subheader("Разбор отдельной операции")
tx_id = st.selectbox("Выберите операцию:", results["transaction_id"].head(300))
row = results[results["transaction_id"] == tx_id].iloc[0]

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Оценка риска", f"{row['risk_score']:.2f}")
    st.write(f"**Категория:** {row['категория']}")
    st.write(f"**Клиент:** {row['customer_id']}")
    st.write(f"**Сумма:** {row['amount_rub']:,.0f} ₽")
    st.write(f"**Время:** {row['timestamp']}")
with col2:
    st.markdown("**Почему операция отмечена:**")
    st.info(row["объяснение"])

# ---------- история клиента ----------
st.divider()
st.subheader(f"История клиента {row['customer_id']}")
h = hist[hist["customer_id"] == row["customer_id"]].sort_values("timestamp")
st.write(f"Всего операций в истории: {len(h)}")
st.dataframe(h[["timestamp", "transaction_type", "amount_rub", "region",
                "device_id", "recipient_account_id", "is_fraud"]],
             use_container_width=True, hide_index=True)

# ---------- распределение риска ----------
st.divider()
st.subheader("Распределение оценок риска по тестовым операциям")
import plotly.express as px
fig = px.histogram(results, x="risk_score", nbins=40,
                   labels={"risk_score": "оценка риска"})
st.plotly_chart(fig, use_container_width=True)
