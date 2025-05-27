import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from prophet import Prophet
from datetime import datetime, timedelta

# 🔐 DB 연결 생략 (engine 이미 설정되어 있다고 가정)

@st.cache_data
def load_data():
    df = pd.read_sql('SELECT date, "Title", "Total_coins" FROM fra_daily', con=engine)
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

st.title("📊 웹툰 매출 대시보드")

titles = df["Title"].unique()
selected_title = st.selectbox("🔍 콘텐츠 선택", sorted(titles))

df_selected = df[df["Title"] == selected_title][["date", "Total_coins"]]
df_selected = df_selected.groupby("date").sum().reset_index()

# 최근 90일만 보기 (필터)
recent_90 = df_selected[df_selected["date"] >= df_selected["date"].max() - timedelta(days=90)]

st.subheader(f"📈 최근 3개월 매출 추이: {selected_title}")
st.line_chart(recent_90.set_index("date")["Total_coins"])

# 🔮 Prophet 예측 (7일)
st.subheader("🔮 매출 예측 (향후 7일)")
prophet_df = df_selected.rename(columns={"date": "ds", "Total_coins": "y"})

model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)

# 마지막 30일 + 향후 7일 시각화
plot_df = forecast[["ds", "yhat", "yhat_upper", "yhat_lower"]].tail(37)
plot_df.set_index("ds", inplace=True)

st.line_chart(plot_df)
