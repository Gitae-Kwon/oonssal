import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from prophet import Prophet
from datetime import timedelta

# 🔐 Neon DB 접속 정보 (Streamlit Secrets에서 가져옴)
user = st.secrets["DB_USER"]
password = st.secrets["DB_PASSWORD"]
host = st.secrets["DB_HOST"]
port = st.secrets["DB_PORT"]
db = st.secrets["DB_NAME"]

# 🚀 DB 연결
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}?sslmode=require")

# 📥 데이터 불러오기 함수
@st.cache_data
def load_data():
    df = pd.read_sql('SELECT date, "Title", "Total_coins" FROM fra_daily', con=engine)
    df["date"] = pd.to_datetime(df["date"])
    return df

# 데이터 로딩
df = load_data()

# 🖥️ 대시보드 제목
st.title("📊 웹툰 매출 대시보드")

# 콘텐츠 선택 필터
titles = df["Title"].unique()
selected_title = st.selectbox("🔍 콘텐츠 선택", sorted(titles))

# 선택된 콘텐츠의 매출 추이
df_selected = df[df["Title"] == selected_title][["date", "Total_coins"]]
df_selected = df_selected.groupby("date").sum().reset_index()

# 최근 90일만 시각화
recent_90 = df_selected[df_selected["date"] >= df_selected["date"].max() - timedelta(days=90)]

st.subheader(f"📈 최근 3개월 매출 추이: {selected_title}")
st.line_chart(recent_90.set_index("date")["Total_coins"])

# 🔮 매출 예측 (7일)
st.subheader("🔮 매출 예측 (향후 7일)")

if df_selected.shape[0] < 10:
    st.warning("⚠️ 예측을 위한 데이터가 충분하지 않습니다.")
else:
    prophet_df = df_selected.rename(columns={"date": "ds", "Total_coins": "y"})

    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    # 마지막 30일 + 향후 7일만 시각화
    plot_df = forecast[["ds", "yhat", "yhat_upper", "yhat_lower"]].tail(37).set_index("ds")
    st.line_chart(plot_df[["yhat"]])
