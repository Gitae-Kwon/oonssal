import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from prophet import Prophet
from datetime import datetime

# 🔐 Neon DB 접속 정보
user = st.secrets["DB_USER"]
password = st.secrets["DB_PASSWORD"]
host = st.secrets["DB_HOST"]
port = st.secrets["DB_PORT"]
db = st.secrets["DB_NAME"]

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}?sslmode=require")

# 📥 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_sql("SELECT date, contents_title, totalcoins FROM fra_daily", con=engine)
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

st.title("📊 웹툰 매출 대시보드")

# 콘텐츠 선택
titles = df["contents_title"].unique()
selected_title = st.selectbox("🔍 콘텐츠 선택", sorted(titles))

# 해당 콘텐츠 데이터 필터링
df_selected = df[df["contents_title"] == selected_title][["date", "totalcoins"]]
df_selected = df_selected.groupby("date").sum().reset_index()

st.subheader(f"📈 매출 추이: {selected_title}")
st.line_chart(df_selected.set_index("date")["totalcoins"])

# 🔮 Prophet 예측
st.subheader("🔮 매출 예측 (7일)")
prophet_df = df_selected.rename(columns={"date": "ds", "totalcoins": "y"})

model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)

st.line_chart(forecast.set_index("ds")[["yhat"]])
