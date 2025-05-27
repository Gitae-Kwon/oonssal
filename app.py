import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from prophet import Prophet
from datetime import timedelta
from prophet.make_holidays import make_holidays_df

# 해당 다음 2년간 프랭스 공휴일
holidays_fr = make_holidays_df(year_list=[2024, 2025], country="FR")

# DB 연결
user = st.secrets["DB_USER"]
password = st.secrets["DB_PASSWORD"]
host = st.secrets["DB_HOST"]
port = st.secrets["DB_PORT"]
db = st.secrets["DB_NAME"]
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}?sslmode=require")

@st.cache_data
def load_data():
    df = pd.read_sql('SELECT date, "Title", "Total_coins" FROM fra_daily', con=engine)
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()
st.title("📊 웹툰 매주 데이션 + 이벤트 통화 데스크")

# 선택된 컨테츠 포함 데이터
selected_title = st.selectbox("🔍 컨테츠 선택", sorted(df["Title"].unique()))
df_selected = df[df["Title"] == selected_title].copy()
df_selected = df_selected.groupby("date")["Total_coins"].sum().reset_index()
df_selected = df_selected.sort_values("date")

# 이벤트일 검증 (7일 가운 포함 평균 보다 70%이상)
df_selected["rolling_avg"] = df_selected["Total_coins"].rolling(window=7, center=True, min_periods=1).mean()
df_selected["event_flag"] = df_selected["Total_coins"] > df_selected["rolling_avg"] * 1.7
df_selected["weekday"] = df_selected["date"].dt.day_name()
event_dates = df_selected[df_selected["event_flag"]]["date"].tolist()

# 일주일 단위 통계
weekday_event_stats = df_selected[df_selected["event_flag"]]["weekday"].value_counts()
st.subheader("🌟 이벤트 발생 일요일 분포")
st.bar_chart(weekday_event_stats)

# 프랭스 공휴일 + 통신 데이터 바이 메지
merged = pd.merge(df_selected, holidays_fr.rename(columns={"ds": "date"}), how="inner", on="date")
merged_weekday = merged[~merged["weekday"].isin(["Saturday", "Sunday"])]
weak_holidays = merged_weekday[~merged_weekday["event_flag"]]
weak_by_weekday = weak_holidays["weekday"].value_counts()

st.subheader(":thinking_face: 가장 협조가 없어 보이는 공휴일의 일요")
st.bar_chart(weak_by_weekday)

# 컨테츠 현황 시각화
recent = df_selected[df_selected["date"] >= df_selected["date"].max() - timedelta(days=90)]
st.subheader(f"현재 {selected_title} 최근 3개월 매주")
st.line_chart(recent.set_index("date")["Total_coins"])

# Prophet 예측
prophet_df = df_selected.rename(columns={"date": "ds", "Total_coins": "y"})
model = Prophet()
model.add_country_holidays(country_name="FR")
model.fit(prophet_df)

future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)
future_7 = forecast[forecast["ds"] > df_selected["date"].max()]

st.subheader("🔮 7일 가장 엄장 발생 예측")
st.line_chart(future_7.set_index("ds")["yhat"])

# 이벤트 예정일 선택기능
st.subheader("🗓 이벤트 예정일 체크")
event_input = st.date_input("통화 가능성 있는 날짜 선택", [], format="YYYY-MM-DD", key="event_input")

if event_input:
    st.success(f"🚀 선택한 이벤트일: {event_input}")
