import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from prophet import Prophet
from datetime import timedelta
from prophet.make_holidays import make_holidays_df

# 해당 다음 2년간 프랑스 공휴일
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

# 데이터 로드
df = load_data()
st.title("📊 웹툰 매출 데이터 + 이벤트 인사이트 대시보드")

# 콘텐츠 선택 (전체 포함)
options = ["전체 콘텐츠"] + sorted(df["Title"].unique())
selected_title = st.selectbox("🔍 콘텐츠 선택", options)

# 선택 또는 전체 매출 데이터 준비
df_selected = (
    df.groupby("date")["Total_coins"].sum().reset_index()
    if selected_title == "전체 콘텐츠"
    else df[df["Title"] == selected_title][["date", "Total_coins"]]
)
df_selected = df_selected.groupby("date")["Total_coins"].sum().reset_index()
df_selected = df_selected.sort_values("date")

# 1) 이벤트일 검증 (7일간 평균 대비 threshold 배수)
# 기존 df_selected["event_flag"] 계산부를 아래로 교체
threshold = 1.7 if selected_title != "전체 콘텐츠" else 1.3
df_selected["rolling_avg"] = (
    df_selected["Total_coins"]
    .rolling(window=7, center=True, min_periods=1)
    .mean()
)
df_selected["event_flag"] = df_selected["Total_coins"] > df_selected["rolling_avg"] * threshold
df_selected["weekday"] = df_selected["date"].dt.day_name()
event_dates = df_selected[df_selected["event_flag"]]["date"].tolist()

# 2) 이벤트 발생 요일 분포 (데이터가 없으면 안내 메시지)
weekday_event_stats = df_selected[df_selected["event_flag"]]["weekday"].value_counts()
st.subheader("🌟 이벤트 발생 요일 분포")
if not weekday_event_stats.empty:
    st.bar_chart(weekday_event_stats)
else:
    st.info("🗒️ 전체 매출 기준 급등 이벤트(170% 이상) 또는 전체 콘텐츠 기준(130% 이상) 이벤트가 없습니다.\n임계치를 낮춰보세요.")

# 2) 공휴일 중 이벤트 효과가 낮은 요일 분석
merged = pd.merge(df_selected, holidays_fr.rename(columns={"ds": "date"}), how="inner", on="date")
merged_weekday = merged[~merged["weekday"].isin(["Saturday", "Sunday"])]
weak_holidays = merged_weekday[~merged_weekday["event_flag"]]
weak_by_weekday = weak_holidays["weekday"].value_counts()
st.subheader("🤔 이벤트 효과가 낮았던 공휴일 요일 분포")
st.bar_chart(weak_by_weekday)

# 3) 최근 3개월 매출 추이
recent = df_selected[df_selected["date"] >= df_selected["date"].max() - timedelta(days=90)]
sub_title = "전체" if selected_title == "전체 콘텐츠" else selected_title
st.subheader(f"📈 '{sub_title}' 최근 3개월 매출 추이")
st.line_chart(recent.set_index("date")["Total_coins"])

# 4) Prophet 예측 (향후 7일)
prophet_df = df_selected.rename(columns={"date": "ds", "Total_coins": "y"})
model = Prophet()
model.add_country_holidays(country_name="FR")
model.fit(prophet_df)
future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)
future_7 = forecast[forecast["ds"] > df_selected["date"].max()]
st.subheader("🔮 향후 7일 매출 예측")
st.line_chart(future_7.set_index("ds")["yhat"])

# 5) 이벤트 예정일 선택 기능
st.subheader("🗓 이벤트 예정일 체크")
event_input = st.date_input("이벤트 가능성 있는 날짜 선택", [], format="YYYY-MM-DD", key="event_input")
if event_input:
    st.success(f"🚀 선택된 이벤트일: {event_input}")
