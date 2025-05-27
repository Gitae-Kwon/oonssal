import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from prophet import Prophet
from datetime import timedelta
from prophet.make_holidays import make_holidays_df

# 다음 2년간 프랑스 공휴일
holidays_fr = make_holidays_df(year_list=[2024, 2025], country="FR")

# DB 연결 정보
user = st.secrets["DB_USER"]
password = st.secrets["DB_PASSWORD"]
host = st.secrets["DB_HOST"]
port = st.secrets["DB_PORT"]
db = st.secrets["DB_NAME"]
engine = create_engine(
    f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}?sslmode=require"
)

@st.cache_data
def load_data():
    df = pd.read_sql('SELECT date, "Title", "Total_coins" FROM fra_daily', con=engine)
    df["date"] = pd.to_datetime(df["date"])
    return df

# 데이터 로드
df = load_data()
st.title("📊 웹툰 매출 데이터 + 이벤트 인사이트 대시보드")

# 콘텐츠 선택 (전체/개별)
options = ["전체 콘텐츠"] + sorted(df["Title"].unique())
selected_title = st.selectbox("🔍 콘텐츠 선택", options)

# 선택된 콘텐츠 또는 전체 매출 데이터 준비
df_selected = (
    df.groupby("date")["Total_coins"].sum().reset_index()
    if selected_title == "전체 콘텐츠"
    else df[df["Title"] == selected_title][["date", "Total_coins"]]
)
# 날짜별 합계 및 정렬
df_selected = df_selected.groupby("date")["Total_coins"].sum().reset_index()
df_selected = df_selected.sort_values("date")

# 이벤트 플래그: 7일 이동 평균 대비 threshold 배수
threshold = 1.7 if selected_title != "전체 콘텐츠" else 1.3
df_selected["rolling_avg"] = (
    df_selected["Total_coins"].rolling(window=7, center=True, min_periods=1).mean()
)
df_selected["event_flag"] = df_selected["Total_coins"] > df_selected["rolling_avg"] * threshold
df_selected["weekday"] = df_selected["date"].dt.day_name()

# 1) 이벤트 발생 요일 분포 계산 및 시각화
weekday_event_stats = df_selected[df_selected["event_flag"]]["weekday"].value_counts()
# 요일별 전체 일수 카운트
total_days_by_weekday = df_selected["weekday"].value_counts()
# 요일별 이벤트 발생 비율
event_rate_by_weekday = (weekday_event_stats / total_days_by_weekday).fillna(0)

st.subheader("🌟 이벤트 발생 요일 분포")
if not weekday_event_stats.empty:
    st.bar_chart(weekday_event_stats)
else:
    st.info(
        f"🗒️ '{selected_title}' 기준 이벤트(기준={threshold*100:.0f}% 증가) 기록이 없습니다."
    )

# 2) 공휴일 중 이벤트 효과가 낮은 요일 분석
merged = pd.merge(
    df_selected,
    holidays_fr.rename(columns={"ds": "date"}),
    how="inner",
    on="date"
)
merged_weekday = merged[~merged["weekday"].isin(["Saturday", "Sunday"])]
weak_holidays = merged_weekday[~merged_weekday["event_flag"]]
weak_by_weekday = weak_holidays["weekday"].value_counts()

st.subheader("🤔 이벤트 효과가 낮았던 공휴일 요일 분포")
st.bar_chart(weak_by_weekday)

# 3) 최근 3개월 매출 추이
recent = df_selected[df_selected["date"] >= df_selected["date"].max() - timedelta(days=90)]
sub_title = "전체 콘텐츠" if selected_title == "전체 콘텐츠" else selected_title
st.subheader(f"📈 '{sub_title}' 최근 3개월 매출 추이")
st.line_chart(recent.set_index("date")["Total_coins"])

# 4) Prophet 예측 (향후 30일)
prophet_df = df_selected.rename(columns={"date": "ds", "Total_coins": "y"})
model = Prophet()
model.add_country_holidays(country_name="FR")
model.fit(prophet_df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# 과거 최대 날짜 기준으로 30일 미래만 필터
today_max = df_selected["date"].max()
future_30 = forecast[forecast["ds"] > today_max]

st.subheader("🔮 향후 30일 매출 예측")
st.line_chart(future_30.set_index("ds")["yhat"])

# 5) 이벤트 예정일 선택 및 적용 기능
st.subheader("🗓 이벤트 예정일 체크 및 적용")
event_input = st.date_input(
    "이벤트 가능성 있는 날짜 선택", [], format="YYYY-MM-DD", key="event_input"
)
apply = st.button("이벤트 적용")
if apply:
    if event_input:
        sel_date = event_input[0] if isinstance(event_input, list) else event_input
        weekday = sel_date.strftime("%A")
        rate = event_rate_by_weekday.get(weekday, 0)
        st.write(f"📈 과거 {weekday} 이벤트 발생 비율: {rate:.1%}")
        if sel_date in future_7["ds"].dt.date.tolist():
            st.success(f"🚀 {sel_date}은 예측 기간(향후 7일)에 포함됩니다.")
            # 예측 차트 갱신
            st.line_chart(future_7.set_index("ds")["yhat"])
        else:
            st.warning("⚠️ 선택한 이벤트일이 향후 7일 예측 기간에 포함되지 않습니다.")
    else:
        st.warning("⚠️ 먼저 이벤트 가능일을 선택해주세요.")
