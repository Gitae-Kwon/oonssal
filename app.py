import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from prophet import Prophet
from datetime import timedelta
from prophet.make_holidays import make_holidays_df
import altair as alt

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

# 데이터 로드 함수 정의
@st.cache_data
def load_coin_data():
    df = pd.read_sql('SELECT date, "Title", "Total_coins" FROM fra_daily', con=engine)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_payment_data():
    df_pay = pd.read_sql(
        'SELECT date, SUM(amount) AS amount FROM payment GROUP BY date',
        con=engine
    )
    df_pay["date"] = pd.to_datetime(df_pay["date"])
    return df_pay

# 데이터 로딩
coin_df = load_coin_data()
pay_df = load_payment_data()

st.title("📊 웹툰 매출 & 결제 분석 대시보드 + 이벤트 인사이트")

# 콘텐츠 선택 (전체/개별)
options = ["전체 콘텐츠"] + sorted(coin_df["Title"].unique())
selected_title = st.selectbox("🔍 콘텐츠 선택", options)

# 1) 코인 매출: 데이터 준비
if selected_title == "전체 콘텐츠":
    df_coin = coin_df.groupby("date")["Total_coins"].sum().reset_index()
else:
    df_coin = coin_df[coin_df["Title"] == selected_title][["date", "Total_coins"]]

df_coin = df_coin.groupby("date")["Total_coins"].sum().reset_index()
df_coin = df_coin.sort_values("date")

# 코인 이벤트 임계치 입력 및 적용
if "coin_thresh" not in st.session_state:
    st.session_state.coin_thresh = 1.7
st.subheader("⚙️ 이벤트 임계치 설정 (코인)")
th_coin = st.number_input(
    "평균 대비 몇 % 이상일 때 코인 이벤트로 간주?", min_value=100, max_value=500,
    value=int(st.session_state.coin_thresh*100), key="coin_thresh_input"
)
if st.button("코인 임계치 적용"):
    st.session_state.coin_thresh = th_coin / 100
coin_threshold = st.session_state.coin_thresh
st.caption(f"현재 코인 이벤트 임계치: {int(coin_threshold*100)}%")

# 코인 이벤트 검출
df_coin["rolling_avg"] = df_coin["Total_coins"].rolling(window=7, center=True, min_periods=1).mean()
df_coin["event_flag"] = df_coin["Total_coins"] > df_coin["rolling_avg"] * coin_threshold
df_coin["weekday"] = df_coin["date"].dt.day_name()

# 코인 이벤트 발생 요일 분포
weekdays_order = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
coin_stats = df_coin[df_coin["event_flag"]]["weekday"].value_counts()
df_coin_ev = pd.DataFrame({
    'weekday': weekdays_order,
    'count': [coin_stats.get(day, 0) for day in weekdays_order]
})
df_coin_ev['negative'] = -df_coin_ev['count']
chart_coin = alt.Chart(df_coin_ev).mark_bar(color='red').encode(
    x=alt.X('weekday:N', sort=weekdays_order, title='요일'),
    y=alt.Y('negative:Q', axis=alt.Axis(title='이벤트 횟수'), scale=alt.Scale(domain=[-max(df_coin_ev['count'])-1, 0]))
).properties(height=250)
st.subheader("🌟 코인 이벤트 발생 요일 분포")
st.altair_chart(chart_coin, use_container_width=True)

# 최근 3개월 코인 매출 추이
recent_coin = df_coin[df_coin["date"] >= df_coin["date"].max() - timedelta(days=90)]
label_coin = "전체 콘텐츠" if selected_title == "전체 콘텐츠" else selected_title
st.subheader(f"📈 '{label_coin}' 최근 3개월 코인 매출 추이")
st.line_chart(recent_coin.set_index("date")["Total_coins"])

# 코인 예측
prophet_coin = df_coin.rename(columns={"date":"ds","Total_coins":"y"})
model_coin = Prophet()
model_coin.add_country_holidays(country_name="FR")
model_coin.fit(prophet_coin)
future_coin = model_coin.make_future_dataframe(periods=7)
forecast_coin = model_coin.predict(future_coin)
coin_fut7 = forecast_coin[forecast_coin["ds"] > df_coin["date"].max()]
st.subheader("🔮 코인 향후 7일 예측")
st.line_chart(coin_fut7.set_index("ds")["yhat"])

# 2) 결제 매출 섹션
st.header("💳 결제 매출 분석")

# 결제 이벤트 임계치 입력 및 적용
if "pay_thresh" not in st.session_state:
    st.session_state.pay_thresh = 1.7
st.subheader("⚙️ 이벤트 임계치 설정 (결제)")
th_pay = st.number_input(
    "평균 대비 몇 % 이상일 때 결제 이벤트로 간주?", min_value=100, max_value=500,
    value=int(st.session_state.pay_thresh*100), key="pay_thresh_input"
)
if st.button("결제 임계치 적용"):
    st.session_state.pay_thresh = th_pay / 100
pay_threshold = st.session_state.pay_thresh
st.caption(f"현재 결제 이벤트 임계치: {int(pay_threshold*100)}%")

# 결제 이벤트 검출
pay_df_sorted = pay_df.sort_values("date").reset_index(drop=True)
pay_df_sorted["rolling_avg"] = pay_df_sorted["amount"].rolling(window=7, center=True, min_periods=1).mean()
pay_df_sorted["event_flag"] = pay_df_sorted["amount"] > pay_df_sorted["rolling_avg"] * pay_threshold
pay_df_sorted["weekday"] = pay_df_sorted["date"].dt.day_name()

# 결제 이벤트 발생 요일 분포
pay_stats = pay_df_sorted[pay_df_sorted["event_flag"]]["weekday"].value_counts()
df_pay_ev = pd.DataFrame({
    'weekday': weekdays_order,
    'count': [pay_stats.get(day, 0) for day in weekdays_order]
})
df_pay_ev['negative'] = -df_pay_ev['count']
chart_pay = alt.Chart(df_pay_ev).mark_bar(color='blue').encode(
    x=alt.X('weekday:N', sort=weekdays_order, title='요일'),
    y=alt.Y('negative:Q', axis=alt.Axis(title='이벤트 횟수'), scale=alt.Scale(domain=[-max(df_pay_ev['count'])-1, 0]))
).properties(height=250)
st.subheader("🌟 결제 이벤트 발생 요일 분포")
st.altair_chart(chart_pay, use_container_width=True)

# 최근 3개월 결제 매출 추이
recent_pay = pay_df[pay_df["date"] >= pay_df["date"].max() - timedelta(days=90)]
st.subheader("📈 결제 매출 최근 3개월 추이")
st.line_chart(recent_pay.set_index("date")["amount"])

# 결제 예측
prophet_pay = pay_df.rename(columns={"date":"ds","amount":"y"})
model_pay = Prophet()
model_pay.add_country_holidays(country_name="FR")
model_pay.fit(prophet_pay)
pay_future = model_pay.make_future_dataframe(periods=7)
pay_forecast = model_pay.predict(pay_future)
pay_fut7 = pay_forecast[pay_forecast["ds"] > pay_df["date"].max()]
st.subheader("🔮 결제 매출 향후 7일 예측")
st.line_chart(pay_fut7.set_index("ds")["yhat"])

# 3) 이벤트 예정일 체크 및 적용
st.subheader("🗓 이벤트 예정일 체크 및 적용")
event_input = st.date_input(
    "이벤트 가능성 있는 날짜 선택", value=None, format="YYYY-MM-DD", key="event_input"
)
apply = st.button("이벤트 적용")
if apply and event_input:
    sel = event_input
    wd = sel.strftime("%A")
    # 코인 비율
    total_coin_days = df_coin[df_coin['weekday'] == wd].shape[0]
    coin_rate = coin_stats.get(wd, 0) / total_coin_days if total_coin_days > 0 else 0
    # 결제 비율
    total_pay_days = pay_df_sorted[pay_df_sorted['weekday'] == wd].shape[0]
    pay_rate = pay_stats.get(wd, 0) / total_pay_days if total_pay_days > 0 else 0
    st.write(f"📈 과거 {wd} 이벤트 비율 - 코인: {coin_rate:.1%}, 결제: {pay_rate:.1%}")
    # 예측 기간 포함 여부
    in_coin = sel in coin_fut7['ds'].dt.date.tolist()
    in_pay = sel in pay_fut7['ds'].dt.date.tolist()
    if in_coin:
        st.success(f"🚀 {sel}은 코인 예측 기간에 포함됩니다.")
    if in_pay:
        st.success(f"🚀 {sel}은 결제 예측 기간에 포함됩니다.")
    if not in_coin and not in_pay:
        st.warning("⚠️ 선택한 날짜가 예측 기간에 포함되지 않습니다.")
elif apply:
    st.warning("⚠️ 먼저 이벤트 날짜를 선택해주세요.")
