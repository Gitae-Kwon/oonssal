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
    df = pd.read_sql('SELECT date, SUM(amount) AS amount FROM payment GROUP BY date', con=engine)
    df["date"] = pd.to_datetime(df["date"])
    return df

# 데이터 로딩
df_coin_raw = load_coin_data()
df_pay_raw = load_payment_data()

st.title("📊 웹툰 매출 & 결제 분석 대시보드 + 이벤트 인사이트")

# 요일 순서 정의
weekdays_order = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

# -- 결제 매출 분석 섹션 --
st.header("💳 결제 매출 분석")
# 임계치 설정 (결제)
if "pay_thresh" not in st.session_state:
    st.session_state.pay_thresh = 1.7
st.subheader("⚙️ 이벤트 임계치 설정 (결제)")
th_pay = st.number_input(
    "평균 대비 몇 % 이상일 때 결제 이벤트로 간주?", min_value=100, max_value=500,
    value=int(st.session_state.pay_thresh*100), key="pay_thresh_input", step=5
)
if st.button("결제 임계치 적용"):
    st.session_state.pay_thresh = th_pay / 100
pay_threshold = st.session_state.pay_thresh
st.caption(f"현재 결제 이벤트 임계치: {int(pay_threshold*100)}%")

# 데이터 준비 및 이벤트 검출
df_pay = df_pay_raw.sort_values("date").reset_index(drop=True)
df_pay['rolling_avg'] = df_pay['amount'].rolling(window=7, center=True, min_periods=1).mean()
df_pay['event_flag'] = df_pay['amount'] > df_pay['rolling_avg'] * pay_threshold
df_pay['weekday'] = df_pay['date'].dt.day_name()

# ① 이벤트 발생 요일 분포
pay_counts = df_pay[df_pay['event_flag']]['weekday'].value_counts()
df_pay_ev = pd.DataFrame({
    'weekday': weekdays_order,
    'count': [pay_counts.get(d,0) for d in weekdays_order]
})
st.subheader("🌟 결제 이벤트 발생 요일 분포")
chart_pay = alt.Chart(df_pay_ev).mark_bar(color='blue').encode(
    x=alt.X('weekday:N', sort=weekdays_order, title='요일'),
    y=alt.Y('count:Q', title='이벤트 횟수', scale=alt.Scale(domain=[0, df_pay_ev['count'].max()+1]))
).properties(height=250)
st.altair_chart(chart_pay, use_container_width=True)

# ② 요일별 평균 이벤트 증가율
df_pay_ev['rate'] = df_pay_ev['weekday'].apply(lambda wd: 
    (df_pay[df_pay['weekday']==wd & df_pay['event_flag']]['amount'] / df_pay[df_pay['weekday']==wd & df_pay['event_flag']]['rolling_avg']).mean() if wd in pay_counts.index else 0)
)
st.subheader("💹 결제 이벤트 발생 시 요일별 평균 증가율")
chart_pay_rate = alt.Chart(df_pay_ev).mark_bar(color='cyan').encode(
    x=alt.X('weekday:N', sort=weekdays_order, title='요일'),
    y=alt.Y('rate:Q', title='평균 증가 배수', scale=alt.Scale(domain=[0, df_pay_ev['rate'].max()*1.1]))
).properties(height=250)
st.altair_chart(chart_pay_rate, use_container_width=True)

# ③ 최근 3개월 추이
recent_pay = df_pay[df_pay['date'] >= df_pay['date'].max() - timedelta(days=90)]
st.subheader("📈 결제 매출 최근 3개월 추이")
st.line_chart(recent_pay.set_index('date')['amount'])

# ④ 예측
prophet_pay = df_pay_raw.rename(columns={'date':'ds','amount':'y'})
model_pay = Prophet()
model_pay.add_country_holidays(country_name='FR')
model_pay.fit(prophet_pay)
pay_future = model_pay.make_future_dataframe(periods=7)
pay_forecast = model_pay.predict(pay_future)
pay_fut7 = pay_forecast[pay_forecast['ds'] > df_pay_raw['date'].max()]
st.subheader("🔮 결제 매출 향후 7일 예측")
st.line_chart(pay_fut7.set_index('ds')['yhat'])

# -- 코인 매출 분석 섹션 --
st.header("🪙 코인 매출 분석")
# 콘텐츠 선택 (전체/개별)
options = ["전체 콘텐츠"] + sorted(df_coin_raw['Title'].unique())
selected_title = st.selectbox("🔍 콘텐츠 선택", options)

# 임계치 설정 (코인)
if "coin_thresh" not in st.session_state:
    st.session_state.coin_thresh = 1.7
st.subheader("⚙️ 이벤트 임계치 설정 (코인)")
th_coin = st.number_input(
    "평균 대비 몇 % 이상일 때 코인 이벤트로 간주?", min_value=100, max_value=500,
    value=int(st.session_state.coin_thresh*100), key="coin_thresh_input", step=5
)
if st.button("코인 임계치 적용"):
    st.session_state.coin_thresh = th_coin / 100
coin_threshold = st.session_state.coin_thresh
st.caption(f"현재 코인 이벤트 임계치: {int(coin_threshold*100)}%")

# 데이터 준비 및 이벤트 검출
def get_coin_df():
    if selected_title == '전체 콘텐츠':
        df = df_coin_raw.groupby('date')['Total_coins'].sum().reset_index()
    else:
        df = df_coin_raw[df_coin_raw['Title']==selected_title][['date','Total_coins']]
    return df.sort_values('date')
df_coin = get_coin_df()
df_coin['rolling_avg'] = df_coin['Total_coins'].rolling(window=7, center=True, min_periods=1).mean()
df_coin['event_flag'] = df_coin['Total_coins'] > df_coin['rolling_avg'] * coin_threshold
df_coin['weekday'] = df_coin['date'].dt.day_name()

# ① 이벤트 발생 요일 분포 (음수)
coin_counts = df_coin[df_coin['event_flag']]['weekday'].value_counts()
df_coin_ev = pd.DataFrame({
    'weekday': weekdays_order,
    'count': [coin_counts.get(d,0) for d in weekdays_order]
})
df_coin_ev['negative'] = -df_coin_ev['count']
st.subheader("🌟 코인 이벤트 발생 요일 분포")
chart_coin = alt.Chart(df_coin_ev).mark_bar(color='red').encode(
    x=alt.X('weekday:N', sort=weekdays_order, title='요일'),
    y=alt.Y('negative:Q', title='이벤트 횟수', scale=alt.Scale(domain=[-df_coin_ev['count'].max()-1,0]))
).properties(height=250)
st.altair_chart(chart_coin, use_container_width=True)

# ② 요일별 평균 이벤트 증가율
df_coin_ev['rate'] = df_coin_ev['weekday'].apply(lambda wd:
    (df_coin[df_coin['weekday']==wd & df_coin['event_flag']]['Total_coins'] / df_coin[df_coin['weekday']==wd & df_coin['event_flag']]['rolling_avg']).mean() if wd in coin_counts.index else 0)
)
st.subheader("💹 코인 이벤트 발생 시 요일별 평균 증가율")
chart_coin_rate = alt.Chart(df_coin_ev).mark_bar(color='orange').encode(
    x=alt.X('weekday:N', sort=weekdays_order, title='요일'),
    y=alt.Y('rate:Q', title='평균 증가 배수', scale=alt.Scale(domain=[0, df_coin_ev['rate'].max()*1.1]))
).properties(height=250)
st.altair_chart(chart_coin_rate, use_container_width=True)

# ③ 최근 3개월 매출 추이
recent_coin = df_coin[df_coin['date'] >= df_coin['date'].max() - timedelta(days=90)]
st.subheader(f"📈 '{selected_title}' 최근 3개월 코인 매출 추이")
st.line_chart(recent_coin.set_index('date')['Total_coins'])

# ④ 예측
prophet_coin = df_coin.rename(columns={'date':'ds','Total_coins':'y'})
model_coin = Prophet()
model_coin.add_country_holidays(country_name='FR')
model_coin.fit(prophet_coin)
future_coin = model_coin.make_future_dataframe(periods=7)
forecast_coin = model_coin.predict(future_coin)
coin_fut7 = forecast_coin[forecast_coin['ds'] > df_coin['date'].max()]
st.subheader("🔮 코인 매출 향후 7일 예측")
st.line_chart(coin_fut7.set_index('ds')['yhat'])

# 3) 이벤트 예정일 체크 및 적용
st.subheader("🗓 이벤트 예정일 체크 및 적용")
event_input = st.date_input("이벤트 가능성 있는 날짜 선택", value=None, format="YYYY-MM-DD", key="event_input")
apply = st.button("이벤트 적용")
if apply and event_input:
    sel = event_input
    wd = sel.strftime("%A")
    pay_total = df_pay[df_pay['weekday']==wd].shape[0]
    pay_cnt = pay_counts.get(wd,0)
    pay_rate = pay_cnt / pay_total if pay_total>0 else 0
    coin_total = df_coin[df_coin['weekday']==wd].shape[0]
    coin_cnt = coin_counts.get(wd,0)
    coin_rate = coin_cnt / coin_total if coin_total>0 else 0
    st.write(f"📈 과거 {wd} 이벤트 비율 - 결제: {pay_rate:.1%}, 코인: {coin_rate:.1%}")
    in_pay = sel in pay_fut7['ds'].dt.date.tolist()
    in_coin = sel in coin_fut7['ds'].dt.date.tolist()
    if in_pay: st.success(f"🚀 {sel}은 결제 예측 기간에 포함됩니다.")
    if in_coin: st.success(f"🚀 {sel}은 코인 예측 기간에 포함됩니다.")
    if not in_pay and not in_coin: st.warning("⚠️ 선택한 날짜가 예측 기간에 포함되지 않습니다.")
elif apply:
    st.warning("⚠️ 먼저 이벤트 날짜를 선택해주세요.")
