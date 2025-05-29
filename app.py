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
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}?sslmode=require")

# 데이터 로드 함수
@st.cache_data
def load_coin_data():
    df = pd.read_sql('SELECT date, "Title", "Total_coins" FROM fra_daily', con=engine)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_payment_data():
    query = '''
    SELECT date,
           SUM(amount) AS amount,
           SUM(CASE WHEN count = 1 THEN 1 ELSE 0 END) AS first_count
    FROM payment
    GROUP BY date
    '''
    df = pd.read_sql(query, con=engine)
    df["date"] = pd.to_datetime(df["date"])
    return df

# 데이터 로딩
coin_df = load_coin_data()
pay_df = load_payment_data()

st.title("📊 웹툰 매출 & 결제 분석 대시보드 + 이벤트 인사이트")

# 요일 순서 정의
weekdays = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

# -- 결제 매출 분석 --
st.header("💳 결제 매출 분석")

# 1) 임계치 설정 (결제)
if "pay_thresh" not in st.session_state:
    st.session_state.pay_thresh = 1.5
st.subheader("⚙️ 이벤트 임계치 설정 (결제)")
th_pay = st.number_input(
    "평균 대비 몇 % 이상일 때 결제 이벤트로 간주?",
    min_value=100, max_value=500,
    value=int(st.session_state.pay_thresh*100), key="pay_thresh_input", step=5
)
if st.button("결제 임계치 적용", key="btn_pay_thresh"):
    st.session_state.pay_thresh = th_pay / 100
pay_thresh = st.session_state.pay_thresh
st.caption(f"현재 결제 이벤트 임계치: {int(pay_thresh*100)}%")

# 2) 결제 데이터 준비 및 이벤트 검출
df_pay = pay_df.sort_values("date").reset_index(drop=True)
df_pay['rolling_avg'] = df_pay['amount'].rolling(window=7, center=True, min_periods=1).mean()
df_pay['event_flag'] = df_pay['amount'] > df_pay['rolling_avg'] * pay_thresh
df_pay['weekday'] = df_pay['date'].dt.day_name()
pay_counts = df_pay[df_pay['event_flag']]['weekday'].value_counts()

# 3) 이벤트 발생 요일 분포
st.subheader("🌟 결제 이벤트 발생 요일 분포")
df_pay_ev = pd.DataFrame({ 'weekday': weekdays, 'count': [pay_counts.get(d,0) for d in weekdays] })
chart1 = alt.Chart(df_pay_ev).mark_bar(color='blue').encode(
    x=alt.X('weekday:N', sort=weekdays, title='요일'),
    y=alt.Y('count:Q', title='이벤트 횟수'),
    tooltip=['weekday','count']
).properties(height=250)
st.altair_chart(chart1, use_container_width=True)

# 4) 요일별 평균 이벤트 증가율
st.subheader("💹 결제 이벤트 발생 시 요일별 평균 증가율")
rates = []
for d in weekdays:
    sub = df_pay[(df_pay['weekday']==d)&(df_pay['event_flag'])]
    rates.append((sub['amount']/sub['rolling_avg']).mean() if not sub.empty else 0)
df_pay_ev['rate'] = rates
chart2 = alt.Chart(df_pay_ev).mark_bar(color='cyan').encode(
    x=alt.X('weekday:N', sort=weekdays, title='요일'),
    y=alt.Y('rate:Q', title='평균 증가 배수'),
    tooltip=['weekday','rate']
).properties(height=250)
st.altair_chart(chart2, use_container_width=True)

# 5) 최근 3개월 추이
st.subheader("📈 결제 매출 최근 3개월 추이")
recent_pay = df_pay[df_pay['date'] >= df_pay['date'].max() - timedelta(days=90)]
st.line_chart(recent_pay.set_index('date')['amount'])

# 6) 결제 매출 향후 15일 예측 (시나리오 포함)
st.subheader("🔮 결제 매출 향후 15일 예측")
# Prophet 적합 및 예측
prophet_pay = df_pay.rename(columns={'date':'ds','amount':'y'})
model_pay = Prophet()
model_pay.add_country_holidays(country_name='DE')
model_pay.fit(prophet_pay)
future_pay = model_pay.make_future_dataframe(periods=15)
forecast = model_pay.predict(future_pay)
pay_fut15 = forecast[forecast['ds'] > df_pay['date'].max()].copy()
# 평일 이벤트 비율 매핑
pay_rate_map = dict(zip(df_pay_ev['weekday'], df_pay_ev['rate']))
pay_fut15['weekday'] = pay_fut15['ds'].dt.day_name()
# 시나리오: yhat_adj = yhat * (1 + rate)
pay_fut15['yhat_event'] = pay_fut15.apply(lambda r: r['yhat'] * (1 + pay_rate_map.get(r['weekday'], 0)), axis=1)

# 기본 예측선
base = alt.Chart(pay_fut15).mark_line(color='steelblue').encode(
    x=alt.X('ds:T', title='날짜'),
    y=alt.Y('yhat:Q', title='예측 결제 매출'),
    tooltip=[alt.Tooltip('ds:T', title='날짜'), alt.Tooltip('yhat:Q', title='기본 예측')]
)
# 이벤트 시나리오 예측선
scenario = alt.Chart(pay_fut15).mark_line(color='red').encode(
    x='ds:T',
    y=alt.Y('yhat_event:Q', title='이벤트 시나리오 예측'),
    tooltip=[alt.Tooltip('ds:T', title='날짜'), alt.Tooltip('yhat_event:Q', title='시나리오 예측')]
)
chart = (base + scenario).properties(height=300).interactive()
st.altair_chart(chart, use_container_width=True)

# 7) 이벤트 예정일 체크 및 적용 (결제)
st.subheader("🗓 결제 이벤트 예정일 체크 및 적용")
evt_date = st.date_input("이벤트 가능성 있는 결제 날짜 선택", key="pay_evt")
if st.button("결제 이벤트 적용", key="btn_evt_apply"):
    if evt_date:
        wd = evt_date.strftime('%A')
        total = df_pay[df_pay['weekday']==wd].shape[0]
        cnt = pay_counts.get(wd,0)
        st.write(f"📈 과거 {wd} 결제 이벤트 비율: {cnt/total:.1%}" if total>0 else "데이터 부족")
    else:
        st.warning("⚠️ 날짜 선택 필요")

# 8) 첫 결제 추이
st.subheader("🚀 첫 결제 추이")
st.line_chart(df_pay.set_index('date')['first_count'])

# -- 코인 매출 분석 --
st.header("🪙 코인 매출 분석")
# 콘텐츠 선택
options = ["전체 콘텐츠"] + sorted(coin_df['Title'].unique())
selected = st.selectbox("🔍 콘텐츠 선택", options)

# 1) 코인 임계치 설정
if "coin_thresh" not in st.session_state:
    st.session_state.coin_thresh = 1.2
st.subheader("⚙️ 이벤트 임계치 설정 (코인)")
th_coin = st.number_input(
    "평균 대비 몇 % 이상일 때 코인 이벤트로 간주?",
    min_value=100, max_value=500,
    value=int(st.session_state.coin_thresh*100), key="coin_thresh_input", step=5
)
if st.button("코인 임계치 적용", key="btn_coin_thresh"):
    st.session_state.coin_thresh = th_coin / 100
coin_thresh = st.session_state.coin_thresh
st.caption(f"현재 코인 이벤트 임계치: {int(coin_thresh*100)}%")

# 2) 코인 데이터 준비 및 이벤트 검출
def get_coin_df():
    if selected=="전체 콘텐츠":
        df = coin_df.groupby('date')['Total_coins'].sum().reset_index()
    else:
        df = coin_df[coin_df['Title']==selected][['date','Total_coins']]
    return df.sort_values('date')

df_coin_sel = get_coin_df()
df_coin_sel['rolling_avg'] = df_coin_sel['Total_coins'].rolling(window=7, center=True, min_periods=1).mean()
df_coin_sel['event_flag'] = df_coin_sel['Total_coins'] > df_coin_sel['rolling_avg'] * coin_thresh
df_coin_sel['weekday'] = df_coin_sel['date'].dt.day_name()
coin_counts = df_coin_sel[df_coin_sel['event_flag']]['weekday'].value_counts()

# 3) 이벤트 발생 요일 분포
st.subheader("🌟 코인 이벤트 발생 요일 분포")
df_coin_ev = pd.DataFrame({'weekday': weekdays, 'count':[coin_counts.get(d,0) for d in weekdays]})
df_coin_ev['negative'] = -df_coin_ev['count']
chart_coin = alt.Chart(df_coin_ev).mark_bar(color='red').encode(
   x=alt.X('weekday:N', sort=weekdays, title='요일'),
   y=alt.Y('negative:Q', title='이벤트 횟수'),
   tooltip=['weekday','count']
).properties(height=250)
st.altair_chart(chart_coin, use_container_width=True)

# 4) 요일별 평균 증가율
st.subheader("💹 코인 이벤트 발생 시 요일별 평균 증가율")
rates2=[]
for d in weekdays:
    sub=df_coin_sel[(df_coin_sel['weekday']==d)&(df_coin_sel['event_flag'])]
    rates2.append((sub['Total_coins']/sub['rolling_avg']).mean() if not sub.empty else 0)
df_coin_ev['rate']=rates2
chart_coin_rate = alt.Chart(df_coin_ev).mark_bar(color='orange').encode(
    x=alt.X('weekday:N', sort=weekdays, title='요일'),
    y=alt.Y('rate:Q', title='평균 증가 배수'),
    tooltip=['weekday','rate']
).properties(height=250)
st.altair_chart(chart_coin_rate, use_container_width=True)

# 5) 최근 3개월 매출 추이
st.subheader(f"📈 '{selected}' 최근 3개월 코인 매출 추이")
recent_coin = df_coin_sel[df_coin_sel['date']>=df_coin_sel['date'].max()-timedelta(days=90)]
st.line_chart(recent_coin.set_index('date')['Total_coins'])

# 6) 코인 매출 향후 15일 예측
st.subheader("🔮 코인 매출 향후 15일 예측")
prophet_coin = df_coin_sel.rename(columns={'date':'ds','Total_coins':'y'})
model_coin = Prophet()
model_coin.add_country_holidays(country_name='DE')
model_coin.fit(prophet_coin)
future_coin = model_coin.make_future_dataframe(periods=15)
forecast_coin = model_coin.predict(future_coin)
coin_fut15 = forecast_coin[forecast_coin['ds'] > df_coin_sel['date'].max()].copy()

# 시나리오: 요일별 이벤트율을 곱한 예측
df_coin_ev_map = dict(zip(df_coin_ev['weekday'], df_coin_ev['rate']))
coin_fut15['weekday'] = coin_fut15['ds'].dt.day_name()
coin_fut15['yhat_event'] = coin_fut15.apply(
    lambda r: r['yhat']*(1 + df_coin_ev_map.get(r['weekday'],0)), axis=1
)
base_c = alt.Chart(coin_fut15).mark_line(color='steelblue').encode(
    x='ds:T', y='yhat:Q', tooltip=['ds','yhat']
)
evt_c = alt.Chart(coin_fut15).mark_line(color='red').encode(
    x='ds:T', y='yhat_event:Q', tooltip=['ds','yhat_event']
)
st.altair_chart((base_c+evt_c).properties(height=300).interactive(), use_container_width=True)
