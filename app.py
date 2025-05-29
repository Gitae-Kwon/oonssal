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

# 데이터 로드
@st.cache_data
def load_coin_data():
    df = pd.read_sql('SELECT date, "Title", "Total_coins" FROM fra_daily', con=engine)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_payment_data():
    query = '''
    SELECT
      date,
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

# 요일 순서
weekdays = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

# -- 결제 매출 분석 --
st.header("💳 결제 매출 분석")
# 임계치 설정 (결제)
if "pay_thresh" not in st.session_state:
    st.session_state.pay_thresh = 1.7
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

# 결제 데이터 준비 및 이벤트 검출
df_pay = pay_df.sort_values("date").reset_index(drop=True)
df_pay['rolling_avg'] = df_pay['amount'].rolling(window=7, center=True, min_periods=1).mean()
df_pay['event_flag'] = df_pay['amount'] > df_pay['rolling_avg'] * pay_thresh
df_pay['weekday'] = df_pay['date'].dt.day_name()
pay_counts = df_pay[df_pay['event_flag']]['weekday'].value_counts()

# 1) 이벤트 발생 요일 분포
st.subheader("🌟 결제 이벤트 발생 요일 분포")
df_pay_ev = pd.DataFrame({
    'weekday': weekdays,
    'count': [pay_counts.get(d,0) for d in weekdays]
})
chart1 = alt.Chart(df_pay_ev).mark_bar(color='blue').encode(
    x=alt.X('weekday:N', sort=weekdays, title='요일'),
    y=alt.Y('count:Q', title='이벤트 횟수'),
    tooltip=['weekday', 'count']
).properties(height=250)
st.altair_chart(chart1, use_container_width=True)

# 2) 요일별 평균 이벤트 증가율
st.subheader("💹 결제 이벤트 발생 시 요일별 평균 증가율")
rates = []
for d in weekdays:
    sub = df_pay[(df_pay['weekday']==d)&(df_pay['event_flag'])]
    rates.append((sub['amount']/sub['rolling_avg']).mean() if not sub.empty else 0)
df_pay_ev['rate'] = rates
chart2 = alt.Chart(df_pay_ev).mark_bar(color='cyan').encode(
    x=alt.X('weekday:N', sort=weekdays, title='요일'),
    y=alt.Y('rate:Q', title='평균 증가 배수'),
    tooltip=['weekday', 'rate']
).properties(height=250)
st.altair_chart(chart2, use_container_width=True)

# 3) 최근 3개월 추이
st.subheader("📈 결제 매출 최근 3개월 추이")
recent_pay = df_pay[df_pay['date']>=df_pay['date'].max()-timedelta(days=90)]
st.line_chart(recent_pay.set_index('date')['amount'])

# 4) 이벤트 예정일 체크 및 적용
st.subheader("🗓 결제 이벤트 예정일 체크 및 적용")
evt_date = st.date_input("이벤트 가능성 있는 결제 날짜 선택", key="pay_evt")
if st.button("결제 이벤트 적용", key="btn_evt_apply"):
    if evt_date:
        wd = evt_date.strftime('%A')
        total = df_pay[df_pay['weekday']==wd].shape[0]
        cnt = pay_counts.get(wd,0)
        st.write(f"📈 과거 {wd} 결제 이벤트 비율: {cnt/total:.1%}" if total>0 else "데이터 부족")
        if evt_date in (df_pay['date'].dt.date.tolist()):
            st.success(f"🚀 {evt_date}은 결제 예측 기간에 포함됩니다.")
        else:
            st.warning("⚠️ 선택 날짜 미포함")
    else:
        st.warning("⚠️ 날짜 선택 필요")

# 5) 첫 결제 추이
st.subheader("🚀 첫 결제 추이")
st.line_chart(df_pay.set_index('date')['first_count'])

# 6) 결제 매출 예측
st.subheader("🔮 결제 매출 향후 7일 예측")
prop_df = df_pay.rename(columns={'date':'ds','amount':'y'})
model = Prophet()
model.add_country_holidays(country_name='FR')
model.fit(prop_df)
fut = model.make_future_dataframe(periods=7)
fc = model.predict(fut)
st.line_chart(fc.set_index('ds')['yhat'])

# -- 코인 매출 분석 --
st.header("🪙 코인 매출 분석")
# 콘텐츠 선택
options = ["전체 콘텐츠"] + sorted(coin_df['Title'].unique())
sel = st.selectbox("🔍 콘텐츠 선택", options)

# 임계치 설정 (코인)
if "coin_thresh" not in st.session_state:
    st.session_state.coin_thresh = 1.7
st.subheader("⚙️ 이벤트 임계치 설정 (코인)")
th_coin = st.number_input(
    "평균 대비 몇 % 이상일 때 코인 이벤트로 간주?",
    min_value=100, max_value=500,
    value=int(st.session_state.coin_thresh*100), key="coin_thresh_in", step=5
)
if st.button("코인 임계치 적용", key="btn_coin"):
    st.session_state.coin_thresh = th_coin/100
c_thresh = st.session_state.coin_thresh
st.caption(f"현재 코인 이벤트 임계치: {int(c_thresh*100)}%")

# 코인 데이터 필터 및 검출
def make_coin_df():
    df = coin_df if sel=="전체 콘텐츠" else coin_df[coin_df['Title']==sel]
    return df.groupby('date')['Total_coins'].sum().reset_index().sort_values('date')
coin_df_sel = make_coin_df()
coin_df_sel['rolling_avg'] = coin_df_sel['Total_coins'].rolling(window=7, center=True, min_periods=1).mean()
coin_df_sel['event_flag'] = coin_df_sel['Total_coins']>coin_df_sel['rolling_avg']*c_thresh
coin_df_sel['weekday'] = coin_df_sel['date'].dt.day_name()
coin_counts = coin_df_sel[coin_df_sel['event_flag']]['weekday'].value_counts()

# 1) 이벤트 발생 요일 분포
st.subheader("🌟 코인 이벤트 발생 요일 분포")
df_ce = pd.DataFrame({'weekday': weekdays, 'count':[coin_counts.get(d,0) for d in weekdays]})
df_ce['neg']=-df_ce['count']
chartc = alt.Chart(df_ce).mark_bar(color='red').encode(
    x=alt.X('weekday:N', sort=weekdays, title='요일'),
    y=alt.Y('neg:Q', title='이벤트 횟수'),
    tooltip=['weekday','count']
).properties(height=250)
st.altair_chart(chartc, use_container_width=True)

# 2) 요일별 평균 증가율
st.subheader("💹 코인 이벤트 발생 시 요일별 평균 증가율")
crates=[]
for d in weekdays:
    sub=coin_df_sel[(coin_df_sel['weekday']==d)&(coin_df_sel['event_flag'])]
    crates.append((sub['Total_coins']/sub['rolling_avg']).mean() if not sub.empty else 0)
df_ce['rate']=crates
chartcr = alt.Chart(df_ce).mark_bar(color='orange').encode(
    x=alt.X('weekday:N', sort=weekdays, title='요일'),
    y=alt.Y('rate:Q', title='평균 증가 배수'),
    tooltip=['weekday','rate']
).properties(height=250)
st.altair_chart(chartcr, use_container_width=True)

# 3) 최근 3개월 추이
st.subheader(f"📈 '{sel}' 최근 3개월 코인 매출 추이")
rc = coin_df_sel[coin_df_sel['date']>=coin_df_sel['date'].max()-timedelta(days=90)]
st.line_chart(rc.set_index('date')['Total_coins'])

# 4) 예측
st.subheader("🔮 코인 매출 향후 7일 예측")
dfpc=coin_df_sel.rename(columns={'date':'ds','Total_coins':'y'})
modc=Prophet()
modc.add_country_holidays(country_name='FR')
modc.fit(dfpc)
fcpc=modc.predict(modc.make_future_dataframe(periods=7))
st.line_chart(fcpc.set_index('ds')['yhat'])
