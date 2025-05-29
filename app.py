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

# 데이터 로드 함수
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

# 데이터 불러오기
coin_df = load_coin_data()
pay_df = load_payment_data()

st.title("📊 웹툰 매출 & 결제 분석 대시보드 + 이벤트 인사이트")

weekdays = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

# -- 결제 매출 분석 --
st.header("💳 결제 매출 분석")

# 1) 결제 임계치 설정\if "pay_thresh" not in st.session_state:
    st.session_state.pay_thresh = 1.5
st.subheader("⚙️ 이벤트 임계치 설정 (결제)")
th_pay = st.number_input(
    "평균 대비 몇 % 이상일 때 결제 이벤트로 간주?",
    min_value=100, max_value=500,
    value=int(st.session_state.pay_thresh*100), key="pay_thresh_input", step=5
)
if st.button("결제 임계치 적용", key="btn_pay_thresh"):
    st.session_state.pay_thresh = th_pay / 100
st.caption(f"현재 결제 이벤트 임계치: {int(st.session_state.pay_thresh*100)}%")

# 2) 이벤트 검출
DF = pay_df.sort_values("date").reset_index(drop=True)
DF['rolling_avg'] = DF['amount'].rolling(window=7, center=True, min_periods=1).mean()
DF['event_flag'] = DF['amount'] > DF['rolling_avg'] * st.session_state.pay_thresh
DF['weekday'] = DF['date'].dt.day_name()
pay_counts = DF[DF['event_flag']]['weekday'].value_counts()

# 3) 이벤트 요일 분포
st.subheader("🌟 결제 이벤트 발생 요일 분포")
df_ev = pd.DataFrame({ 'weekday': weekdays, 'count': [pay_counts.get(d,0) for d in weekdays] })
chart_ev = alt.Chart(df_ev).mark_bar(color='blue').encode(
    x=alt.X('weekday:N', sort=weekdays, title='요일'),
    y=alt.Y('count:Q', title='이벤트 횟수'),
    tooltip=['weekday','count']
).properties(height=250)
st.altair_chart(chart_ev, use_container_width=True)

# 4) 요일별 평균 증가율
st.subheader("💹 결제 이벤트 발생 시 요일별 평균 증가율")
rates = [
    (DF[(DF['weekday']==d)&DF['event_flag']]['amount']/DF[(DF['weekday']==d)&DF['event_flag']]['rolling_avg']).mean()
    if not DF[(DF['weekday']==d)&DF['event_flag']].empty else 0 for d in weekdays
]
df_ev['rate'] = rates
chart_rate = alt.Chart(df_ev).mark_bar(color='cyan').encode(
    x=alt.X('weekday:N', sort=weekdays, title='요일'),
    y=alt.Y('rate:Q', title='평균 증가 배수'),
    tooltip=['weekday','rate']
).properties(height=250)
st.altair_chart(chart_rate, use_container_width=True)

# 5) 최근 3개월 추이
st.subheader("📈 결제 매출 최근 3개월 추이")
recent = DF[DF['date']>=DF['date'].max()-timedelta(days=90)]
st.line_chart(recent.set_index('date')['amount'])

# 6) 결제 매출 향후 15일 예측 (시나리오)
st.subheader("🔮 결제 매출 향후 15일 예측")
prophet_df = DF.rename(columns={'date':'ds','amount':'y'})
model = Prophet(); model.add_country_holidays(country_name='DE'); model.fit(prophet_df)
future = model.make_future_dataframe(periods=15)
forecast = model.predict(future)
pay_fut = forecast[forecast['ds']>DF['date'].max()].copy()
# 이벤트 시나리오
rate_map = dict(zip(df_ev['weekday'], df_ev['rate']))
pay_fut['weekday'] = pay_fut['ds'].dt.day_name()
pay_fut['yhat_event'] = pay_fut['yhat'] * (1 + pay_fut['weekday'].map(rate_map).fillna(0))
# 플롯 함수
def plot_pay(apply=False):
    base = alt.Chart(pay_fut).mark_line(color='steelblue').encode(x='ds:T', y='yhat:Q')
    if apply:
        evt = alt.Chart(pay_fut).mark_line(color='red').encode(x='ds:T', y='yhat_event:Q')
        return (base + evt).properties(height=300).interactive()
    return base.properties(height=300).interactive()
# 시나리오 적용
if 'apply_event' not in st.session_state:
    st.session_state.apply_event = False
if st.button('시나리오 적용', key='btn_apply'):
    st.session_state.apply_event = True
st.altair_chart(plot_pay(st.session_state.apply_event), use_container_width=True)

# 7) 이벤트 예정일 체크
st.subheader("🗓 결제 이벤트 예정일 체크 및 적용")
evt = st.date_input('이벤트 날짜 선택', key='pay_evt')
if st.button('결제 이벤트 적용', key='btn_evt'):
    if evt:
        wd=evt.strftime('%A'); total=DF[DF['weekday']==wd].shape[0]; cnt=pay_counts.get(wd,0)
        st.write(f"📈 과거 {wd} 이벤트 비율: {cnt/total:.1%}" if total>0 else '데이터 부족')
    else:
        st.warning('⚠️ 날짜 선택 필요')

# 8) 첫 결제 추이
st.subheader('🚀 첫 결제 추이')
st.line_chart(DF.set_index('date')['first_count'])

# -- 코인 매출 분석 --
st.header('🪙 코인 매출 분석')
options = ['전체 콘텐츠'] + sorted(coin_df['Title'])
selected = st.selectbox('🔍 콘텐츠 선택', options)
# 코인 임계치
if 'coin_thresh' not in st.session_state:
    st.session_state.coin_thresh = 1.2
st.subheader('⚙️ 이벤트 임계치 설정 (코인)')
th_coin = st.number_input('평균 대비 %(코인)',100,500,value=int(st.session_state.coin_thresh*100),step=5,key='coin_in')
if st.button('코인 임계치 적용',key='btn_coin'):
    st.session_state.coin_thresh = th_coin/100
st.caption(f"현재 코인 이벤트 임계치: {int(st.session_state.coin_thresh*100)}%")
# 데이터 준비
def get_coin_df():
    if selected=='전체 콘텐츠': return coin_df.groupby('date')['Total_coins'].sum().reset_index()
    return coin_df[coin_df['Title']==selected][['date','Total_coins']].reset_index(drop=True)

df_coin_sel = get_coin_df().sort_values('date')
df_coin_sel['rolling_avg'] = df_coin_sel['Total_coins'].rolling(window=7,center=True,min_periods=1).mean()
df_coin_sel['event_flag'] = df_coin_sel['Total_coins']>df_coin_sel['rolling_avg']*st.session_state.coin_thresh
df_coin_sel['weekday'] = df_coin_sel['date'].dt.day_name()
coin_counts = df_coin_sel[df_coin_sel['event_flag']]['weekday'].value_counts()

# 이벤트 분포
st.subheader('🌟 코인 이벤트 발생 요일 분포')
df_ce=pd.DataFrame({'weekday':weekdays,'count':[coin_counts.get(d,0) for d in weekdays]})
chart_ce=alt.Chart(df_ce).mark_bar(color='red').encode(x=alt.X('weekday:N',sort=weekdays),y='count:Q',tooltip=['weekday','count']).properties(height=250)
st.altair_chart(chart_ce,use_container_width=True)
# 평균 증가율
st.subheader('💹 코인 이벤트 발생 시 요일별 평균 증가율')
rates2=[(df_coin_sel[(df_coin_sel['weekday']==d)&df_coin_sel['event_flag']]['Total_coins']/df_coin_sel[(df_coin_sel['weekday']==d)&df_coin_sel['event_flag']]['rolling_avg']).mean() if not df_coin_sel[(df_coin_sel['weekday']==d)&df_coin_sel['event_flag']].empty else 0 for d in weekdays]
df_ce['rate']=rates2
chart_ce2=alt.Chart(df_ce).mark_bar(color='orange').encode(x=alt.X('weekday:N',sort=weekdays),y='rate:Q',tooltip=['weekday','rate']).properties(height=250)
st.altair_chart(chart_ce2,use_container_width=True)
# 추이
st.subheader(f"📈 '{selected}' 최근 3개월 코인 매출 추이")
st.line_chart(df_coin_sel[df_coin_sel['date']>=df_coin_sel['date'].max()-timedelta(days=90)].set_index('date')['Total_coins'])
# 예측
st.subheader('🔮 코인 매출 향후 15일 예측')
pc=df_coin_sel.rename(columns={'date':'ds','Total_coins':'y'})
mc=Prophet(); mc.add_country_holidays(country_name='DE'); mc.fit(pc)
fut_c=mc.predict(mc.make_future_dataframe(periods=15))
st.line_chart(fut_c[fut_c['ds']>df_coin_sel['date'].max()].set_index('ds')['yhat'])
