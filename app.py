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
user     = st.secrets["DB_USER"]
password = st.secrets["DB_PASSWORD"]
host     = st.secrets["DB_HOST"]
port     = st.secrets["DB_PORT"]
db       = st.secrets["DB_NAME"]
engine   = create_engine(
    f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}?sslmode=require"
)

# 데이터 로드 함수
@st.cache_data
def load_coin_data():
    df = pd.read_sql('SELECT date, "Title", "Total_coins" FROM fra_daily', con=engine)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_payment_data():
    # date별 amount, first_count 집계
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
pay_df  = load_payment_data()

st.title("📊 웹툰 매출 & 결제 분석 대시보드 + 이벤트 인사이트")
weekdays = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]


# -- 1. 결제 매출 분석 --
st.header("💳 결제 매출 분석")

# 1-1) 이벤트 임계치 설정
if "pay_thresh" not in st.session_state:
    st.session_state.pay_thresh = 1.5
st.subheader("⚙️ 이벤트 임계치 설정 (결제)")
th_pay = st.number_input(
    "평균 대비 몇 % 이상일 때 결제 이벤트로 간주?",
    min_value=100, max_value=500,
    value=int(st.session_state.pay_thresh*100),
    key="pay_thresh_input", step=5
)
if st.button("결제 임계치 적용", key="btn_pay_thresh"):
    st.session_state.pay_thresh = th_pay / 100
st.caption(f"현재 결제 이벤트 임계치: {int(st.session_state.pay_thresh*100)}%")

# 1-2) 데이터 준비 및 이벤트 검출
df_pay = pay_df.sort_values("date").reset_index(drop=True)
df_pay['rolling_avg'] = df_pay['amount'].rolling(window=7, center=True, min_periods=1).mean()
df_pay['event_flag']  = df_pay['amount'] > df_pay['rolling_avg'] * st.session_state.pay_thresh
df_pay['weekday']     = df_pay['date'].dt.day_name()
pay_counts = df_pay[df_pay['event_flag']]['weekday'].value_counts()

# 1-3) 이벤트 발생 요일 분포
st.subheader("🌟 결제 이벤트 발생 요일 분포")
df_ev = pd.DataFrame({
    'weekday': weekdays,
    'count':   [pay_counts.get(d, 0) for d in weekdays]
})
chart_ev = alt.Chart(df_ev).mark_bar(color='blue').encode(
    x=alt.X('weekday:N', sort=weekdays, title='요일'),
    y=alt.Y('count:Q',    title='이벤트 횟수'),
    tooltip=['weekday','count']
).properties(height=250)
st.altair_chart(chart_ev, use_container_width=True)

# 1-4) 요일별 평균 이벤트 증가율
st.subheader("💹 결제 이벤트 발생 시 요일별 평균 증가율")
rates = []
for d in weekdays:
    sub = df_pay[(df_pay['weekday']==d) & df_pay['event_flag']]
    rates.append((sub['amount']/sub['rolling_avg']).mean() if not sub.empty else 0)
df_ev['rate'] = rates
chart_rate = alt.Chart(df_ev).mark_bar(color='cyan').encode(
    x=alt.X('weekday:N', sort=weekdays, title='요일'),
    y=alt.Y('rate:Q',     title='평균 증가 배수'),
    tooltip=['weekday','rate']
).properties(height=250)
st.altair_chart(chart_rate, use_container_width=True)

# 1-5) 최근 3개월 매출 추이
st.subheader("📈 결제 매출 최근 3개월 추이")
recent_pay = df_pay[df_pay['date'] >= df_pay['date'].max() - timedelta(days=90)]
st.line_chart(recent_pay.set_index('date')['amount'])

# 1-6) 결제 매출 향후 15일 예측 (시나리오)
st.subheader("🔮 결제 매출 향후 15일 예측")
prophet_df = df_pay.rename(columns={'date':'ds','amount':'y'})
model_pay  = Prophet()
model_pay.add_country_holidays(country_name='DE')
model_pay.fit(prophet_df)
future     = model_pay.make_future_dataframe(periods=15)
forecast   = model_pay.predict(future)
pay_fut    = forecast[forecast['ds'] > df_pay['date'].max()].copy()
# 과거 이벤트율 매핑
rate_map   = dict(zip(df_ev['weekday'], df_ev['rate']))
pay_fut['weekday']    = pay_fut['ds'].dt.day_name()
pay_fut['yhat_event'] = pay_fut['yhat'] * (1 + pay_fut['weekday'].map(rate_map).fillna(0))

def plot_pay(apply_event=False):
    base = alt.Chart(pay_fut).mark_line(color='steelblue').encode(
        x=alt.X('ds:T', title='날짜'),
        y=alt.Y('yhat:Q', title='예측 결제 매출')
    )
    if apply_event:
        evt = alt.Chart(pay_fut).mark_line(color='red').encode(
            x='ds:T', y='yhat_event:Q'
        )
        return (base + evt).properties(height=300).interactive()
    return base.properties(height=300).interactive()

# 1-7) 시나리오 적용/해제
if 'apply_event' not in st.session_state:
    st.session_state.apply_event = False
c1, c2 = st.columns(2)
with c1:
    if st.button('시나리오 적용', key='btn_apply'):
        st.session_state.apply_event = True
with c2:
    if st.button('시나리오 해제', key='btn_reset'):
        st.session_state.apply_event = False
st.altair_chart(plot_pay(st.session_state.apply_event), use_container_width=True)

# 1-8) 이벤트 예정일 체크 및 적용
st.subheader("🗓 결제 이벤트 예정일 체크 및 적용")
evt_date = st.date_input("이벤트 날짜 선택", key='pay_evt')
if st.button('결제 이벤트 적용', key='btn_evt'):
    if evt_date:
        wd    = evt_date.strftime('%A')
        total = df_pay[df_pay['weekday']==wd].shape[0]
        cnt   = pay_counts.get(wd,0)
        st.write(f"📈 과거 {wd} 이벤트 비율: {cnt/total:.1%}" if total>0 else "데이터 부족")
    else:
        st.warning("⚠️ 날짜 선택 필요")

# 1-9) 첫 결제 추이 (최근 3개월)
st.subheader("🚀 첫 결제 추이 (최근 3개월)")
recent_first = df_pay[df_pay["date"] >= df_pay["date"].max() - timedelta(days=90)]

chart_first = (
    alt.Chart(recent_first)
       .mark_line(point=True)
       .encode(
           x=alt.X("date:T", title="날짜"),
           y=alt.Y("first_count:Q", title="첫 결제 건수")
       )
       .properties(height=300)
)
st.altair_chart(chart_first, use_container_width=True)

# -- 2. 코인 매출 분석 --
st.header("🪙 코인 매출 분석")

# -- 2-0) 분석 기간 설정 및 Top N 작품 --
coin_date_range = st.date_input("코인 분석 기간 설정", [], key="coin_date_range")

if len(coin_date_range) == 2:
    start_coin = pd.to_datetime(coin_date_range[0])
    end_coin   = pd.to_datetime(coin_date_range[1])

    # 기간 내 집계
    df_period = coin_df[
        (coin_df["date"] >= start_coin) &
        (coin_df["date"] <= end_coin)
    ]
    coin_sum = (
        df_period
        .groupby("Title")["Total_coins"]
        .sum()
        .sort_values(ascending=False)
    )

    # 최초 런칭일 구하기
    first_launch = coin_df.groupby("Title")["date"].min()

    # Top N + 더보기
    if "coin_top_n" not in st.session_state:
        st.session_state.coin_top_n = 10
    top_n = st.session_state.coin_top_n

    df_top = (
        coin_sum
        .head(top_n)
        .reset_index(name="Total_coins")
    )
    df_top.insert(0, "Rank", range(1, len(df_top) + 1))

    # 신작 여부
    df_top["launch_date"] = df_top["Title"].map(first_launch)
    df_top["is_new"]      = df_top["launch_date"] >= start_coin

    # 화면에 보일 컬럼만 선택
    df_display = df_top[["Rank","Title","Total_coins","launch_date"]].copy()
    df_display = df_display.rename(columns={"launch_date":"Launch Date"})

    # 신작 강조 함수
    def _highlight_new(row):
        return [
            "color: yellow" if (col == "Title" and df_top.loc[row.name, "is_new"]) else ""
            for col in df_display.columns
        ]

    # 스타일링: 
    # 1) Total_coins에 천 단위 콤마 적용 
    # 2) 전체 셀 중앙 정렬 
    styled = (
        df_display.style
                  .apply(_highlight_new, axis=1)
                  .format({"Total_coins": "{:,}"})
                  .set_table_styles([
                      {"selector": "th", "props": [("text-align", "center")]},
                      {"selector": "td", "props": [("text-align", "center")]}
                  ])
    )

html = styled.to_html(index=False, escape=False)

st.subheader(f"📋 Top {top_n} 작품 (코인 사용량)")
st.markdown(html, unsafe_allow_html=True)

    # 더보기 버튼
    if len(coin_sum) > top_n:
        if st.button("더보기", key="btn_coin_more"):
            st.session_state.coin_top_n += 10
            
# 2-1) 콘텐츠 선택
options = ["전체 콘텐츠"] + sorted(coin_df['Title'].unique())
selected = st.selectbox("🔍 콘텐츠 선택", options)

# 2-2) 이벤트 임계치 설정 (코인)
if 'coin_thresh' not in st.session_state:
    st.session_state.coin_thresh = 1.2
st.subheader("⚙️ 이벤트 임계치 설정 (코인)")
th_coin = st.number_input(
    "평균 대비 몇 % 이상일 때 코인 이벤트로 간주?",
    min_value=100, max_value=500,
    value=int(st.session_state.coin_thresh*100),
    key='coin_thresh_input', step=5
)
if st.button('코인 임계치 적용', key='btn_coin'):
    st.session_state.coin_thresh = th_coin / 100
st.caption(f"현재 코인 이벤트 임계치: {int(st.session_state.coin_thresh*100)}%")


# 2-3) 코인 데이터 준비 및 이벤트 검출
def get_coin_df():
    if selected == "전체 콘텐츠":
        return coin_df.groupby('date')['Total_coins'].sum().reset_index()
    return coin_df[coin_df['Title']==selected][['date','Total_coins']].reset_index(drop=True)

df_coin_sel = get_coin_df().sort_values('date')
df_coin_sel['rolling_avg']  = df_coin_sel['Total_coins'].rolling(window=7, center=True, min_periods=1).mean()
df_coin_sel['event_flag']   = df_coin_sel['Total_coins'] > df_coin_sel['rolling_avg'] * st.session_state.coin_thresh
df_coin_sel['weekday']      = df_coin_sel['date'].dt.day_name()
coin_counts = df_coin_sel[df_coin_sel['event_flag']]['weekday'].value_counts()


# 2-4) 이벤트 발생 요일 분포
st.subheader("🌟 코인 이벤트 발생 요일 분포")
df_ce = pd.DataFrame({
    'weekday': weekdays,
    'count':   [coin_counts.get(d,0) for d in weekdays]
})
chart_ce = alt.Chart(df_ce).mark_bar(color='red').encode(
    x=alt.X('weekday:N', sort=weekdays, title='요일'),
    y=alt.Y('count:Q',    title='이벤트 횟수'),
    tooltip=['weekday','count']
).properties(height=250)
st.altair_chart(chart_ce, use_container_width=True)

# 2-5) 요일별 평균 이벤트 증가율
st.subheader("💹 코인 이벤트 발생 시 요일별 평균 증가율")
rates2=[]
for d in weekdays:
    sub = df_coin_sel[(df_coin_sel['weekday']==d)&df_coin_sel['event_flag']]
    rates2.append((sub['Total_coins']/sub['rolling_avg']).mean() if not sub.empty else 0)
df_ce['rate'] = rates2
chart_ce2 = alt.Chart(df_ce).mark_bar(color='orange').encode(
    x=alt.X('weekday:N', sort=weekdays, title='요일'),
    y=alt.Y('rate:Q',     title='평균 증가 배수'),
    tooltip=['weekday','rate']
).properties(height=250)
st.altair_chart(chart_ce2, use_container_width=True)

# 2-6) 최근 3개월 매출 추이
st.subheader(f"📈 '{selected}' 최근 3개월 코인 매출 추이")
recent_coin = df_coin_sel[df_coin_sel['date'] >= df_coin_sel['date'].max() - timedelta(days=90)]
st.line_chart(recent_coin.set_index('date')['Total_coins'])

# 2-7) 코인 매출 향후 15일 예측
st.subheader("🔮 코인 매출 향후 15일 예측")
prophet_coin = df_coin_sel.rename(columns={'date':'ds','Total_coins':'y'})
model_coin   = Prophet(); model_coin.add_country_holidays(country_name='DE')
model_coin.fit(prophet_coin)
future_coin   = model_coin.make_future_dataframe(periods=15)
forecast_coin = model_coin.predict(future_coin)
coin_fut      = forecast_coin[forecast_coin['ds'] > df_coin_sel['date'].max()]
st.line_chart(coin_fut.set_index('ds')['yhat'])


# -- 3. 결제 주기 분석 --
st.header("⏱ 결제 주기 & 평균 결제금액 계산")

# 기간 설정
col1, col2, col3 = st.columns(3)
with col1:
    date_range = st.date_input("결제 주기 기간 설정", [], key="cycle_date_range")
with col2:
    k = st.number_input("첫 번째 결제 건수 (count)", min_value=1, value=2, key="cycle_k")
with col3:
    m = st.number_input("두 번째 결제 건수 (count)", min_value=1, value=3, key="cycle_m")

if st.button("결제 주기 계산", key="btn_cycle"):
    if len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        # raw payment 로드 (platform 포함)
        df_raw = pd.read_sql(
            'SELECT user_id, platform, count, date, amount FROM payment',
            con=engine
        )
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        # 기간 및 count 필터링
        mask = (
            (df_raw['date'] >= start) &
            (df_raw['date'] <= end) &
            (df_raw['count'].isin([k, m]))
        )
        df = df_raw.loc[mask, ['user_id','platform','count','date','amount']]

        # 두 결제 분리
        df_k = (
            df[df['count'] == k]
            .set_index('user_id')[['date','amount','platform']]
            .rename(columns={'date':'date_k','amount':'amt_k'})
        )
        df_m = (
            df[df['count'] == m]
            .set_index('user_id')[['date','amount']]
        )
        df_m.columns = ['date_m','amt_m']

        joined = df_k.join(df_m, how='inner')

        # 결제 주기 계산
        joined['days_diff']   = (joined['date_m'] - joined['date_k']).dt.days
        avg_cycle    = joined['days_diff'].mean()
        median_cycle = joined['days_diff'].median()
        mode_cycle   = joined['days_diff'].mode().iloc[0] if not joined['days_diff'].mode().empty else 0

        # 결제금액 통계
        amt_series    = joined[['amt_k','amt_m']].stack()
        avg_amount    = amt_series.mean()
        median_amount = amt_series.median()
        mode_amount   = amt_series.mode().iloc[0] if not amt_series.mode().empty else 0

        # 결과 출력
        st.success(f"결제주기 → 평균: {avg_cycle:.1f}일 | 중앙값: {median_cycle:.1f}일 | 최빈값: {mode_cycle:.1f}일")
        st.success(f"결제금액 → 평균: {avg_amount:.2f} | 중앙값: {median_amount:.2f} | 최빈값: {mode_amount:.2f}")

        # 플랫폼 분포 (코드→전체명 매핑)
        plat_counts = joined['platform'].value_counts()
        total_plat  = plat_counts.sum()
        mapping     = {'M':'Mobile Web','W':'PC Web','P':'Android','A':'Apple'}
        parts       = [
            f"{mapping.get(plat,plat)}: {cnt}건 ({cnt/total_plat:.1%})"
            for plat, cnt in plat_counts.items()
        ]
        st.success("플랫폼 분포 → " + ", ".join(parts))

    else:
        st.error("❗️ 시작일과 종료일을 모두 선택해주세요.")
