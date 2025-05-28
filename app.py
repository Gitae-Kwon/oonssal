Webtoon Event Dashboard
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
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
    # 결제 매출: payment 테이블, amount 컬럼 합계
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

# 1) 코인 매출 데이터 준비 & 이벤트 분석
if selected_title == "전체 콘텐츠":
    df_selected = coin_df.groupby("date")["Total_coins"].sum().reset_index()
else:
    df_selected = coin_df[coin_df["Title"] == selected_title][["date", "Total_coins"]]

df_selected = df_selected.groupby("date")["Total_coins"].sum().reset_index().sort_values("date")

# 이벤트 임계치 입력 및 적용
if "threshold" not in st.session_state:
    st.session_state.threshold = 1.7
st.subheader("⚙️ 이벤트 임계치 설정 (코인)")
th_input = st.number_input(
    "평균 대비 몇 % 이상일 때 이벤트로 간주?", min_value=100, max_value=500,
    value=int(st.session_state.threshold*100)
)
if st.button("코인 임계치 적용"):
    st.session_state.threshold = th_input / 100
threshold = st.session_state.threshold
st.caption(f"현재 코인 이벤트 임계치: {int(threshold*100)}%")

# 이벤트 플래그 계산
df_selected["rolling_avg"] = df_selected["Total_coins"].rolling(window=7, center=True, min_periods=1).mean()
df_selected["event_flag"] = df_selected["Total_coins"] > df_selected["rolling_avg"] * threshold
df_selected["weekday"] = df_selected["date"].dt.day_name()

# 이벤트 발생 요일 분포 차트
weekday_event = df_selected[df_selected["event_flag"]]["weekday"].value_counts()
weekdays_order = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
df_ev = pd.DataFrame({
    'weekday': weekdays_order,
    'count': [weekday_event.get(day, 0) for day in weekdays_order]
})
df_ev['negative'] = -df_ev['count']
coin_chart = alt.Chart(df_ev).mark_bar(color='red').encode(
    x=alt.X('weekday:N', sort=weekdays_order, title='요일'),
    y=alt.Y('negative:Q', axis=alt.Axis(title='이벤트 횟수'), scale=alt.Scale(domain=[-max(df_ev['count'])-1, 0]))
).properties(height=250)
st.subheader("🌟 코인 이벤트 발생 요일 분포")
st.altair_chart(coin_chart, use_container_width=True)

# 최근 3개월 코인 매출 추이
recent_coin = df_selected[df_selected["date"] >= df_selected["date"].max() - timedelta(days=90)]
ctitle = "전체 콘텐츠" if selected_title == "전체 콘텐츠" else selected_title
st.subheader(f"📈 '{ctitle}' 최근 3개월 코인 매출 추이")
st.line_chart(recent_coin.set_index("date")["Total_coins"])

# 코인 매출 예측 (7일)
prophet_df = df_selected.rename(columns={"date": "ds", "Total_coins": "y"})
model = Prophet()
model.add_country_holidays(country_name="FR")
model.fit(prophet_df)
future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)
fut7 = forecast[forecast["ds"] > df_selected["date"].max()]
st.subheader("🔮 코인 향후 7일 예측")
st.line_chart(fut7.set_index("ds")["yhat"])

# 2) 결제 매출 섹션
st.header("💳 결제 매출 분석")
# 최근 3개월 결제 매출
recent_pay = pay_df[pay_df["date"] >= pay_df["date"].max() - timedelta(days=90)]
st.subheader("📈 결제 매출 최근 3개월 추이")
st.line_chart(recent_pay.set_index("date")["amount"])

# 결제 예측 (7일)
pay_prophet = pay_df.rename(columns={"date": "ds", "amount": "y"})
pay_model = Prophet()
pay_model.add_country_holidays(country_name="FR")
pay_model.fit(pay_prophet)
pay_future = pay_model.make_future_dataframe(periods=7)
pay_forecast = pay_model.predict(pay_future)
pay_fut7 = pay_forecast[pay_forecast["ds"] > pay_df["date"].max()]
st.subheader("🔮 결제 매출 향후 7일 예측")
st.line_chart(pay_fut7.set_index("ds")["yhat"])

# 3) 이벤트 예정일 선택 및 적용 기능
st.subheader("🗓 이벤트 예정일 체크 및 적용")
event_input = st.date_input(
    "이벤트 가능성 있는 날짜 선택", value=None, format="YYYY-MM-DD", key="event_input"
)
if st.button("이벤트 적용") and event_input:
    sel = event_input
    wd = sel.strftime("%A")
    total_days = df_selected[df_selected['weekday'] == wd].shape[0]
    rate = weekday_event.get(wd, 0) / total_days if total_days > 0 else 0
    st.write(f"📈 과거 {wd} 코인 이벤트 발생 비율: {rate:.1%}")
    if sel in fut7['ds'].dt.date.tolist():
        st.success(f"🚀 {sel}은 코인 예측 기간에 포함됩니다.")
    else:
        st.warning("⚠️ 선택한 날짜가 코인 예측 기간에 포함되지 않습니다.")
elif st.button("이벤트 적용"):
    st.warning("⚠️ 먼저 이벤트 날짜를 선택해주세요.")

