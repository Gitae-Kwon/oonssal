import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# --- DB 연결 설정 ---
user = "neondb_owner"
password = st.secrets["npg_4zyS9iFYvTOo"]  # 또는 직접 입력 (테스트용은 하드코딩도 가능)
host = "ep-long-shape-a81tjfy3-pooler.eastus2.azure.neon.tech"
db = "neondb"
port = 5432
conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}?sslmode=require"
engine = create_engine(conn_str)

# --- 앱 내용 ---
st.title("📈 웹툰 매출 분석 대시보드")

@st.cache_data
def load_data():
    return pd.read_sql("SELECT * FROM sales_table", engine)

df = load_data()
st.dataframe(df)

st.line_chart(df.groupby("date")["sales_amount"].sum())
