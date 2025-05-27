import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# 🔐 Secrets에서 불러오기
user = st.secrets["DB_USER"]
password = st.secrets["DB_PASSWORD"]
host = st.secrets["DB_HOST"]
port = st.secrets["DB_PORT"]
dbname = st.secrets["DB_NAME"]

# SQLAlchemy 연결 문자열
conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
engine = create_engine(conn_str)

# 데이터 불러오기
df = pd.read_sql("SELECT * FROM sales_table", engine)
st.title("📊 웹툰 매출 분석")
st.dataframe(df)
