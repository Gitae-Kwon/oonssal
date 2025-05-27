from sqlalchemy import create_engine
import pandas as pd

# 연결 정보
user = "neondb_owner"
password = "npg_4zyS9iFYvTOo"
host = "ep-long-shape-a81tjfy3-pooler.eastus2.azure.neon.tech"
database = "neondb"
port = 5432

# SQLAlchemy 접속 문자열
conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}?sslmode=require"
engine = create_engine(conn_str)

# 연결 테스트
with engine.connect() as conn:
    df = pd.read_sql("SELECT current_date;", conn)
    print("✅ 연결 성공! 오늘 날짜:", df.iloc[0, 0])
