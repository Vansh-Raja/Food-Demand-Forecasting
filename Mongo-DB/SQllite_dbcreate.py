import pandas as pd
from sqlalchemy import create_engine

csv_file_path = 'train1.csv'
df = pd.read_csv(csv_file_path)

sqlite_db_path = r"C:\Users\ayush\pythonProject2023\my_database.db"

engine = create_engine(f'sqlite:///{sqlite_db_path}')
df.to_sql('selected_data', engine, index=False, if_exists='replace')
print('SQLite table "selected_data" created successfully')
