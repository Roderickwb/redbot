import sqlite3
import pandas as pd

db_path = "/market_data.db"

conn = sqlite3.connect(db_path)

# Haal gegevens op uit de ticker-tabel
df = pd.read_sql_query("SELECT * FROM ticker ORDER BY timestamp DESC LIMIT 10;", conn)

print(df)

conn.close()
