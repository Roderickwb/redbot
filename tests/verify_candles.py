# verify_candles.py
import sqlite3
import os

DB_PATH = r"C:\Users\My ACER\PycharmProjects\PythonProject4\data\market_data.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("--------- Per market/interval COUNT ----------")
cursor.execute("""
    SELECT market, interval, COUNT(*)
    FROM candles
    GROUP BY market, interval
    ORDER BY market, interval
""")
rows = cursor.fetchall()
for row in rows:
    print(row)

print("\n--------- Laatste 3 rijen per market/interval ----------")
cursor.execute("""
    SELECT market, interval, timestamp, close
    FROM candles
    ORDER BY id DESC
    LIMIT 30
""")
rows = cursor.fetchall()
for row in rows:
    print(row)

conn.close()
