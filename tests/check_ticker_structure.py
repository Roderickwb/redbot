import sqlite3

# Geef hier het pad naar je database op
db_path = "C:/Users/My ACER/PycharmProjects/PythonProject4/market_data.db"

# Verbind met de database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Controleer de structuur van de 'ticker'-tabel
cursor.execute("PRAGMA table_info(ticker);")
columns = cursor.fetchall()

# Print de kolommen van de 'ticker'-tabel
print("Structuur van de 'ticker'-tabel:")
for column in columns:
    print(f"Kolomnaam: {column[1]}, Type: {column[2]}")

conn.close()
