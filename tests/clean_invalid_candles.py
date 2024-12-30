import sqlite3
import logging
import os

# Configureer logging
logging.basicConfig(level=logging.INFO)

DB_FILE = os.path.abspath(os.path.join('C:\\Users\\My ACER\\PycharmProjects\\PythonProject4', 'market_data.db'))

def delete_invalid_candles():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # Verwijder rijen waar 'timestamp' geen integer is of onlogische waarden bevat
        cursor.execute("""
            DELETE FROM candles 
            WHERE typeof(timestamp) != 'integer' 
               OR timestamp < 0
        """)
        deleted = cursor.rowcount
        conn.commit()
        logging.info(f"Ongeldige candle records verwijderd: {deleted} record(s).")
    except sqlite3.Error as e:
        logging.error(f"Fout bij het verwijderen van ongeldige candle records: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    delete_invalid_candles()
