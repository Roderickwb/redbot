import sqlite3

def remove_duplicates(db_path):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # Verwijder duplicaten in ticker
    cursor.execute("""
        DELETE FROM ticker
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM ticker
            GROUP BY timestamp, market
        );
    """)

    # Verwijder duplicaten in orderbook_bids
    cursor.execute("""
        DELETE FROM orderbook_bids
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM orderbook_bids
            GROUP BY timestamp, market, bid_p, bid_q
        );
    """)

    # Verwijder duplicaten in orderbook_asks
    cursor.execute("""
        DELETE FROM orderbook_asks
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM orderbook_asks
            GROUP BY timestamp, market, ask_p, ask_q
        );
    """)

    connection.commit()
    connection.close()
    print("Duplicaten succesvol verwijderd.")

# Voer opschoonscript uit
if __name__ == "__main__":
    db_path = "test_market_data.db"  # Pas aan naar de juiste database
    remove_duplicates(db_path)
