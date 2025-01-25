if __name__ == "__main__":
    db = DatabaseManager()
    db.drop_candles_table()   # gooi oude weg
    db.create_candles_table() # maak nieuwe
    ...
