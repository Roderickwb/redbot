import time
from src.database_manager.database_manager import DatabaseManager
from src.config.config import DB_FILE

def main():
    print("=== MINIMAL TEST ===")

    db = DatabaseManager(db_path=DB_FILE)
    print("DB Manager aangemaakt.")

    # We roepen alleen .create_tables() aan
    db.create_tables()
    print("Tables ensured/created.")

    # Nu simpelweg 5 seconden wachten
    time.sleep(5)
    print("Eind minimal test.")

if __name__ == "__main__":
    main()
