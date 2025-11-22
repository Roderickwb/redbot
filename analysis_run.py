# analysis_run.py
#
# Eenvoudig script om de CoinAnalyzer te draaien voor één coin
# en alles in de console te printen.

from src.analysis.coin_analyzer import analyze_all_and_print

def main():
    # Kies hier een coin waar je al trades voor hebt, bv. BTC-EUR
    analyze_all_and_print(last_n_trades=50, last_n_hold=200)

if __name__ == "__main__":
    main()
