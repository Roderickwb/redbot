from src.analysis.coin_profile_generator import (
    load_analysis_files,
    derive_profile,
    write_profiles_to_db,
)

STRATEGY_NAME = "trend_4h"

def run():
    analyses = load_analysis_files()
    profiles = {}
    for symbol, analysis in analyses.items():
        profiles[symbol] = derive_profile(analysis)

    write_profiles_to_db(profiles, strategy_name=STRATEGY_NAME)

if __name__ == "__main__":
    run()


