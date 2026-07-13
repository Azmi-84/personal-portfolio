import pandas as pd

from smuba.data.clean import clean_data


def test_clean_data_drops_duplicates_and_id_columns():
    df = pd.DataFrame({
        "user_id": [1, 2, 2],
        "age": [20, 25, 25],
        "daily_usage_minutes": [100, 200, 200],
    })
    cleaned = clean_data(df, {"id_columns": ["user_id"]})

    assert "user_id" not in cleaned.columns
    assert len(cleaned) == 2
