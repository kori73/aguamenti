import pandas as pd

from aguamenti.categorical import create_all_hierarchy


def test_hierarchy():
    hierarchy = {
        "item": {"unique_count": 2, "dependencies": set()},
        "store": {"unique_count": 2, "dependencies": set()},
        "item_group": {"unique_count": 1, "dependencies": {"item"}}
    }
    res = create_all_hierarchy(hierarchy)
    expected = {
        "item": pd.Series([0, 0, 1, 1]),
        "store": pd.Series([0, 1, 0, 1]),
        "item_group": pd.Series([0, 0, 0, 0])
    }
    for key, value in expected.items():
        assert all(res[key] == value)
