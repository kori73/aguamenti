import pandas as pd
import numpy as np

from aguamenti.categorical import create_all_hierarchy

np.random.seed(42)


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


def test_hierarchy_multiple():
    hierarchy = {
        "item": {"unique_count": 4, "dependencies": set()},
        "store": {"unique_count": 3, "dependencies": set()},
        "item_group": {"unique_count": 2, "dependencies": {"item"}},
        "store_group": {"unique_count": 2, "dependencies": {"store"}}
    }
    res = create_all_hierarchy(hierarchy)
    expected = {
        "item": pd.Series([0, 1, 2, 3] * 3),
        "store": pd.Series([0] * 4 + [1] * 4 + [2] * 4),
        "item_group": pd.Series([0, 1, 0, 0] * 3),
        "store_group": pd.Series([0] * 4 + [1] * 4 + [0] * 4)
    }
    for key, value in expected.items():
        assert all(res[key] == value), f"fails {key}, {res[key]}"