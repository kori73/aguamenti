import dask.dataframe as dd

from aguamenti.sources import DataGenerator


def test_data_generator():
    hierarchy = {
        "item": {"unique_count": 2, "dependencies": set()},
        "store": {"unique_count": 2, "dependencies": set()},
        "item_group": {"unique_count": 1, "dependencies": {"item"}}
    }
    gen = DataGenerator("2020-01-01", "2021-01-01", hierarchy)
    res = gen.generate()
    expected = 1468
    assert len(res) == expected
    assert len(res.drop_duplicates(["date", "item", "store"])) == expected


def test_data_generator_partitioned():
    hierarchy = {
        "item": {"unique_count": 2, "dependencies": set()},
        "store": {"unique_count": 2, "dependencies": set()},
        "item_group": {"unique_count": 1, "dependencies": {"item"}}
    }
    gen = DataGenerator("2020-01-01", "2021-01-01", hierarchy, partition_col="item_group")
    res = gen.main()
    expected = 1468
    assert isinstance(res, dd.DataFrame)
    assert res.npartitions == 1
    assert len(res) == expected
    assert len(res.drop_duplicates(["date", "item", "store"])) == expected
