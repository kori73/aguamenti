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


def test_explain():
    hierarchy = {
        "item": {"unique_count": 1000, "dependencies": set()},
        "store": {"unique_count": 2000, "dependencies": set()},
        "item_group": {"unique_count": 10, "dependencies": {"item"}}
    }
    gen = DataGenerator("2020-01-01", "2021-01-01", hierarchy, partition_col="item_group")
    res = gen.explain(return_description=True)
    assert "Will be 7340000000 rows" in res
    assert "Will be unique by ['item', 'store', 'date']" in res
    assert "Each unique ['item', 'store'] will have 367 dates" in res
    assert "Will be partitioned by item_group, resulting in 10 partitions" in res
