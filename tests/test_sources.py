from aguamenti.sources import DataGenerator


def test_data_generator():
    gen = DataGenerator("2020-01-01", "2021-01-01", 1000)
    res = gen.generate()