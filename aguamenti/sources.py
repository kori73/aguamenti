import numpy as np
import pandas as pd


def dates(start, end):
    return pd.date_range(start, end)


def categorical(size):
    return np.arange(size)


def fourier(n=10, P=0.5, size=1000):
    xs = np.arange(0, 1, 1/size).reshape(-1, 1)
    ns = (np.arange(n) + 1).reshape(1, -1)
    inner = xs @ (2 * np.pi / P * ns)
    a = np.random.normal(size=n).reshape(-1, 1)
    b = np.random.normal(size=n).reshape(-1, 1)

    res = (np.sin(inner) @ a + np.cos(inner) @ b)
    return res


class DataGenerator:
    def __init__(
            self,
            start_date,
            end_date,
            partition_rows
        ):
        self.start_date = start_date
        self.end_date = end_date
        self.partition_rows = partition_rows
        self.dates = pd.date_range(self.start_date, self.end_date)
        self.unit_rows = len(self.dates)
        self.repeat = self.partition_rows // self.unit_rows

    def generate_dates(self):
        return self.dates.repeat(self.repeat)

    def generate_categorical(self):
        return np.tile(categorical(self.unit_rows), self.repeat)

    def generate_timeseries(self):
        return np.concatenate(
            [fourier(size=self.unit_rows) for i in range(self.repeat)]
        )

    def generate(self):
        df = pd.DataFrame({"date": self.generate_dates()})
        df["x"] = self.generate_categorical()
        df["y"] = self.generate_timeseries()
        return df
