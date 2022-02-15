from venv import create
import numpy as np
import pandas as pd


from .categorical import create_all_hierarchy


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
            hierarchy,
        ):
        self.start_date = start_date
        self.end_date = end_date
        self.hierarchy = hierarchy
        self.hierarchy_df = create_all_hierarchy(hierarchy)
        self.dates = pd.date_range(self.start_date, self.end_date)
        self.unit_rows = len(self.dates)
        self.repeat = len(self.hierarchy_df)

    def generate_dates(self):
        return self.dates.repeat(self.repeat)

    def generate_categorical(self):
        return create_all_hierarchy(self.hierarchy)

    def generate_timeseries(self):
        return np.concatenate(
            [fourier(size=self.unit_rows) for i in range(self.repeat)]
        )

    def generate(self):
        categoricals = pd.concat([self.hierarchy_df] * self.unit_rows).reset_index(drop=True)
        df = pd.DataFrame({"date": self.generate_dates()})
        df["y"] = self.generate_timeseries()
        df = pd.concat([df, categoricals], axis=1)
        return df
