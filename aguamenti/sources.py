import numpy as np
import pandas as pd
import dask.dataframe as dd

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
            partition_col=None
        ):
        self.start_date = start_date
        self.end_date = end_date
        self.hierarchy = hierarchy
        self.hierarchy_df = create_all_hierarchy(hierarchy)
        self.dates = pd.date_range(self.start_date, self.end_date)
        self.unit_rows = len(self.dates)
        self.repeat = len(self.hierarchy_df)
        self.partition_col = partition_col

    def generate_dates(self, repeat):
        return self.dates.repeat(repeat)

    def generate_categorical(self):
        return create_all_hierarchy(self.hierarchy)

    def generate_timeseries(self, repeat):
        return np.concatenate(
            [fourier(size=self.unit_rows) for i in range(repeat)]
        )

    def get_distinct_partition_vals(self):
        df = self.hierarchy_df
        return sorted(df[self.partition_col].unique())

    def generate(self, partition=None):
        if partition is None:
            hierarchy_df = self.hierarchy_df
        else:
            hierarchy_df = self.hierarchy_df[self.hierarchy_df[self.partition_col] == partition]
        categoricals = pd.concat([hierarchy_df] * self.unit_rows).reset_index(drop=True)
        df = pd.DataFrame({"date": self.generate_dates(len(hierarchy_df))})
        df["y"] = self.generate_timeseries(len(hierarchy_df))
        df = pd.concat([df, categoricals], axis=1)
        return df

    def main(self):
        if self.partition_col is None:
            return self.generate(self.hierarchy_df)

        if self.partition_col in self.hierarchy_df.columns:
            dsk = {}

            distinct_partitions = self.get_distinct_partition_vals()
            divisions = list(distinct_partitions)
            divisions = divisions + [divisions[-1]]
            for partition_val in distinct_partitions:
                dsk[("generate", partition_val)] = (self.generate, partition_val)
            meta = self.generate(distinct_partitions[0])
            return dd.DataFrame(dsk, "generate", meta, divisions)
