from functools import reduce

import numpy as np
import pandas as pd


def create_granular_hierarchy(granular_hierarchy, hierarchy):
    dfs = []
    for col in granular_hierarchy:
        vals = np.arange(hierarchy[col]["unique_count"])
        dfs.append(pd.DataFrame({col: vals}))
    return reduce(lambda x, y: x.merge(y, how="cross"), dfs)


def create_all_hierarchy(hierarchy):
    granular_hierarchy = [k for k, v in hierarchy.items() if not v.get("dependencies")]
    result = create_granular_hierarchy(granular_hierarchy, hierarchy)

    dependents = [k for k, v in hierarchy.items() if v.get("dependencies")]

    for col in dependents:
        vals = np.arange(hierarchy[col]["unique_count"])
        dependencies = hierarchy[col]["dependencies"]
        for dep in dependencies:
            dep_vals = result[dep].unique()
            mapped_vals = np.random.choice(vals, size=len(dep_vals))
            mapping = pd.DataFrame({col: mapped_vals, dep: dep_vals})
            result = result.merge(mapping, on=dep)
    return result