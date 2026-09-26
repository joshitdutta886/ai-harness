"""Basic statistics."""


def mean(values):
    if not values:
        raise ValueError("mean() of empty data")
    return sum(values) / len(values)


def median(values):
    if not values:
        raise ValueError("median() of empty data")
    data = sorted(values)
    n = len(data)
    return data[n // 2]
