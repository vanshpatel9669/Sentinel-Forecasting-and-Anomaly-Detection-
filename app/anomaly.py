import statistics

def cusum_anomalies(values: list[float], threshold: float = 4.0, drift: float = 0.5) -> list[int]:
    if len(values) < 5:
        return []
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values) or 1.0
    pos = neg = 0.0
    anomalies = []
    for i, value in enumerate(values):
        z = (value - mean) / stdev
        pos = max(0.0, pos + z - drift)
        neg = min(0.0, neg + z + drift)
        if pos > threshold or abs(neg) > threshold:
            anomalies.append(i)
            pos = neg = 0.0
    return anomalies
