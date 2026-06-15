def exponential_smoothing(values: list[float], horizon: int = 5, alpha: float = 0.35) -> list[float]:
    if not values:
        return []
    level = values[0]
    for v in values[1:]:
        level = alpha * v + (1 - alpha) * level
    return [round(level, 3) for _ in range(horizon)]
