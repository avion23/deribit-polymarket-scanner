def safe_float(
    value,
    default: float = 0.0,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    try:
        result = float(value) if value is not None else default
        if min_val is not None:
            result = max(min_val, result)
        if max_val is not None:
            result = min(max_val, result)
        return result
    except (ValueError, TypeError):
        return default
