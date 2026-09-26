def apply_discount(amount, percent):
    """Return `amount` reduced by `percent` percent (0-100)."""
    if percent < 0 or percent > 100:
        raise ValueError("percent must be between 0 and 100")
    return amount - amount * percent
