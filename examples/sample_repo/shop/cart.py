"""A tiny shopping cart used as the harness demo repository."""
from .pricing import apply_discount


class Cart:
    def __init__(self):
        self.items = []  # list of (name, unit_price, quantity)

    def add(self, name, unit_price, quantity=1):
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self.items.append((name, unit_price, quantity))

    def subtotal(self):
        return sum(price * qty for _, price, qty in self.items)

    def total(self, discount_percent=0):
        return round(apply_discount(self.subtotal(), discount_percent), 2)
