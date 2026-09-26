Title: Cart.total() returns a negative number when a discount is applied

When I apply a 10% discount to a cart worth 200, `Cart.total(discount_percent=10)`
returns -1800 instead of 180.

Steps to reproduce:
```python
from shop.cart import Cart
c = Cart()
c.add("chair", 100, 2)
print(c.total(discount_percent=10))   # expected 180.0, got -1800
```

Expected: the discount should reduce the total by the given percentage.
