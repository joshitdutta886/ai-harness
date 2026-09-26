import unittest

from shop.cart import Cart


class CartTests(unittest.TestCase):
    def test_subtotal(self):
        c = Cart()
        c.add("pen", 2.5, 4)
        c.add("book", 10)
        self.assertEqual(c.subtotal(), 20)

    def test_total_without_discount(self):
        c = Cart()
        c.add("pen", 2.5, 4)
        self.assertEqual(c.total(), 10)

    def test_invalid_quantity(self):
        with self.assertRaises(ValueError):
            Cart().add("pen", 1, 0)


if __name__ == "__main__":
    unittest.main()
