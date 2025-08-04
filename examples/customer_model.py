"""
Customer model - provides context for conflict resolution
"""

from enum import Enum


class CustomerType(Enum):
    """Customer types with their discount rates"""
    REGULAR = "regular"  # 5% discount
    PREMIUM = "premium"  # 15% discount
    VIP = "vip"  # 30% discount
    ENTERPRISE = "enterprise"  # 25% discount


class Customer:
    """Customer with type and purchase history"""

    def __init__(self, id, name, customer_type):
        self.id = id
        self.name = name
        self.type = customer_type
        self.purchase_history = []

    def get_discount_rate(self):
        """Get discount rate based on customer type"""
        discount_rates = {
            CustomerType.REGULAR: 0.05,
            CustomerType.PREMIUM: 0.15,
            CustomerType.VIP: 0.30,
            CustomerType.ENTERPRISE: 0.25
        }
        return discount_rates.get(self.type, 0)

    def apply_discount(self, price):
        """Apply customer discount to price"""
        return price * (1 - self.get_discount_rate())


class Product:
    """Product with ID and price"""

    def __init__(self, id, name, price):
        self.id = id
        self.name = name
        self.price = price

    def __repr__(self):
        return f"Product({self.id}, {self.name}, ${self.price})"