"""
Example file with merge conflicts
This file demonstrates how DiffSense can resolve merge conflicts
"""

def calculate_discount(price, customer_type):
    """Calculate discount based on customer type"""

<<<<<<< HEAD
    # Premium customers get 20% discount
    if customer_type == "premium":
        return price * 0.8
    elif customer_type == "regular":
        return price * 0.95
    else:
        return price
=======
    # VIP customers get special treatment
    if customer_type == "vip":
        return price * 0.7
    elif customer_type == "premium":
        return price * 0.85
    elif customer_type == "regular":
        return price * 0.95
    else:
        return price
>>>>>>> feature/vip-customers


def format_price(price):
    """Format price for display"""
<<<<<<< HEAD
    return f"${price:.2f}"
=======
    # Add currency symbol and thousand separators
    return f"${price:,.2f}"
>>>>>>> feature/vip-customers


class ShoppingCart:
    def __init__(self):
        self.items = []
<<<<<<< HEAD
        self.discount = 0
=======
        self.discount_percentage = 0
        self.customer = None
>>>>>>> feature/vip-customers

    def add_item(self, item, quantity=1):
        """Add item to cart"""
<<<<<<< HEAD
        self.items.append({
            "item": item,
            "quantity": quantity,
            "price": item.price * quantity
        })
=======
        # Check if item already exists
        for cart_item in self.items:
            if cart_item["item"].id == item.id:
                cart_item["quantity"] += quantity
                return

        self.items.append({
            "item": item,
            "quantity": quantity
        })
>>>>>>> feature/vip-customers

    def calculate_total(self):
        """Calculate total with discounts"""
<<<<<<< HEAD
        subtotal = sum(item["price"] for item in self.items)
        return subtotal * (1 - self.discount)
=======
        subtotal = sum(
            item["item"].price * item["quantity"]
            for item in self.items
        )

        if self.customer:
            return calculate_discount(subtotal, self.customer.type)
        return subtotal * (1 - self.discount_percentage / 100)
>>>>>>> feature/vip-customers