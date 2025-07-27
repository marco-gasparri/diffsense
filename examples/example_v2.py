def calculate_total(a, b, fee=0):
    subtotal = a + b
    return subtotal + fee

def print_receipt(total, currency="$"):
    print(f"{currency} {total}")
