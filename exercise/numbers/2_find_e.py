import math

def calculate_e(decimal_places):
    """
    Calculate the mathematical constant 'e' up to the specified number of decimal places using a Taylor series expansion.

    Args:
        decimal_places (int): The number of decimal places to calculate 'e' up to.

    Returns:
        float: An approximate value of 'e' to the specified decimal places.
    """
    if decimal_places < 1:
        return "Invalid input. Please enter a positive integer."

    # Set a maximum limit to prevent excessive calculations
    max_limit = 10000
    if decimal_places > max_limit:
        return f"Decimal places cannot exceed {max_limit} due to program limitations."

    e = 0
    factorial = 1

    for n in range(max_limit):
        term = 1 / factorial
        e += term

        if abs(term) < 1 / (10**decimal_places):
            break

        factorial *= (n + 1)

    return round(e, decimal_places)

def main():
    decimal_places = int(input("Enter the number of decimal places for 'e': "))
    result = calculate_e(decimal_places)
    print(f"Approximate value of 'e' to {decimal_places} decimal places: {result}")

if __name__ == "__main__":
    main()
