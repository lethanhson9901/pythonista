def calculate_pi(decimal_places):
    """
    Calculate pi (π) to a specified number of decimal places using the Bailey-Borwein-Plouffe BBP formula.

    Args:
        decimal_places (int): The number of decimal places to calculate pi to.

    Returns:
        str: The value of pi to the specified decimal places as a string.
    """
    if decimal_places <= 0:
        return "3"  # Default to 3 if no decimal places requested

    pi = 0.0

    for k in range(decimal_places):
        term = (1 / 16 ** k) * (
            4 / (8 * k + 1) - 2 / (8 * k + 4) - 1 / (8 * k + 5) - 1 / (8 * k + 6)
        )
        pi += term

    return f"{pi:.{decimal_places}f}"

def main():
    try:
        decimal_places = int(input("Enter the number of decimal places for pi: "))
        if 0 <= decimal_places <= 1000:
            pi_value = calculate_pi(decimal_places)
            print(f"Pi to {decimal_places} decimal places is: {pi_value}")
        else:
            print("Please enter a valid number of decimal places between 0 and 1000.")
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

if __name__ == "__main__":
    main()
