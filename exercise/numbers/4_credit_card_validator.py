"""
The Luhn algorithm, also known as the "modulus 10" or "mod 10" algorithm, is a simple checksum formula used to validate a variety of identification numbers, such as credit card numbers, IMEI numbers, and social security numbers. 
Its primary purpose is to catch errors in data entry and prevent simple mistakes, such as mistyping a digit.

Here's a step-by-step explanation of the Luhn algorithm:

1. Starting from the rightmost digit (the least significant digit) and moving left, double the value of every second digit. If the result is greater than 9, subtract 9 from it. This step is commonly applied to odd-positioned digits, as you'll see in the next steps.

2. Sum all the digits of the original number (not doubled) that were not modified in step 1. This step involves adding the digits in even positions and the digits that were doubled in odd positions.

3. Add the results from steps 1 and 2 together.

4. If the total from step 3 is divisible by 10 (i.e., the remainder is 0), the number is considered valid according to the Luhn algorithm.

Here's an example to illustrate the Luhn algorithm using a hypothetical credit card number: 4112 3456 7890 1234.

1. Starting from the right and doubling every second digit:

   - 4 (no change)
   - 1 (doubled, becomes 2)
   - 1 (no change)
   - 2 (doubled, becomes 4)
   - 3 (no change)
   - 4 (doubled, becomes 8)
   - 5 (no change)
   - 6 (doubled, becomes 2)
   - 7 (no change)
   - 8 (doubled, becomes 7)
   - 9 (no change)
   - 0 (doubled, becomes 0)
   - 1 (no change)
   - 2 (doubled, becomes 4)
   - 3 (no change)
   - 4 (doubled, becomes 8)

2. Sum all the digits:

   4 + 1 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 0 + 1 + 2 + 3 + 4 = 60

3. Add the results from steps 1 and 2 together:

   60 + 34 = 94

4. Check if the total from step 3 is divisible by 10 (i.e., the remainder is 0). In this case, 94 % 10 equals 4, so it's not divisible by 10, and the number is not valid.

In practice, when a credit card number is generated or entered, it should pass this Luhn check to ensure that it's a potentially valid number before further verification with the issuing bank.
"""

def is_valid_credit_card(card_number):
    """
    Validate a credit card number using the Luhn algorithm.

    Args:
        card_number (str): The credit card number as a string.

    Returns:
        bool: True if the card number is potentially valid, False if not.
    """
    # Remove non-digit characters (e.g., spaces or hyphens)
    card_number = ''.join(filter(str.isdigit, card_number))

    if len(card_number) < 13 or len(card_number) > 19:
        return False

    # Reverse the digits
    reversed_digits = card_number[::-1]

    total = 0
    is_second = False

    for char in reversed_digits:
        digit = int(char)
        if is_second:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
        is_second = not is_second

    return total % 10 == 0

def main():
    card_number = input("Enter a credit card number: ")
    if is_valid_credit_card(card_number):
        print("This credit card number is potentially valid.")
    else:
        print("This credit card number is not valid.")

if __name__ == "__main__":
    main()
