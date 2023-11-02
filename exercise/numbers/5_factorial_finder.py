def factorial_iterative(n):
    """
    Calculate the factorial of a positive integer 'n' using an iterative (loop) approach.

    Args:
        n (int): The positive integer for which to calculate the factorial.

    Returns:
        int: The factorial of 'n'.
    """
    if n < 0:
        return "Factorial is undefined for negative numbers"
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def factorial_recursive(n):
    """
    Calculate the factorial of a positive integer 'n' using a recursive approach.

    Args:
        n (int): The positive integer for which to calculate the factorial.

    Returns:
        int: The factorial of 'n'.
    """
    if n < 0:
        return "Factorial is undefined for negative numbers"
    if n == 0:
        return 1
    else:
        return n * factorial_recursive(n - 1)

def main():
    n = int(input("Enter a positive integer to calculate its factorial: "))
    factorial = factorial_recursive(n)
    print(f"The factorial of {n} (using recursion) is {factorial}")

    factorial = factorial_iterative(n)
    print(f"The factorial of {n} (using a loop) is {factorial}")

if __name__ == "__main__":
    main()
