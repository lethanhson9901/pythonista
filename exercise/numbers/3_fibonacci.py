def generate_fibonacci_sequence(n):
    """
    Generate the Fibonacci sequence up to the nth number.

    Args:
        n (int): The number of terms to generate in the Fibonacci sequence.

    Returns:
        list: A list containing the Fibonacci sequence up to the nth number.
    """
    if n <= 0:
        return []

    fibonacci_sequence = [0, 1]
    while len(fibonacci_sequence) < n:
        next_term = fibonacci_sequence[-1] + fibonacci_sequence[-2]
        fibonacci_sequence.append(next_term)

    return fibonacci_sequence

def main():
    n = int(input("Enter the number of Fibonacci sequence terms you want to generate: "))
    fibonacci_sequence = generate_fibonacci_sequence(n)
    
    if fibonacci_sequence:
        print(f"Fibonacci Sequence (up to {n} terms): {fibonacci_sequence}")
    else:
        print("Please enter a positive integer for the number of terms.")

if __name__ == "__main__":
    main()
