def is_happy_number(n):
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(digit) ** 2 for digit in str(n))
    return n == 1

def find_happy_numbers(count):
    happy_numbers = []
    n = 1
    while len(happy_numbers) < count:
        if is_happy_number(n):
            happy_numbers.append(n)
        n += 1
    return happy_numbers

def main():
    count = 8
    happy_numbers = find_happy_numbers(count)
    print(f"The first {count} happy numbers are: {happy_numbers}")

if __name__ == "__main__":
    main()
