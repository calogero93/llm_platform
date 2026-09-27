import random


def piva_check_digit(first_ten: str) -> int:
    """Check digit of an Italian partita IVA (Luhn-like over the first 10 digits)."""
    total = 0
    for i, ch in enumerate(first_ten):
        digit = int(ch)
        if i % 2 == 1:  # 2nd, 4th, ... digit (1-based even positions)
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return (10 - total % 10) % 10


def is_valid_piva(value: str) -> bool:
    return len(value) == 11 and value.isdigit() and piva_check_digit(value[:10]) == int(value[10])


def random_piva(rng: random.Random) -> str:
    # 7-digit taxpayer number + 3-digit provincial office code (001-100) + check digit.
    first_ten = f"{rng.randrange(1, 10_000_000):07d}{rng.randrange(1, 101):03d}"
    return first_ten + str(piva_check_digit(first_ten))
