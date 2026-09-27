import random

import pytest

from doc_extraction.synth.piva import is_valid_piva, piva_check_digit, random_piva


@pytest.mark.parametrize("piva", ["00905811006", "00488410010"])  # ENI, Telecom Italia
def test_known_real_vat_numbers_are_valid(piva: str) -> None:
    assert is_valid_piva(piva)


@pytest.mark.parametrize("piva", ["00905811007", "0090581100", "0090581100a"])
def test_wrong_check_digit_or_format_is_invalid(piva: str) -> None:
    assert not is_valid_piva(piva)


def test_check_digit_of_known_number() -> None:
    assert piva_check_digit("0090581100") == 6


def test_random_vat_numbers_are_valid() -> None:
    rng = random.Random(0)
    assert all(is_valid_piva(random_piva(rng)) for _ in range(500))
