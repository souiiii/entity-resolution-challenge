from src.preprocessing.addresses import normalize_address
from src.preprocessing.names import normalize_business_name
from src.preprocessing.normalization import normalize_text


def test_normalize_text():
    assert normalize_text("  ACME, Inc.  ") == "acme inc"


def test_name_normalization():
    assert normalize_business_name("ACME Pvt. Ltd.") == "acme pvt ltd"


def test_optional_legal_suffix_removal():
    assert normalize_business_name(
        "ACME Pvt. Ltd.",
        remove_legal_suffixes=True,
    ) == "acme"


def test_address_normalization():
    assert normalize_address("10 Market Street") == "10 market st"
