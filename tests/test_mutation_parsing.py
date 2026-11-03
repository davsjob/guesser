import pytest

from guesser.graphs import parse_mutation


def test_formats():
    assert parse_mutation("A:L45F") == [("A", "LEU", 45, "PHE")]
    assert parse_mutation("L45F", default_chain="B") == [("B", "LEU", 45, "PHE")]
    assert parse_mutation("A:LEU45PHE") == [("A", "LEU", 45, "PHE")]
    assert parse_mutation("A:L45F+B:G50A") == [("A", "LEU", 45, "PHE"), ("B", "GLY", 50, "ALA")]


def test_bad():
    with pytest.raises(ValueError):
        parse_mutation("45F")
