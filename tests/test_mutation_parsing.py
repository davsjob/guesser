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


def test_read_mutation_csv(tmp_path):
    from guesser.graphs import read_mutation_csv
    p = tmp_path / "m.csv"
    p.write_text("pdb,mutation,chain,ddg\nprot,L45F,B,1.5\nprot.pdb,A:L45F+A:G50A,,\n")
    t = read_mutation_csv(p)
    assert list(t.pdb) == ["prot.pdb", "prot.pdb"]
    assert t.mut_list[0] == [("B", "LEU", 45, "PHE")]
    assert t.mut_list[1] == [("A", "LEU", 45, "PHE"), ("A", "GLY", 50, "ALA")]
    assert t.ddg[0] == 1.5 and t.ddg.isna()[1] and list(t.input) == ["L45F", "A:L45F+A:G50A"]
