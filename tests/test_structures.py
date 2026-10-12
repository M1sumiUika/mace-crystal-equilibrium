"""几何和单位测试；不调用模型，不产生正式材料预测。"""
import numpy as np
import pytest
from ase import units

from mace_crystal_equilibrium.structures import MATERIALS, make_crystal, scale_volume
from mace_crystal_equilibrium.units import EV_A3_TO_GPA


@pytest.mark.parametrize("material,n_conventional,n_primitive", [("Si", 8, 2), ("Al", 4, 1)])
def test_atom_counts_and_volume_to_lattice(material, n_conventional, n_primitive):
    # 任意给定 a，避免测试只对默认初始猜测成立。
    lattice = 5.6 if material == "Si" else 4.2
    conventional = make_crystal(material, a_A=lattice, cubic=True)
    primitive = make_crystal(material, a_A=lattice, cubic=False)
    assert len(conventional) == n_conventional == MATERIALS[material]["n_atoms"]
    assert len(primitive) == n_primitive
    assert conventional.pbc.all() and primitive.pbc.all()
    np.testing.assert_allclose(conventional.cell.array, lattice * np.eye(3), atol=1e-13)
    assert conventional.get_volume() == pytest.approx(lattice**3)
    volume_per_atom = conventional.get_volume() / len(conventional)
    assert primitive.get_volume() / len(primitive) == pytest.approx(volume_per_atom)
    assert (n_conventional * volume_per_atom) ** (1 / 3) == pytest.approx(lattice)


@pytest.mark.parametrize("material", ["Si", "Al"])
@pytest.mark.parametrize("volume_ratio", [0.9, 1.1])
def test_volume_scan_preserves_fractional_coordinates(material, volume_ratio):
    atoms = make_crystal(material)
    old_cell = atoms.cell.array.copy()
    old_positions = atoms.positions.copy()
    old_fractional = atoms.get_scaled_positions(wrap=False)
    scaled = scale_volume(atoms, volume_ratio)
    assert scaled.get_volume() / atoms.get_volume() == pytest.approx(volume_ratio)
    np.testing.assert_allclose(scaled.cell.array, old_cell * volume_ratio ** (1 / 3), atol=1e-13)
    np.testing.assert_allclose(scaled.get_scaled_positions(wrap=False), old_fractional, atol=1e-13)
    # 扫描不能原地累计缩放参考晶胞。
    np.testing.assert_array_equal(atoms.cell.array, old_cell)
    np.testing.assert_array_equal(atoms.positions, old_positions)


@pytest.mark.parametrize("volume_ratio", [0, -0.1])
def test_nonpositive_volume_rejected(volume_ratio):
    with pytest.raises(ValueError, match="positive"):
        scale_volume(make_crystal("Si"), volume_ratio)


def test_ev_per_angstrom_cubed_to_gpa():
    # eV -> J，Å³ -> m³，Pa -> GPa；独立按 SI 定义核算。
    si_conversion = 1.602176634e-19 / (1e-10) ** 3 / 1e9
    assert EV_A3_TO_GPA == pytest.approx(si_conversion, rel=1e-12)
    # ASE 默认 CODATA 2014 与 SI 精确定义只存在末位差别。
    assert EV_A3_TO_GPA == pytest.approx(1 / units.GPa, rel=1e-7)
