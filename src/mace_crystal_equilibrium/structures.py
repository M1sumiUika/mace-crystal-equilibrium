"""构造理想立方晶体；初值只是扫描中心，不是参考测量值。"""
from ase.build import bulk

MATERIALS = {
    "Si": {"phase": "diamond", "n_atoms": 8, "a_ref": 5.43},
    "Al": {"phase": "fcc", "n_atoms": 4, "a_ref": 4.05},
}


def make_crystal(material, a_A=None, cubic=True):
    spec = MATERIALS[material]
    return bulk(material, spec["phase"], a=a_A or spec["a_ref"], cubic=cubic)


def scale_volume(atoms, volume_ratio):
    if volume_ratio <= 0:
        raise ValueError("Volume ratio must be positive")
    result = atoms.copy()
    # 各向同性体积变化的边长比例为体积比例的立方根。
    result.set_cell(atoms.cell * volume_ratio ** (1 / 3), scale_atoms=True)
    return result

