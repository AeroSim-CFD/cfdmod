"""Generate the compact, anonymized structural-import fixtures.

Run: uv run python fixtures/tests/dynamics/imports/_generate_fixtures.py

Writes a small TQS "PORTELS" export (tqs/PORTELS_*.TXT) and an Eberick
per-floor workbook (eberick/modal.xlsx). Both describe the same tiny
3-floor building so tests can cross-check the two readers. Nothing here
comes from a real project: coordinates, masses and shapes are synthetic.

The TQS files are byte-format-faithful to a real export (Latin-1,
``//`` comments, comma decimals, TAB-separated), except comment text is
plain ASCII (the parser ignores comment lines).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent

# 3 floors x 4 corner nodes; plan 10 m x 6 m centred at (1.0, 0.5).
FLOOR_Z = [3.0, 6.0, 9.0]
CORNERS = np.array([[-4.0, -2.5], [6.0, -2.5], [6.0, 3.5], [-4.0, 3.5]])
NODE_MASS = [30.0, 20.0, 20.0, 30.0]  # unequal -> exercises mass weighting
PERIODS = [1.0, 0.40]
# Per-floor rigid-diaphragm shape [DX, DY, RZ] per mode (uniform across a floor's nodes).
FLOOR_SHAPES = {
    1: [(0.33, 0.0, 0.0), (0.66, 0.0, 0.0), (1.0, 0.0, 0.0)],  # sway X
    2: [(0.0, 0.30, 0.010), (0.0, 0.65, 0.020), (0.0, 1.0, 0.030)],  # sway Y + torsion
}


def _num(v: float) -> str:
    """Format like TQS: 3-sig-fig scientific with a comma decimal."""
    return f"{v:.3E}".replace(".", ",")


def _node_id(floor: int, corner: int) -> int:
    return (floor + 1) * 10 + corner  # floors 0..2 -> 10.., 20.., 30..


def write_tqs() -> None:
    out = HERE / "tqs"
    out.mkdir(parents=True, exist_ok=True)
    node_ids = [_node_id(f, k) for f in range(len(FLOOR_Z)) for k in range(4)]

    modos = [
        "// Numero total de modos",
        str(len(PERIODS)),
        "// Modo; Periodo (s); Frequencia angular (rad/s); Frequencia (Hz)",
    ]
    for i, T in enumerate(PERIODS, start=1):
        f = 1.0 / T
        modos.append("\t".join([f"{i:03d}", _num(T), _num(2 * np.pi * f), _num(f)]))
    (out / "PORTELS_MODOS.TXT").write_text("\n".join(modos) + "\n", encoding="latin-1")

    nos = ["// Numero total de nos", str(len(node_ids)), "// No; X (m); Y (m); Z (m)"]
    massas = ["// No; Massa em X (ton/g); Massa em Y (ton/g); Massa em Z (ton/g)"]
    for f in range(len(FLOOR_Z)):
        for k in range(4):
            nid = _node_id(f, k)
            x, y = CORNERS[k]
            nos.append("\t".join([f"{nid:06d}", _num(x), _num(y), _num(FLOOR_Z[f])]))
            m = NODE_MASS[k]
            massas.append("\t".join([f"{nid:06d}", _num(m), _num(m), _num(0.0)]))
    (out / "PORTELS_NOS.TXT").write_text("\n".join(nos) + "\n", encoding="latin-1")
    (out / "PORTELS_MASSAS.TXT").write_text("\n".join(massas) + "\n", encoding="latin-1")

    formas = []
    for mode in (1, 2):
        formas += ["// Modo", f"{mode:03d}", "// No; DX; DY; RZ"]
        for f in range(len(FLOOR_Z)):
            dx, dy, rz = FLOOR_SHAPES[mode][f]
            for k in range(4):
                formas.append("\t".join([f"{_node_id(f, k):06d}", _num(dx), _num(dy), _num(rz)]))
    (out / "PORTELS_FORMAS2.TXT").write_text("\n".join(formas) + "\n", encoding="latin-1")
    print("wrote", out)


def write_eberick() -> None:
    out = HERE / "eberick"
    out.mkdir(parents=True, exist_ok=True)
    total_mass = float(sum(NODE_MASS))
    xg = float(np.average(CORNERS[:, 0], weights=NODE_MASS))
    yg = float(np.average(CORNERS[:, 1], weights=NODE_MASS))
    inertia = float(
        sum(m * ((x - xg) ** 2 + (y - yg) ** 2) for (x, y), m in zip(CORNERS, NODE_MASS))
    )

    floors = pd.DataFrame(
        {
            "Pavimento": [f"Pav{f + 1}" for f in range(len(FLOOR_Z))],
            "Cota": FLOOR_Z,
            "Massa": [total_mass] * len(FLOOR_Z),
            "Inercia": [inertia] * len(FLOOR_Z),
            "Xcg": [xg] * len(FLOOR_Z),
            "Ycg": [yg] * len(FLOOR_Z),
        }
    )
    modes = pd.DataFrame({"Modo": [1, 2], "Periodo": PERIODS})
    rows = []
    for mode in (1, 2):
        for f in range(len(FLOOR_Z)):
            dx, dy, rz = FLOOR_SHAPES[mode][f]
            rows.append({"Pavimento": f"Pav{f + 1}", "Modo": mode, "DX": dx, "DY": dy, "RZ": rz})
    shapes = pd.DataFrame(rows)

    with pd.ExcelWriter(out / "modal.xlsx", engine="openpyxl") as w:
        floors.to_excel(w, sheet_name="Pavimentos", index=False)
        modes.to_excel(w, sheet_name="Modos", index=False)
        shapes.to_excel(w, sheet_name="Formas", index=False)
    print("wrote", out / "modal.xlsx")


if __name__ == "__main__":
    write_tqs()
    write_eberick()
