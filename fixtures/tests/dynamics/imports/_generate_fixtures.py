"""Generate the compact, anonymized structural-import fixtures.

Run: uv run python fixtures/tests/dynamics/imports/_generate_fixtures.py

Writes a small TQS "PORTELSSE" export (tqs/PORTELSSE_*.TXT, the modern
prefix, including a PISOS floor table) and a real-layout Eberick per-floor
export (eberick/DISTRIBUICAO_DAS_MASSAS_DOS_PAVIMENTOS.xlsx +
FORMAS_MODAIS_DOS_PAVIMENTOS.xlsx). Both describe a tiny 3-floor building.
Nothing here comes from a real project: coordinates, masses and shapes are
synthetic and the identifying header block is blank.

The TQS files are byte-format-faithful to a real export (Latin-1,
``//`` comments, comma decimals, TAB-separated), except comment text is
plain ASCII (the parser ignores comment lines).
"""

from __future__ import annotations

import pathlib

import numpy as np

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
    (out / "PORTELSSE_MODOS.TXT").write_text("\n".join(modos) + "\n", encoding="latin-1")

    nos = ["// Numero total de nos", str(len(node_ids)), "// No; X (m); Y (m); Z (m)"]
    massas = ["// No; Massa em X (ton/g); Massa em Y (ton/g); Massa em Z (ton/g)"]
    for f in range(len(FLOOR_Z)):
        for k in range(4):
            nid = _node_id(f, k)
            x, y = CORNERS[k]
            nos.append("\t".join([f"{nid:06d}", _num(x), _num(y), _num(FLOOR_Z[f])]))
            m = NODE_MASS[k]
            massas.append("\t".join([f"{nid:06d}", _num(m), _num(m), _num(0.0)]))
    (out / "PORTELSSE_NOS.TXT").write_text("\n".join(nos) + "\n", encoding="latin-1")
    (out / "PORTELSSE_MASSAS.TXT").write_text("\n".join(massas) + "\n", encoding="latin-1")

    formas = []
    for mode in (1, 2):
        formas += ["// Modo", f"{mode:03d}", "// No; DX; DY; RZ"]
        for f in range(len(FLOOR_Z)):
            dx, dy, rz = FLOOR_SHAPES[mode][f]
            for k in range(4):
                formas.append("\t".join([f"{_node_id(f, k):06d}", _num(dx), _num(dy), _num(rz)]))
    (out / "PORTELSSE_FORMAS2.TXT").write_text("\n".join(formas) + "\n", encoding="latin-1")

    # PISOS floor table (name column deliberately contains spaces).
    pisos = ["// Piso; Nome; Nivel (m)"]
    names = ["Pav 1", "Pav 2", "Pav 3"]
    for f, z in enumerate(FLOOR_Z):
        pisos.append("\t".join([f"{f:03d}", names[f], _num(z)]))
    (out / "PORTELSSE_PISOS.TXT").write_text("\n".join(pisos) + "\n", encoding="latin-1")
    print("wrote", out)


# Eberick per-floor values (its native cm / tf.s^2/cm units), same 3 floors.
EB_NAMES = ["PAV 1", "PAV 2", "PAV 3"]
EB_ELEV_CM = [300.0, 600.0, 900.0]
EB_MASS = [0.30, 0.25, 0.20]  # tf.s^2/cm
EB_INERTIA = [75000.0, 62500.0, 50000.0]  # tf.s^2.cm -> radius sqrt(I/M)=500 cm = 5 m
EB_XCG_CM = [50.0, 50.0, 50.0]
EB_YCG_CM = [30.0, 30.0, 30.0]
EB_FREQ_HZ = [0.25, 0.60]
# Per-floor [Dx(cm), Dy(cm), Rz(rad)] per mode.
EB_SHAPES = {
    0: [(1.0, 0.0, 1.0e-5), (2.0, 0.0, 2.0e-5), (3.0, 0.0, 3.0e-5)],  # sway X
    1: [(0.0, 1.0, -1.0e-5), (0.0, 2.0, -2.0e-5), (0.0, 3.0, -3.0e-5)],  # sway Y
}


def _eb_num(v: float) -> str:
    """Eberick writes shape values as comma-decimal scientific strings."""
    return f"{v:.4E}".replace(".", ",")


def write_eberick() -> None:
    import openpyxl

    out = HERE / "eberick"
    out.mkdir(parents=True, exist_ok=True)

    # Identifying header block is blanked (anonymized); the reader skips it.
    head = [["OBRA", ""], ["Tipo", ""], ["Titulo", ""], ["Endereco", ""], ["Cliente", ""], []]

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "DISTRIBUICAO_DAS_MASSAS_DOS_PAV"
    for r in head:
        ws.append(r)
    ws.append(["Dados da analise dinamica"])
    ws.append([])
    ws.append(["Distribuicao das massas dos pavimentos"])
    ws.append(["O momento de inercia e dado em relacao ao centro de massa"])
    ws.append([])
    ws.append(
        [
            "Pavimento",
            "Altura (cm)",
            "Elevacao em relacao ao nivel do solo (cm)",
            "Massa (tf.s2/cm)",
            "Momento de inercia da massa (tf.s2.cm)",
            "Centro de massa",
            "",
        ]
    )
    ws.append(["", "", "", "", "", "Xcg (cm)", "Ycg (cm)"])
    for i in range(len(EB_NAMES)):
        ws.append(
            [
                EB_NAMES[i],
                300.0,
                EB_ELEV_CM[i],
                EB_MASS[i],
                EB_INERTIA[i],
                EB_XCG_CM[i],
                EB_YCG_CM[i],
            ]
        )
    ws.append(["AltoQi | Tecnologia aplicada a engenharia"])
    wb.save(out / "DISTRIBUICAO_DAS_MASSAS_DOS_PAVIMENTOS.xlsx")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "FORMAS_MODAIS_DOS_PAVIMENTOS"
    for r in head:
        ws.append(r)
    ws.append(["Dados da analise dinamica"])
    ws.append(["Formas modais dos pavimentos"])
    ws.append(["As formas modais sao dadas no centro de massa"])
    for mode in range(len(EB_FREQ_HZ)):
        ws.append(["", f"Modo {mode + 1}"])
        ws.append(["", f"Frequencia (Hz): {EB_FREQ_HZ[mode]}"])
        ws.append(["", "Pavimento", "Dx (cm)", "Dy (cm)", "Rz (rad)"])
        for i in range(len(EB_NAMES)):
            dx, dy, rz = EB_SHAPES[mode][i]
            ws.append(["", EB_NAMES[i], _eb_num(dx), _eb_num(dy), _eb_num(rz)])
    ws.append(["AltoQi | Tecnologia aplicada a engenharia"])
    wb.save(out / "FORMAS_MODAIS_DOS_PAVIMENTOS.xlsx")
    print("wrote", out / "(Eberick 2-file set)")


if __name__ == "__main__":
    write_tqs()
    write_eberick()
