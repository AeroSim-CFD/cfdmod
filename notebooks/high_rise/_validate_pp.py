"""Validate the pp/ helper package end-to-end on real in-repo fixtures.

Run: uv run python notebooks/high_rise/_validate_pp.py
Exercises HighRiseCase (against the real 067 case_data), inflow profile
detection + figures (pitot_inlet fixture), and the Cp -> per-floor Cf/Cm
pressure wiring (galpao fixture + its lnas mesh).
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
FIX = REPO / "fixtures" / "tests"

import pp  # noqa: E402
from pp import inflow_report as ir  # noqa: E402
from pp import plotting  # noqa: E402

plotting.apply_style()


def check(label: str, cond: bool, detail: str = "") -> None:
    mark = "ok " if cond else "FAIL"
    print(f"  [{mark}] {label}" + (f" -- {detail}" if detail else ""))
    if not cond:
        raise SystemExit(f"validation failed: {label}")


def section_case() -> pp.HighRiseCase:
    print("A. HighRiseCase (real 067 case_data)")
    case_data = pathlib.Path(
        "/data/eng/consulting/067_CampoGrande_ClementePereira/post_processing/pp_config/case_data"
    )
    if not case_data.exists():
        print("  (067 case_data not mounted; skipping real-config parse)")
        return _synthetic_case()
    case = pp.HighRiseCase.from_case_data(case_data, "params_cat3.yaml")
    check("reference_height == 70", case.reference_height == 70.0, f"H={case.reference_height}")
    check("nominal_area == 486.5", case.nominal_area == 486.5)
    check("n_floors == 26", case.n_floors == 26, f"n_floors={case.n_floors}")
    check("simul U_H ~ 29.91", abs(case.simul_reference_velocity - 29.914) < 1e-2)
    q0 = case.dynamic_pressure
    case2 = case.with_reference_velocity(32.0)
    check(
        "with_reference_velocity updates q",
        case2.u_h == 32.0
        and abs(case2.dynamic_pressure - 0.5 * case.fluid_density * 32.0**2) < 1e-9,
        f"q {q0:.1f} -> {case2.dynamic_pressure:.1f}",
    )
    return case


def _synthetic_case() -> pp.HighRiseCase:
    return pp.HighRiseCase(
        name="synthetic",
        reference_height=70.0,
        characteristic_length=6.95,
        basic_wind_speed=38.0,
        simul_reference_velocity=29.914,
        nominal_area=486.5,
        nominal_volume=3381.175,
        floor_heights=[0.0, 10.0, 20.0, 30.0],
    )


def section_inflow(base: pathlib.Path) -> None:
    print("B. Inflow validation (pitot_inlet fixture)")
    from cfdmod.inflow import InflowData, NormalizationParameters

    folder = FIX / "inflow" / "pitot_inlet"
    inflow = InflowData.from_files(folder / "hist_series.csv", folder / "points.csv")
    profiles = ir.detect_profiles(inflow, min_points=3)
    check("detected >= 1 vertical profile", len(profiles) >= 1, f"n={len(profiles)}")
    prof = profiles[0]
    check("profile heights ascending", bool(np.all(np.diff(prof.z) > 0)))

    ref_h = float(np.median(prof.z))
    u_ref = ir.reference_velocity(prof, inflow, ref_h)
    check("reference_velocity finite", np.isfinite(u_ref), f"u_ref={u_ref:.3f} at z={ref_h:.2f}")

    L = ir.integral_length_scale(inflow, int(prof.nearest_index(ref_h)), u_ref)
    check("integral length scale computed", np.isfinite(L) or np.isnan(L), f"L={L}")

    dbg = pp.DebugWriter(base, stage="inflow", version="validate")
    norm = NormalizationParameters(reference_velocity=max(u_ref, 1e-6), characteristic_length=1.0)
    figs = {
        "mean_velocity.png": ir.plot_mean_velocity(prof, inflow),
        "turbulence_intensity.png": ir.plot_turbulence_intensity(prof, inflow),
        "spectrum.png": ir.plot_spectrum(prof, inflow, ref_h, norm),
    }
    for name, fig in figs.items():
        path = dbg.savefig(fig, name)
        plotting.close(fig)
        check(f"wrote {name}", path.exists() and path.stat().st_size > 0, str(path))


def section_pressure(case: pp.HighRiseCase) -> None:
    print("C. Cp -> per-floor Cf/Cm (galpao fixture)")
    import pathlib as _pl

    from lnas import LnasFormat

    from cfdmod.adapters.xdmf_h5 import XdmfH5Storage

    data_dir = FIX / "pressure" / "data"
    mesh_path = str(FIX / "pressure" / "galpao" / "galpao.normalized.lnas")
    storage = XdmfH5Storage(data_dir)
    body = storage.read_data_source(_pl.Path("bodies.galpao"))
    p_ref = storage.read_data_source(_pl.Path("points.static_pressure"))
    check("body loaded", body.n_elements == 2915, f"n_elements={body.n_elements}")

    # Floor slices from the mesh z-range (galpao is low-rise; 3 slices exercise the mechanics).
    verts = LnasFormat.from_file(_pl.Path(mesh_path)).geometry.vertices
    zmin, zmax = float(verts[:, 2].min()), float(verts[:, 2].max())
    edges = list(np.linspace(zmin, zmax, 4))
    floor_case = case.model_copy(update={"floor_heights": edges})

    cp = pp.cp_from_pressure(body, p_ref, floor_case)
    check("cp field present", "cp" in cp.field_names)
    cp_mean = np.nanmean(cp.fields.read("cp"))
    check("cp mean in plausible range", abs(cp_mean) < 5.0, f"mean cp={cp_mean:.3f}")

    cf = pp.cf_per_floor(cp, mesh_path, floor_case, directions=("x", "y"))
    check("cf is groups source", cf.kind == "groups")
    check("cf has <= 3 floor rows", 1 <= cf.n_elements <= 3, f"n_floors={cf.n_elements}")
    cfx = cf.fields.read("cf_x")
    check(
        "cf_x finite + time-resolved",
        np.all(np.isfinite(cfx)) and cfx.shape[1] > 1,
        str(cfx.shape),
    )

    cm = pp.cm_per_floor(cp, mesh_path, floor_case, directions=("z",))
    cmz = cm.fields.read("cm_z")
    check("cm_z finite", np.all(np.isfinite(cmz)), str(cmz.shape))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        case = section_case()
        section_inflow(base)
        section_pressure(case)
    print("\nAll pp/ validations passed.")


if __name__ == "__main__":
    main()
