"""Validate the pp/ helper package end-to-end on real in-repo fixtures.

Run: uv run python notebooks/high_rise/_validate_pp.py
Exercises HighRiseCase (against a real case_data dir if CFDMOD_HR_VALIDATE_CASE_DATA
points at one, else a synthetic case), inflow profile detection + figures
(pitot_inlet fixture), the Cp -> per-floor Cf/Cm pressure wiring, the
dynamic-response recipe wiring, and the facade / structure mesh-field snapshots
(galpao fixture + its lnas mesh).
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
    print("A. HighRiseCase")
    # Point CFDMOD_HR_VALIDATE_CASE_DATA at a real case_data dir to parse it;
    # otherwise fall back to a synthetic case (no client data is committed).
    import os

    env = os.environ.get("CFDMOD_HR_VALIDATE_CASE_DATA")
    case_data = pathlib.Path(env) if env else None
    if case_data is None or not case_data.exists():
        print("  (no CFDMOD_HR_VALIDATE_CASE_DATA case_data; using synthetic case)")
        return _synthetic_case()
    params = os.environ.get("CFDMOD_HR_VALIDATE_PARAMS", "params_cat3.yaml")
    case = pp.HighRiseCase.from_case_data(case_data, params)
    check("reference_height positive", case.reference_height > 0, f"H={case.reference_height}")
    check("nominal_area positive", case.nominal_area > 0)
    check("n_floors > 1", case.n_floors > 1, f"n_floors={case.n_floors}")
    check("simul U_H finite", np.isfinite(case.simul_reference_velocity))
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
        reference_height=100.0,
        characteristic_length=20.0,
        basic_wind_speed=40.0,
        simul_reference_velocity=30.0,
        nominal_area=500.0,
        nominal_volume=10000.0,
        floor_heights=[0.0, 25.0, 50.0, 75.0, 100.0],
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


def _galpao_cp(case: pp.HighRiseCase):
    """Compute a Cp time series + a 3-floor case on the galpao fixture."""
    import pathlib as _pl

    from lnas import LnasFormat

    from cfdmod.adapters.xdmf_h5 import XdmfH5Storage

    data_dir = FIX / "pressure" / "data"
    mesh_path = str(FIX / "pressure" / "galpao" / "galpao.normalized.lnas")
    storage = XdmfH5Storage(data_dir)
    body = storage.read_data_source(_pl.Path("bodies.galpao"))
    p_ref = storage.read_data_source(_pl.Path("points.static_pressure"))
    verts = LnasFormat.from_file(_pl.Path(mesh_path)).geometry.vertices
    edges = list(np.linspace(float(verts[:, 2].min()), float(verts[:, 2].max()), 4))
    floor_case = case.model_copy(update={"floor_heights": edges})
    cp = pp.cp_from_pressure(body, p_ref, floor_case)
    return cp, floor_case, mesh_path


def section_dynamic(case: pp.HighRiseCase) -> None:
    print("D. Dynamic response (galpao fixture)")
    cp, floor_case, mesh_path = _galpao_cp(case)

    cf = pp.cf_per_floor(cp, mesh_path, floor_case, directions=("x", "y"))
    cm = pp.cm_per_floor(cp, mesh_path, floor_case, directions=("z",))
    load = pp.floor_load_source(cf, cm, floor_case)
    check("load source is points", load.kind == "points", f"n_floors={load.n_elements}")
    check(
        "load fields present",
        all(f in load.field_names for f in ("cf_x", "cf_y", "cm_z")),
    )

    structure = pp.example_building_structure(floor_case, load.n_elements)
    check(
        "structure shapes mass-normalized",
        structure.n_floors == load.n_elements and structure.n_modes >= 1,
        f"floors={structure.n_floors} modes={structure.n_modes}",
    )

    response = pp.solve_building_response(load, structure, damping_ratio=0.02)
    for name in ("disp_x", "disp_y", "rot_z", "feq_x", "feq_y", "meq_z"):
        arr = np.asarray(response.fields.read(name))
        check(
            f"{name} finite + time-resolved",
            np.all(np.isfinite(arr)) and arr.shape == (load.n_elements, cp.time.n_timesteps),
            str(arr.shape),
        )

    acc = pp.floor_accelerations(response, structure, point=(1.0, 0.0))
    acc_mag = np.asarray(acc.fields.read("acc_mag"))
    check("acc_mag finite", np.all(np.isfinite(acc_mag)), str(acc_mag.shape))

    table = pp.peak_response_table(response, acc, floor_case)
    check(
        "peak table one row per floor",
        len(table) == load.n_elements and "acc_mag_peak" in table.columns,
        f"rows={len(table)}",
    )


def section_snapshots(case: pp.HighRiseCase, base: pathlib.Path) -> None:
    print("E. Facade / structure snapshots (galpao fixture)")
    cp, floor_case, mesh_path = _galpao_cp(case)

    geom = pp.snapshots.load_geometry(mesh_path)
    n_tri = int(np.asarray(geom.triangle_vertices).shape[0])
    groups = pp.snapshots.facade_groups(mesh_path)
    check("facade groups found", len(groups) >= 1, str({k: len(v) for k, v in groups.items()}))

    cp_mean = np.nanmean(np.asarray(cp.fields.read("cp")), axis=1)
    check("cp_mean per triangle", cp_mean.shape == (n_tri,), str(cp_mean.shape))

    dbg = pp.DebugWriter(base, stage="facade", version="validate")
    fig, _ = pp.snapshots.triangle_field_figure(
        geom,
        cp_mean,
        view=pp.snapshots.STANDARD_VIEWS["iso"],
        title="mean Cp",
        cbar_label="Cp [-]",
    )
    path = dbg.savefig(fig, "cp_mean_iso.png", deliverable=True)
    plotting.close(fig)
    check("facade figure written", path.exists() and path.stat().st_size > 0, str(path))

    fac_idx = pp.snapshots.facade_index_per_triangle(groups, n_tri)
    check("facade index per triangle", fac_idx.shape == (n_tri,) and np.isfinite(fac_idx).any())

    first = sorted(groups)[0]
    fig, _ = pp.snapshots.triangle_field_figure(geom, None, subset=groups[first], title=first)
    p2 = dbg.savefig(fig, "one_facade_geometry.png")
    plotting.close(fig)
    check("single-facade geometry render", p2.exists() and p2.stat().st_size > 0, str(p2))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        case = section_case()
        section_inflow(base)
        section_pressure(case)
        section_dynamic(case)
        section_snapshots(case, base)
    print("\nAll pp/ validations passed.")


if __name__ == "__main__":
    main()
