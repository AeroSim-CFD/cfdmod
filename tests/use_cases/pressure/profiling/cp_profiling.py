from cfdmod.use_cases.pressure.main import main as main_cp


# Install memory-profiler (pip install -U memory-profiler) and run this as 
# 
# mprof run poetry run python tests/use_cases/pressure/profiling/cp_profiling.py
# mprof plot
def main_cp_profiling():
    output = "./output/profiling"
    p = "./fixtures/tests/pressure/data/bodies.galpao.data.grouped.h5"
    s = "./fixtures/tests/pressure/data/points.static_pressure.data.grouped.h5"
    config = "./fixtures/tests/pressure/cp_params.yaml"
    mesh = "./fixtures/tests/pressure/galpao/galpao.normalized.lnas"

    main_cp(["--output", output, "--p", p, "--s", s, "--config", config, "--mesh", mesh])


if __name__ == "__main__":
    main_cp_profiling()
