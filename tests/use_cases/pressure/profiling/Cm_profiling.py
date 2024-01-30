from memory_profiler import profile

from cfdmod.use_cases.pressure.moment.main import main as main_Cm


@profile
def main_Cm_profiling():
    output = "./output/profiling"
    cp = "./fixtures/tests/pressure/data/cp_t.grouped.h5"
    config = "./fixtures/tests/pressure/Cm_params.yaml"
    mesh = "./fixtures/tests/pressure/galpao/galpao.normalized.lnas"

    main_Cm(["--output", output, "--cp", cp, "--config", config, "--mesh", mesh])


if __name__ == "__main__":
    main_Cm_profiling()
