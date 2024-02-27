from cfdmod.use_cases.pressure.force.main import main as main_Cf


def main_Cf_profiling():
    output = "./output/profiling"
    cp = "./fixtures/tests/pressure/data/cp_t.grouped.h5"
    config = "./fixtures/tests/pressure/Cf_params.yaml"
    mesh = "./fixtures/tests/pressure/galpao/galpao.normalized.lnas"

    main_Cf(["--output", output, "--cp", cp, "--config", config, "--mesh", mesh])


if __name__ == "__main__":
    main_Cf_profiling()
