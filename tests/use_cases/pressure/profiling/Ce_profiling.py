from cfdmod.use_cases.pressure.shape.main import main as main_Ce


def main_Ce_profiling():
    output = "./output/profiling"
    cp = "./fixtures/tests/pressure/data/cp_t.grouped.h5"
    config = "./fixtures/tests/pressure/Ce_params.yaml"
    mesh = "./fixtures/tests/pressure/galpao/galpao.normalized.lnas"

    main_Ce(["--output", output, "--cp", cp, "--config", config, "--mesh", mesh])


if __name__ == "__main__":
    main_Ce_profiling()
