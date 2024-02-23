from memory_profiler import profile

from cfdmod.use_cases.pressure.main import main as main_cp


@profile
def main_cp_profiling():
    output = "./output/profiling"
    # p = "./fixtures/tests/pressure/data/cp_t.grouped.h5"
    # s = "./fixtures/tests/pressure/data/cp_t.grouped.h5"
    # config = "./fixtures/tests/pressure/cp_params.yaml"
    # mesh = "./fixtures/tests/pressure/galpao/galpao.normalized.lnas"
    p = "/home/ubuntu/Documentos/Repositories/insight/Docker/local/volume/divided_cp/body_pressure_treated.h5"
    s = "/home/ubuntu/Documentos/Repositories/insight/Docker/local/volume/divided_cp/static_pressure_treated.h5"
    config = "/home/ubuntu/Documentos/Repositories/insight/Docker/local/volume/divided_cp/cp_params.yaml"
    mesh = "/home/ubuntu/Documentos/Repositories/insight/Docker/local/volume/divided_cp/G100.merged.lnas"

    main_cp(["--output", output, "--p", p, "--s", s, "--config", config, "--mesh", mesh])


if __name__ == "__main__":
    main_cp_profiling()
