from cfdmod.logger import logger
from cfdmod.use_cases.pressure.main import main


def test_main():
    output = "./output/pressure"
    p = "./fixtures/tests/pressure/data/new.bodies.galpao.data.resampled.h5"
    s = "./fixtures/tests/pressure/data/new.points.static_pressure.data.resampled.h5"
    config = "./fixtures/tests/pressure/cp_params.yaml"
    mesh = "./fixtures/tests/pressure/galpao/galpao.lnas"

    logger.disabled = True
    main(["--output", output, "--p", p, "--s", s, "--config", config, "--mesh", mesh])
    logger.disabled = False
