from cfdmod.logger import logger
from cfdmod.use_cases.pressure.moment.main import main


def test_main():
    output = "./output/pressure"
    cp = "./fixtures/tests/pressure/data/cp_t.normalized.h5"
    config = "./fixtures/tests/pressure/Cm_params.yaml"
    mesh = "./fixtures/tests/pressure/galpao/galpao.normalized.lnas"

    logger.disabled = True
    main(["--output", output, "--cp", cp, "--config", config, "--mesh", mesh])
    logger.disabled = False
