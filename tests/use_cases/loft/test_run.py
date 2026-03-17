import pathlib

import lnas

from cfdmod.logger import logger
from cfdmod.use_cases.loft.parameters import LoftCaseConfig
from cfdmod.use_cases.loft.run import run_loft


def test_run_loft():
    config = pathlib.Path("./fixtures/tests/loft/loft_params.yaml")
    surface = pathlib.Path("./fixtures/tests/loft/terrain.stl")
    output = pathlib.Path("./output/loft_run")

    cfg = LoftCaseConfig.from_file(config)
    geom = lnas.LnasFormat.from_file(surface).geometry

    logger.disabled = True
    run_loft(cfg, geom, output)
    logger.disabled = False


def test_run_loft_complex():
    config = pathlib.Path("./fixtures/tests/loft/complex_loft_params.yaml")
    surface = pathlib.Path("./fixtures/tests/loft/complex_terrain.stl")
    output = pathlib.Path("./output/complex_loft_run")

    cfg = LoftCaseConfig.from_file(config)
    geom = lnas.LnasFormat.from_file(surface).geometry

    logger.disabled = True
    run_loft(cfg, geom, output)
    logger.disabled = False
