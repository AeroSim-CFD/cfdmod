from cfdmod.logger import logger
from cfdmod.use_cases.loft.main import main


def test_terrain():
    config = "./fixtures/tests/loft/loft_params.yaml"
    surface = "./fixtures/tests/loft/terrain.stl"
    output = "./output/loft"

    logger.disabled = True
    main(["--config", config, "--surface", surface, "--output", output])
    logger.disabled = False


def test_complex_terrain():
    config = "./fixtures/tests/loft/complex_loft_params.yaml"
    surface = "./fixtures/tests/loft/complex_terrain.stl"
    output = "./output/complex_loft"

    logger.disabled = True
    main(["--config", config, "--surface", surface, "--output", output])
    logger.disabled = False