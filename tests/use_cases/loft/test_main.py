from cfdmod.logger import logger
from cfdmod.use_cases.loft.main import main


def test_main():
    config = "./fixtures/tests/loft/loft_params.yaml"
    surface = "./fixtures/tests/loft/terrain.stl"
    output = "./output/loft"

    logger.disabled = True
    main(["--config", config, "--surface", surface, "--output", output])
    logger.disabled = False
