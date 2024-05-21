import unittest

from cfdmod.logger import logger
from cfdmod.use_cases.loft.main import main


class TestLoftMain(unittest.TestCase):
    def test_main(self):
        config = "./fixtures/tests/loft/loft_params.yaml"
        surface = "./fixtures/tests/loft/terrain.stl"
        output = "./output/loft"

        logger.disabled = True
        main(["--config", config, "--surface", surface, "--output", output])
        logger.disabled = False


if __name__ == "__main__":
    unittest.main()
