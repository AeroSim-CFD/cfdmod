import unittest

from cfdmod.logger import logger
from cfdmod.use_cases.pressure.shape.main import main


class TestCeMain(unittest.TestCase):
    def test_main(self):
        output = "./output/pressure"
        cp = "./fixtures/tests/pressure/data/cp_t.resampled.hdf"
        config = "./fixtures/tests/pressure/Ce_params.yaml"
        mesh = "./fixtures/tests/pressure/galpao/galpao.normalized.lnas"

        logger.disabled = True
        main(["--output", output, "--cp", cp, "--config", config, "--mesh", mesh])
        logger.disabled = False


if __name__ == "__main__":
    unittest.main()
