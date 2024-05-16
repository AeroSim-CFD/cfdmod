import pathlib
import unittest

from cfdmod.logger import logger
from cfdmod.use_cases.pressure.main import main


class TestCpMain(unittest.TestCase):
    def test_undivided_legacy_format(self):
        output = "./output/pressure"
        p = "./fixtures/tests/pressure/data/bodies.galpao.data.resampled.h5"
        s = "./fixtures/tests/pressure/data/points.static_pressure.data.resampled.h5"
        config = "./fixtures/tests/pressure/cp_params.yaml"
        mesh = "./fixtures/tests/pressure/galpao/galpao.lnas"

        logger.disabled = True
        main(["--output", output, "--p", p, "--s", s, "--config", config, "--mesh", mesh])
        logger.disabled = False

    def test_main(self):
        output = "./output/pressure"
        p = "./fixtures/tests/pressure/data/new.bodies.galpao.data.resampled.h5"
        s = "./fixtures/tests/pressure/data/new.points.static_pressure.data.resampled.h5"
        config = "./fixtures/tests/pressure/cp_params.yaml"
        mesh = "./fixtures/tests/pressure/galpao/galpao.lnas"

        logger.disabled = True
        main(["--output", output, "--p", p, "--s", s, "--config", config, "--mesh", mesh])
        logger.disabled = False

    def test_legacy_format(self):
        output = "./output/pressure"
        p = "./fixtures/tests/pressure/data/bodies.galpao.data.grouped.h5"
        s = "./fixtures/tests/pressure/data/points.static_pressure.data.grouped.h5"
        config = "./fixtures/tests/pressure/cp_params.yaml"
        mesh = "./fixtures/tests/pressure/galpao/galpao.lnas"

        logger.disabled = True
        main(["--output", output, "--p", p, "--s", s, "--config", config, "--mesh", mesh])
        logger.disabled = False


if __name__ == "__main__":
    unittest.main()
