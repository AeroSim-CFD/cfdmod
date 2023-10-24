import unittest

from cfdmod.use_cases.pressure.main import main


class TestCeMain(unittest.TestCase):
    def test_main(self):
        output = "./output/pressure"
        p = "./fixtures/tests/pressure/data/bodies.galpao.data.resampled.h5"
        s = "./fixtures/tests/pressure/data/points.static_pressure.data.resampled.h5"
        config = "./fixtures/tests/pressure/cp_params.yaml"
        mesh = "./fixtures/tests/pressure/galpao/galpao.lnas"

        main(["--output", output, "--p", p, "--s", s, "--config", config, "--mesh", mesh])


if __name__ == "__main__":
    unittest.main()
