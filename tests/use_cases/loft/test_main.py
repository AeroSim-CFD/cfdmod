import unittest

from cfdmod.use_cases.loft.main import main


class TestLoftMain(unittest.TestCase):
    def test_main(self):
        config = "./fixtures/tests/loft/loft_params.yaml"
        surface = "./fixtures/tests/loft/terrain.stl"
        output = "./fixtures/tests/loft"

        main(["--config", config, "--surface", surface, "--output", output])


if __name__ == "__main__":
    unittest.main()
