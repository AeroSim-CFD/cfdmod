import pathlib

from cfdmod.roughness.parameters import GenerationParams, RadialParams
from cfdmod.roughness.run import run_linear, run_radial


def test_run_linear():
    config = pathlib.Path("./fixtures/tests/roughness_gen/roughness_params.yaml")
    output = pathlib.Path("./output/roughness_gen_run/linear")

    cfg = GenerationParams.from_file(config)
    run_linear(cfg, output)

    assert (output / "roughness_elements.stl").exists()


def test_run_radial():
    config = pathlib.Path("./fixtures/tests/roughness_gen/radial_params.yaml")
    output = pathlib.Path("./output/roughness_gen_run/radial")

    cfg = RadialParams.from_file(config)
    run_radial(cfg, output)

    assert (output / "roughness_elements.stl").exists()
