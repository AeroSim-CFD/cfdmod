import json
from pathlib import Path

from json_schema_for_humans.generate import generate_from_filename

from cfdmod.use_cases.loft.parameters import LoftCaseConfig
from cfdmod.use_cases.pressure.cp_config import CpCaseConfig
from cfdmod.use_cases.pressure.force.Cf_config import CfCaseConfig
from cfdmod.use_cases.pressure.moment.Cm_config import CmCaseConfig
from cfdmod.use_cases.pressure.shape.Ce_config import CeCaseConfig
from cfdmod.use_cases.roughness_gen.parameters import GenerationParams, PositionParams


class GlobalSchema(
    CpCaseConfig,
    CeCaseConfig,
    CfCaseConfig,
    CmCaseConfig,
    LoftCaseConfig,
    GenerationParams,
    PositionParams,
): ...


def main():
    path = Path("./output")
    json_path = path / "schema.json"
    html_path = path / "schema.html"

    schema_dict = GlobalSchema.model_json_schema(by_alias=True, mode="serialization")
    schema_dict["required"] = []
    schema_dict["title"] = "Cfdmod config schema"

    schema = json.dumps(schema_dict, indent=2)
    cleaned_schema = (
        schema.replace("-Infinity", "PLACE_HOLDER_INF")
        .replace("Infinity", '"Infinity"')
        .replace("PLACE_HOLDER_INF", '"-Infinity"')
    )

    with open(json_path, "w") as f:
        f.write(cleaned_schema)

    generate_from_filename(
        schema_file_name=json_path,
        result_file_name=html_path,
        copy_css=True,
        copy_js=True,
        link_to_reused_ref=False,
    )


if __name__ == "__main__":
    main()
