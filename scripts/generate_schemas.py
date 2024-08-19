import json
from pathlib import Path

from json_schema_for_humans.generate import generate_from_filename
from pydantic import BaseModel

from cfdmod.use_cases.loft.parameters import LoftCaseConfig
from cfdmod.use_cases.pressure.cp_config import CpCaseConfig
from cfdmod.use_cases.pressure.force.Cf_config import CfCaseConfig
from cfdmod.use_cases.pressure.moment.Cm_config import CmCaseConfig
from cfdmod.use_cases.pressure.shape.Ce_config import CeCaseConfig
from cfdmod.use_cases.roughness_gen.parameters import GenerationParams, PositionParams


def main():
    path = Path("./output")
    json_path = path / "schema.json"
    html_path = path / "schema.html"

    schema_list: list[tuple[str, BaseModel]] = [
        ("CpCaseConfig", CpCaseConfig),
        ("CeCaseConfig", CeCaseConfig),
        ("CfCaseConfig", CfCaseConfig),
        ("CmCaseConfig", CmCaseConfig),
        ("LoftCaseConfig", LoftCaseConfig),
        ("GenerationParams", GenerationParams),
        ("PositionParams", PositionParams),
    ]
    schema_dict = {}

    for s_name, s_cls in schema_list:
        if len(schema_dict) != 0:
            s_schema = s_cls.model_json_schema(by_alias=True, mode="serialization")
            schema_dict["$defs"] = s_schema["$defs"] | schema_dict["$defs"]
            schema_dict["properties"] = s_schema["properties"] | schema_dict["properties"]
        else:
            schema_dict = s_cls.model_json_schema(by_alias=True, mode="serialization")

    schema_dict["required"] = []
    schema_dict["title"] = "Cfdmod config schema"

    schema = json.dumps(schema_dict, indent=2)
    with open(json_path, "w") as f:
        f.write(schema)

    generate_from_filename(
        schema_file_name=json_path,
        result_file_name=html_path,
        copy_css=True,
        copy_js=True,
        link_to_reused_ref=False,
    )


if __name__ == "__main__":
    main()
