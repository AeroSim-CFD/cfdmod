import hashlib
import pathlib

from pydantic import BaseModel


class HashableConfig(BaseModel):
    def sha256(self) -> str:
        """Hash config dict and returns a string with the hash hexcode

        Returns:
            str: Config data object hash
        """
        hash_sha256 = hashlib.sha256()
        hash_sha256.update(self.model_dump_json().encode("utf-8"))

        return hash_sha256.hexdigest()

    def to_dict(self) -> dict:
        """Serialize configuration to a plain Python dictionary.

        Returns:
            dict: Dictionary representation of the configuration.
        """
        return self.model_dump()

    def to_yaml(self, path: pathlib.Path):
        """Serialize configuration to a YAML file.

        Args:
            path (pathlib.Path): File path to write the YAML output.
        """
        from cfdmod.utils import save_yaml

        save_yaml(self.to_dict(), path)
