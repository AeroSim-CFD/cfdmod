import hashlib

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
