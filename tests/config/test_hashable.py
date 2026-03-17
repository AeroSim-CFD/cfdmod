import pathlib
import tempfile

from cfdmod.config.hashable import HashableConfig


def test_equal_hashes():
    class A(HashableConfig):
        a: str

    sha1 = A(a="a").sha256()
    sha2 = A(a="a").sha256()

    assert sha1 == sha2


def test_hashing_order():
    class A(HashableConfig):
        a: str
        b: str

    sha1 = A(a="a", b="b").sha256()
    sha2 = A(b="b", a="a").sha256()

    assert sha1 == sha2


def test_non_equal_hashes():
    class A(HashableConfig):
        a: str

    sha1 = A(a="a").sha256()
    sha2 = A(a="b").sha256()

    assert sha1 != sha2


def test_to_dict():
    class A(HashableConfig):
        x: int
        y: str

    instance = A(x=42, y="hello")
    result = instance.to_dict()

    assert result == {"x": 42, "y": "hello"}


def test_to_dict_nested():
    class Inner(HashableConfig):
        value: float

    class Outer(HashableConfig):
        name: str
        inner: Inner

    instance = Outer(name="test", inner=Inner(value=3.14))
    result = instance.to_dict()

    assert result == {"name": "test", "inner": {"value": 3.14}}


def test_to_yaml_roundtrip():
    class A(HashableConfig):
        x: int
        y: str

        @classmethod
        def from_file(cls, path: pathlib.Path):
            from cfdmod.utils import read_yaml
            return cls(**read_yaml(path))

    instance = A(x=10, y="world")

    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = pathlib.Path(tmp) / "config.yaml"
        instance.to_yaml(yaml_path)

        assert yaml_path.exists()

        loaded = A.from_file(yaml_path)
        assert loaded.x == instance.x
        assert loaded.y == instance.y
