from cfdmod.api.configs.hashable import HashableConfig


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
