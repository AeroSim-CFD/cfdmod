import pathlib
import unittest

from cfdmod.api.configs.hashable import HashableConfig


class TestHashableConfigs(unittest.TestCase):
    def test_equal_hashes(self):
        class A(HashableConfig):
            a: str

        sha1 = A(a="a").sha256()
        sha2 = A(a="a").sha256()

        self.assertEqual(sha1, sha2)

    def test_hashing_order(self):
        class A(HashableConfig):
            a: str
            b: str

        sha1 = A(a="a", b="b").sha256()
        sha2 = A(b="b", a="a").sha256()

        self.assertEqual(sha1, sha2)

    def test_non_equal_hashes(self):
        class A(HashableConfig):
            a: str

        sha1 = A(a="a").sha256()
        sha2 = A(a="b").sha256()

        self.assertNotEqual(sha1, sha2)


if __name__ == "__main__":
    unittest.main()
