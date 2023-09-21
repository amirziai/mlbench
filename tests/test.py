import unittest

from mlbench.utils import seq


class TestUtils(unittest.TestCase):
    def test_seq(self):
        assert seq(1) == [1]
        assert seq([1]) == [1]
        assert seq([1, 2]) == [1, 2]
        assert seq("xy") == ["xy"]
        assert seq(["x", "y"]) == ["x", "y"]
