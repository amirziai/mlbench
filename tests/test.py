import unittest

import numpy as np

from mlbench import eval_binary, ap
from mlbench.utils import seq, seq_of_seq


class TestUtils(unittest.TestCase):
    def test_seq(self):
        assert seq(1) == [1]
        assert seq([1]) == [1]
        assert seq([1, 2]) == [1, 2]
        assert seq("xy") == ["xy"]
        assert seq(["x", "y"]) == ["x", "y"]
        assert seq([]) == []
        x = np.array([True, False])
        assert np.array_equal(seq(x), x)

    def test_seq_of_seq(self):
        assert seq_of_seq([1]) == [[1]]
        assert seq_of_seq([[1]]) == [[1]]
        assert seq_of_seq([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]

    def test_eval(self):
        m = eval_binary(
            y_true=[True, False],
            y_pred=[True, False],
        )
        assert m.value_point_estimate == 1
        assert m.beat_baseline
        assert not m.is_stat_sig
        m = eval_binary(
            y_true=[True, False],
            y_pred=[[True, False]] * 10,
        )
        assert m.value_point_estimate == 1
        assert m.beat_baseline
        assert m.is_stat_sig
        m = eval_binary(
            y_true=[False, True],
            y_pred=[[True, False]] * 10,
        )
        assert m.value_point_estimate == 0
        assert not m.beat_baseline
        assert not m.is_stat_sig
        assert m.experiment_cnt == 10
        m = eval_binary(
            y_true=[[False, True], [True, False]],
            y_pred=[[True, False]] * 2,
        )
        assert m.value_point_estimate == 0.5
        assert not m.beat_baseline
        assert not m.is_stat_sig
        assert m.experiment_cnt == 2
        assert m.sample_cnt == 2

    def test_ap(self):
        assert ap.expected_average_precision(n_=10_000, p_=5_000) == 0.5
        assert ap.expected_average_precision(n_=2, p_=1) == 0.75
        assert ap.minimum_average_precision(n_=2, p_=1) == 0.5
        assert ap.minimum_average_precision(n_=4, p_=1) == 0.25
