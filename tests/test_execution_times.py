from collections import deque
from typing import Any, Generator, Tuple
from unittest import TestCase

import numpy as np
import pandas as pd

from edgedroid.data import load_default_exec_time_data
from edgedroid.execution_times import Binner, ExecutionTimeModelFactory, \
    ModelException, _EmpiricalExecutionTimeModel, \
    _TheoreticalExecutionTimeModel, \
    _calculate_impairment_chunks, preprocess_data


class TestDataPreprocessing(TestCase):
    impairment_test = {
        'impairment'     : pd.Series([
            1, 1, 2, 2, 2, 0, 0, 3, 0, 0, 1, 1
        ], dtype=pd.CategoricalDtype(categories=[0, 1, 2, 3], ordered=True)),
        'prev_transition': np.array([
            0, 0, 0, 1, 1, 1, -1, -1, 1, -1, -1, 1
        ]),
        'prev_duration'  : np.array([
            np.nan, 1, 2, 1, 2, 3, 1, 2, 1, 1, 2, 1
        ])
    }

    # test_data = pd.DataFrame({
    #     'run_id'     : [1, 1, 2, 2, 2, 3, 3],
    #     'neuroticism': [1.5, 1.5, 0.5, 0.5, 0.5, 2.5, 2.5],
    #     'step_seq'   : [1, 2, 1, 2, 3, 1, 2],
    #     'exec_time'  : [1, 1, 1, 2, 3, 1, 2],
    #     'delay'      : [0.5, 1.5, 2.5, 1.5, 2.5, 0.5, 0.5],
    # })

    test_data = pd.DataFrame({
        'run_id'     : [1, 1, 1, 1, 1, 1],
        'neuroticism': [0.5, 1.5, 2.5, 0.5, 1.5, 2.5],
        'step_seq'   : [1, 2, 3, 4, 5, 6],
        'exec_time'  : [1, 1, 2, 2, 3, 3],
        'delay'      : [1.5, 0.5, 0.5, 0.5, 2.5, 2.5],
    })

    exp_data = pd.DataFrame({
        'run_id'         : [1, 1, 1, 1, 1, 1],
        'step_seq'       : [1, 2, 3, 4, 5, 6],
        'exec_time'      : [1, 1, 2, 2, 3, 3],
        'neuroticism'    : [0, 1, 2, 0, 1, 2],
        'prev_impairment': [-1, 1, 0, 0, 0, 2],
        'prev_duration'  : [-1, 0, 0, 0, 1, 0],
        'transition'     : [0, 0, -1, -1, -1, 1]
    })

    neuro_bins = np.array([-np.inf, 1, 2, np.inf])
    delay_bins = np.array([-np.inf, 1, 2, np.inf])
    duration_bins = np.array([-np.inf, 2, np.inf])

    def test_impairment_chunks(self) -> None:
        prev_trans, prev_dur = _calculate_impairment_chunks(
            impairment=self.impairment_test['impairment'])

        np.testing.assert_array_equal(
            self.impairment_test['prev_transition'],
            prev_trans
        )
        np.testing.assert_array_equal(
            self.impairment_test['prev_duration'],
            prev_dur
        )

    def test_preprocess_data(self) -> None:
        test_data = self.test_data.set_index(['run_id', 'step_seq'],
                                             verify_integrity=True)
        exp_data = self.exp_data.set_index(['run_id', 'step_seq'],
                                           verify_integrity=True)

        result = preprocess_data(
            neuroticism_bins=self.neuro_bins,
            impairment_bins=self.delay_bins,
            duration_bins=self.duration_bins,
            execution_time_data=test_data
        )

        pd.testing.assert_frame_equal(result.data, exp_data,
                                      check_dtype=False)

        # very simple, multi-run test
        test_data = deque()
        exp_data = deque()
        for i in range(40):
            test_df = self.test_data.copy()
            test_df['run_id'] = i
            test_data.append(test_df)

            exp_df = self.exp_data.copy()
            exp_df['run_id'] = i
            exp_data.append(exp_df)

        test_data = pd.concat(test_data, ignore_index=True) \
            .set_index(['run_id', 'step_seq'], verify_integrity=True)
        exp_data = pd.concat(exp_data, ignore_index=True) \
            .set_index(['run_id', 'step_seq'], verify_integrity=True)

        result = preprocess_data(
            neuroticism_bins=self.neuro_bins,
            impairment_bins=self.delay_bins,
            duration_bins=self.duration_bins,
            execution_time_data=test_data
        )

        pd.testing.assert_frame_equal(result.data, exp_data,
                                      check_dtype=False)

    def test_binner(self) -> None:
        # test the Binner class
        # two bins: [1, 2) and [2, 3)
        bins = [1, 2, 3]

        binner = Binner(bin_edges=bins)

        with self.assertRaises(Binner.BinningError):
            # falls outside of left edge
            binner.bin(0)

        with self.assertRaises(Binner.BinningError):
            # falls outside of right edge
            binner.bin(4)

        with self.assertRaises(Binner.BinningError):
            # ranges are left-exclusive
            binner.bin(1)

        self.assertEqual(binner.bin(1.5), 0)
        self.assertEqual(binner.bin(2.7), 1)
        self.assertEqual(binner.bin(2), 0)
        self.assertEqual(binner.bin(3), 1)

        # now with some arrays
        with self.assertRaises(Binner.BinningError):
            binner.bin(np.linspace(start=0, stop=1, num=50))

        with self.assertRaises(Binner.BinningError):
            binner.bin(np.linspace(start=3.1, stop=10, num=50))

        np.testing.assert_array_equal(
            binner.bin([1.5, 2.5, 3]),
            [0, 1, 1]
        )


class TestModels(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rng = np.random.default_rng()

        raw_data = load_default_exec_time_data()
        cls.proc_data = preprocess_data(execution_time_data=raw_data)
        cls.raw_data = raw_data

    def test_model_factory(self) -> None:
        factory = ExecutionTimeModelFactory()

        emp_model = factory.make_model(neuroticism=0.5, empirical=True)
        theo_model = factory.make_model(neuroticism=0.5, empirical=False)

        self.assertIsInstance(emp_model, _EmpiricalExecutionTimeModel)
        self.assertIsInstance(theo_model, _TheoreticalExecutionTimeModel)

    def test_model_iterator(self) -> None:
        # tests for the model iterator util
        # data = preprocess_data()

        for model_cls in (_EmpiricalExecutionTimeModel,
                          _TheoreticalExecutionTimeModel):
            model = model_cls(
                data=self.proc_data.data.copy(),
                neuro_level=0,
                impair_binner=self.proc_data.impairment_binner,
                dur_binner=self.proc_data.duration_binner
            )

            it = model.execution_time_iterator()
            value = next(it)
            self.assertIsInstance(value, (float, int))
            with self.assertRaises(ModelException):
                next(it)  # exception should be raised here, on the second call

            # if we call set_delay() between iterations, no problem
            it.set_delay(5)
            for i, _ in enumerate(it):
                if i > 10:
                    break
                it.set_delay(10)

    def _model_test_generator(self) \
            -> Generator[Tuple[Any, pd.DataFrame, pd.DataFrame], None, None]:
        model_data = self.proc_data.data

        # run tests with three different, random subjects
        run_ids = model_data.index.get_level_values(0)
        run_ids = self.rng.choice(run_ids, size=3)

        # run tests for each subject (run_id)
        for run_id in run_ids:
            run_raw_data = self.raw_data.xs(run_id, drop_level=False).copy()
            run_model_data = model_data.xs(run_id, drop_level=False).copy()

            yield run_id, run_raw_data, run_model_data

    def test_empirical_model(self) -> None:
        # we use the default data to test
        # raw_data = load_default_exec_time_data()
        # proc_data = preprocess_data(execution_time_data=raw_data)

        # model_data = self.proc_data.data
        #
        # # run tests with three different, random subjects
        # run_ids = model_data.index.get_level_values(0)
        # run_ids = self.rng.choice(run_ids, size=3)
        #
        # # run tests for each subject (run_id)
        # for run_id in run_ids:
        #     run_raw_data = self.raw_data.xs(run_id, drop_level=False).copy()
        #     run_model_data = model_data.xs(run_id, drop_level=False).copy()

        for run_id, run_raw_data, run_model_data in \
                self._model_test_generator():
            # create a model matching the selected run_id
            neuro = run_raw_data.neuroticism.iloc[0]
            neuro_level = self.proc_data.neuroticism_binner.bin(neuro)

            model = _EmpiricalExecutionTimeModel(
                data=run_model_data,  # use only data from this participant
                neuro_level=neuro_level,
                impair_binner=self.proc_data.impairment_binner,
                dur_binner=self.proc_data.duration_binner
            )

            # generate an execution time for each step of the original run,
            # then compare obtained distributions for similarity
            model_iter = model.execution_time_iterator()
            model_etimes = np.empty(shape=len(run_raw_data.index))

            for i, (row, exec_time) in \
                    enumerate(zip(run_raw_data.itertuples(name='Step'),
                                  model_iter)):
                # outputs of empirical model should all be in the original data
                self.assertTrue(np.any(np.isclose(exec_time,
                                                  run_raw_data.exec_time)))

                model_etimes[i] = exec_time
                model_iter.set_delay(row.delay)

                # TODO: more complex test?

    def test_theoretical_model(self) -> None:
        pass
