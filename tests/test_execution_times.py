from collections import deque
from collections import deque
from typing import Any
from unittest import TestCase

import numpy as np
import pandas as pd

from edgedroid.data import load_default_exec_time_data
from edgedroid.execution_times import Binner, ExecutionTimeModelFactory, \
    ModelException, _EmpiricalExecutionTimeModel, \
    _TheoreticalExecutionTimeModel, \
    _process_impairment, preprocess_data


class TestDataPreprocessing(TestCase):
    def test_impairment_chunks(self) -> None:
        # expected behavior:
        # the starting step of a task has previous impairment 0
        # the following steps have transitions depending on the impairment of
        # the starting step
        # if the starting step had a low impairment, transition is 0
        # however, if starting step had a high impairment, transition is 1
        # (i.e. we consider the step before the first step to have low
        # impairment)

        impairment_test_low = {
            'impairment'     : np.array([
                0, 0, 2, 2, 2, 0, 0, 3, 0, 0, 1, 1
            ]),
            'proc_impairment': np.array([
                0, 0, 0, 2, 2, 2, 0, 0, 3, 0, 0, 1
            ]),
            'transition'     : np.array([
                0, 0, 0, 1, 1, 1, -1, -1, 1, -1, -1, 1
            ]),
            'duration'       : np.array([
                -1, 1, 2, 1, 2, 3, 1, 2, 1, 1, 2, 1
            ])
        }

        impairment_test_high = {
            'impairment'     : np.array([
                2, 2, 1, 1, 1, 0, 0, 3, 0, 0, 1, 1
            ]),
            'proc_impairment': np.array([
                0, 2, 2, 1, 1, 1, 0, 0, 3, 0, 0, 1
            ]),
            'transition'     : np.array([
                0, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1
            ]),
            'duration'       : np.array([
                -1, 1, 2, 1, 2, 3, 1, 2, 1, 1, 2, 1
            ])
        }

        for imp_test in (impairment_test_low, impairment_test_high):
            proc_imp, prev_trans, prev_dur = _process_impairment(
                impairment=imp_test['impairment'])

            np.testing.assert_array_equal(
                imp_test['proc_impairment'],
                proc_imp,
                err_msg=f'Processed impairments don\'t'
                        f' match!\n{imp_test["impairment"]=}'
            )

            np.testing.assert_array_equal(
                imp_test['transition'],
                prev_trans,
                err_msg=f'Processed transitions don\'t'
                        f' match!\n{imp_test["impairment"]=}'
            )

            np.testing.assert_array_equal(
                imp_test['duration'],
                prev_dur,
                err_msg=f'Processed durations don\'t'
                        f' match!\n{imp_test["impairment"]=}'
            )

    def test_preprocess_data(self) -> None:
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
            'prev_impairment': [0, 1, 0, 0, 0, 2],
            'prev_duration'  : [-1, 0, 0, 0, 1, 0],
            'transition'     : [0, 1, -1, -1, -1, 1]
        })

        neuro_bins = np.array([-np.inf, 1, 2, np.inf])
        delay_bins = np.array([-np.inf, 1, 2, np.inf])
        duration_bins = np.array([0, 2, np.inf])

        test_data = test_data.set_index(['run_id', 'step_seq'],
                                        verify_integrity=True)
        exp_data = exp_data.set_index(['run_id', 'step_seq'],
                                      verify_integrity=True)

        result = preprocess_data(
            neuroticism_bins=neuro_bins,
            impairment_bins=delay_bins,
            duration_bins=duration_bins,
            execution_time_data=test_data
        )

        pd.testing.assert_frame_equal(result.data, exp_data,
                                      check_dtype=False)

        # very simple, multi-run test
        test_data_d = deque()
        exp_data_d = deque()
        for i in range(40):
            test_df = test_data.copy().reset_index()
            test_df['run_id'] = i
            test_data_d.append(test_df)

            exp_df = exp_data.copy().reset_index()
            exp_df['run_id'] = i
            exp_data_d.append(exp_df)

        test_data = pd.concat(test_data_d, ignore_index=True) \
            .set_index(['run_id', 'step_seq'], verify_integrity=True)
        exp_data = pd.concat(exp_data_d, ignore_index=True) \
            .set_index(['run_id', 'step_seq'], verify_integrity=True)

        result = preprocess_data(
            neuroticism_bins=neuro_bins,
            impairment_bins=delay_bins,
            duration_bins=duration_bins,
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
        cls.run_ids = raw_data.index.get_level_values(0).unique()

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

    def test_base_case_empirical(self):
        # base case test
        # 1 participant, 1 neuroticism values
        # all bins are single
        # all data extracted from the model should come from the participant

        data = self.raw_data.xs(self.rng.choice(self.run_ids),
                                drop_level=False).copy()
        self.assertEqual(len(data.neuroticism.unique()), 1)

        proc_data = preprocess_data(
            execution_time_data=data,
            neuroticism_bins=np.array([-np.inf, np.inf]),
            impairment_bins=np.array([-np.inf, np.inf]),
            duration_bins=np.array([0, np.inf])
        )
        proc_df = proc_data.data

        np.testing.assert_array_equal(proc_df.neuroticism.unique(), [0])
        np.testing.assert_array_equal(proc_df.prev_impairment.unique(), [0])
        np.testing.assert_array_equal(proc_df.prev_duration.unique(), [-1, 0])
        np.testing.assert_array_equal(proc_df.transition.unique(), [0])

        model = _EmpiricalExecutionTimeModel(
            data=proc_df,
            neuro_level=0,
            impair_binner=proc_data.impairment_binner,
            dur_binner=proc_data.duration_binner
        )

        # sample for first step should correspond to value for initial row
        np.testing.assert_almost_equal(model.get_initial_step_execution_time(),
                                       data.exec_time.values[0])

        # delay for initial step
        delay = data.delay.values[0]

        # execution times obtained after the first step
        # should always be taken from the collection of steps where N > 1
        data = data.iloc[1:]
        for step in data.itertuples(name='Step'):
            exec_time = model.get_execution_time(delay)  # use previous delay
            self.assertTrue(np.any(
                np.isclose(exec_time, data.exec_time.values)))

            # update delay at end of loop, always
            delay = step.delay

    def test_duration_binning_empirical(self):
        # duration binning test
        # 1 participant, 1 neuroticism level
        # all bins except duration are single
        # duration is binned such that each step gets its own bin
        # all data extracted from the model should come from the participant,
        # and each sampled execution time should match the obtained execution
        # time from the experiments

        # grab a random participant
        data = self.raw_data.xs(self.rng.choice(self.run_ids),
                                drop_level=False).copy()
        self.assertEqual(len(data.neuroticism.unique()), 1)

        # all bins are single EXCEPT DURATION
        # duration bins are set up so that each step gets it's OWN duration lvl
        # this way we avoid randomness when sampling
        num_steps = len(data.index)
        durations = np.arange(1, num_steps)

        # add 0.1 tolerance, to make sure each whole number actually falls
        # within a bin
        dur_bins = np.concatenate(([0], durations, [np.inf])) + 0.1

        proc_data = preprocess_data(
            execution_time_data=data,
            neuroticism_bins=np.array([-np.inf, np.inf]),
            impairment_bins=np.array([-np.inf, np.inf]),
            duration_bins=dur_bins
        )
        proc_df = proc_data.data

        np.testing.assert_array_equal(proc_df.neuroticism.unique(), [0])
        np.testing.assert_array_equal(proc_df.prev_impairment.unique(), [0])
        np.testing.assert_array_equal(proc_df.transition.unique(), [0])
        np.testing.assert_array_equal(proc_df.prev_duration.unique(),
                                      np.concatenate(([-1],
                                                      np.arange(0, 167))))

        model = _EmpiricalExecutionTimeModel(
            data=proc_df,
            neuro_level=0,
            impair_binner=proc_data.impairment_binner,
            dur_binner=proc_data.duration_binner
        )

        prev_step: Any = None
        for i, step in enumerate(data.itertuples(name='Step')):
            if i == 0:
                # sample for first step should
                # correspond to value for initial row
                np.testing.assert_almost_equal(
                    model.get_initial_step_execution_time(),
                    data.exec_time.values[i])
            else:
                # use previous delay
                exec_time = model.get_execution_time(prev_step.delay)

                # since duration is binned in such a way that each steps gets
                # its own bin, the sampled value here should always
                # correspond to the execution time of the current step
                self.assertTrue(np.isclose(
                    exec_time,
                    step.exec_time
                ))

            # save step
            prev_step = step

    def test_impairment_transition_empirical(self):
        # impairment transition test
        # all bins are single, except impairment
        # delay is binned into two levels (i.e. below and above a threshold)
        # at each step, the execution time then comes from the impairment of
        # the previous step and the most recent transition!

        # this test needs specially designed data
        # we need to test low-to-high transitions and high-to-low transitions
        # this means at least two different delays.
        # everything else is factored away, so we only need three steps
        # first step always has a different distribution
        # we set first step impairment to low
        # second step for the the low-to-high transitions and high impairment
        # third step for the high-to-low transition
        test_data = pd.DataFrame({
            'run_id'     : [0, 0, 0],
            'step_seq'   : [1, 2, 3],
            'exec_time'  : [1, 2, 3],
            'neuroticism': [0, 0, 0],
            'delay'      : [2, 1, 2]
        }).set_index(['run_id', 'step_seq'], verify_integrity=True)

        delay_bins = np.array([-np.inf, 1.5, np.inf])
        proc_data = preprocess_data(
            execution_time_data=test_data,
            neuroticism_bins=np.array([-np.inf, np.inf]),
            impairment_bins=delay_bins,
            duration_bins=np.array([0, np.inf])
        )
        proc_df = proc_data.data

        np.testing.assert_array_equal(proc_df.neuroticism.unique(), [0])
        np.testing.assert_array_equal(proc_df.prev_impairment.unique(), [0, 1])
        self.assertSetEqual(set(proc_df.transition.unique()), {-1, 0, 1})
        np.testing.assert_array_equal(proc_df.prev_duration.unique(), [-1, 0])

        model = _EmpiricalExecutionTimeModel(
            data=proc_df,
            neuro_level=0,
            impair_binner=proc_data.impairment_binner,
            dur_binner=proc_data.duration_binner
        )

        # step 1 should have a different distribution
        # sample for first step should correspond to value for initial row
        np.testing.assert_almost_equal(model.get_initial_step_execution_time(),
                                       test_data.exec_time.values[0])
        step1_delay = test_data.delay.values[0]

        # step 2 should correspond to the exec time for the second row
        step2_etime = model.get_execution_time(delay=step1_delay)
        np.testing.assert_almost_equal(step2_etime,
                                       test_data.exec_time.values[1])
        step3_delay = test_data.delay.values[1]

        # step 3
        step3_etime = model.get_execution_time(delay=step3_delay)
        np.testing.assert_almost_equal(step3_etime,
                                       test_data.exec_time.values[2])
