from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque

import cv2
import nptyping as npt
import numpy as np
import pandas as pd
from gabriel_lego import FrameResult, LEGOTask

import edgedroid.data as e_data
from edgedroid.execution_times import TheoreticalExecutionTimeModel, \
    preprocess_data
from edgedroid.frames import FrameModel, FrameSet


def processing_thread_loop(ui_input_q: Deque,
                           ui_guidance_q: Deque,
                           done_flag: threading.Event):
    task = pd.read_csv('./latin_sqr_0.csv')
    states = [np.array(eval(s), dtype=int) for s in task['state']]
    # noinspection PyTypeChecker
    lego_task = LEGOTask(states)

    probs = pd.read_csv('edgedroid/data/resources/frame_probabilitites.csv')
    frame_model = FrameModel(probs)
    frameset = FrameSet.from_datafile('Square0', './square00.npz')

    data = preprocess_data(*e_data.load_default_exec_time_data())
    timing_model = TheoreticalExecutionTimeModel(data=data, neuroticism=0.5)

    previous_success = frameset.get_initial_frame()
    previous_t = time.monotonic()
    dt = 0

    rng = np.random.default_rng()

    try:
        # submit initial frame to start task
        guidance = lego_task.get_current_guide_illustration()
        msg = lego_task.get_current_instruction()
        ui_input_q.append((previous_success, 'initial'))
        ui_guidance_q.append((guidance, msg))

        result = lego_task.submit_frame(previous_success)
        assert result == FrameResult.SUCCESS

        step_time = timing_model.get_execution_time()

        for step in range(30):
            print(f'Target execution time {step_time:0.03f} seconds.')
            ti = time.monotonic()
            print(lego_task.get_current_instruction())

            for frame, instant in frame_model.step_iterator(
                    target_time=step_time):
                current_t = time.monotonic()
                dt = current_t - previous_t
                previous_t = current_t

                guidance = lego_task.get_current_guide_illustration()
                msg = lego_task.get_current_instruction()

                if frame == 'repeat':
                    img = previous_success
                else:
                    img = frameset.get_frame(step, frame)

                ui_input_q.append(
                    (img, f'Tag: {frame} | '
                          f'Time: {instant:0.03f} s / {step_time:0.03f} s'))
                ui_guidance_q.append((guidance, msg))

                result = lego_task.submit_frame(img)
                try:
                    match frame:
                        case 'repeat':
                            assert result == FrameResult.NO_CHANGE
                        case 'low_confidence':
                            assert result == FrameResult.LOW_CONFIDENCE
                        case 'blank':
                            assert result in (FrameResult.JUNK_FRAME,
                                              FrameResult.CV_ERROR)
                        case 'success':
                            assert result == FrameResult.SUCCESS
                            actual_step_time = time.monotonic() - ti
                            print(f'Finished step {step}, actual step time: '
                                  f'{actual_step_time:0.03f} seconds')

                            previous_success = img
                            break
                        case _:
                            raise RuntimeError()
                finally:
                    time.sleep(np.abs(rng.normal(loc=0.1, scale=0.03)))

            timing_model.set_delay(dt)
            step_time = timing_model.get_execution_time()
    finally:
        done_flag.set()


if __name__ == '__main__':
    view_width = 1200
    view_height = 300

    ui_input_q = deque(maxlen=1)
    ui_guidance_q = deque(maxlen=1)

    # initial black images
    current_guidance = np.zeros(
        (view_height, view_width // 2, 3),
        dtype=np.uint8
    )
    current_input = np.zeros(
        (view_height, view_width // 2, 3),
        dtype=np.uint8
    )

    # process loop check
    proc_done = threading.Event()


    def resize_add_text(img: npt.NDArray, text: str) -> npt.NDArray:
        w = view_width // 2

        img = cv2.resize(img, (w, view_height), cv2.INTER_AREA)
        cv2.putText(
            img=img,
            text=f'{text[:50]}',
            color=(0, 0, 255),
            org=(view_height // 10, w // 10),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            thickness=1
        )
        return img


    # start process thread
    pt = threading.Thread(
        target=processing_thread_loop,
        kwargs=dict(
            ui_input_q=ui_input_q,
            ui_guidance_q=ui_guidance_q,
            done_flag=proc_done
        )
    )
    pt.start()

    # ui loop
    _ = cv2.namedWindow('DEMO')
    while not proc_done.is_set():
        try:
            guidance, msg = ui_guidance_q.popleft()
            current_guidance = resize_add_text(guidance, msg)
        except IndexError:
            pass

        try:
            frame, tag = ui_input_q.popleft()
            current_input = resize_add_text(frame, tag)
        except IndexError:
            pass

        output_img = np.concatenate((current_guidance,
                                     current_input), axis=1)
        cv2.imshow('DEMO', output_img)
        cv2.waitKey(25)

    cv2.destroyAllWindows()
    pt.join()
