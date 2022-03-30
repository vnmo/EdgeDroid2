from __future__ import annotations

import threading
from collections import deque
from typing import Deque

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from gabriel_lego import FrameResult, LEGOTask

import edgedroid.data as e_data
from edgedroid.data.load import load_default_frame_probabilities, load_default_trace
from edgedroid.execution_times import TheoreticalExecutionTimeModel, preprocess_data
from edgedroid.frames import FrameModel
from edgedroid.model import EdgeDroidModel


def processing_thread_loop(
    ui_input_q: Deque, ui_guidance_q: Deque, done_flag: threading.Event
):
    task = pd.read_csv("./latin_sqr_0.csv")
    states = [np.array(eval(s), dtype=int) for s in task["state"]]
    # noinspection PyTypeChecker
    lego_task = LEGOTask(states)

    # load data and build model
    data = preprocess_data(*e_data.load_default_exec_time_data())
    timing_model = TheoreticalExecutionTimeModel(data=data, neuroticism=0.5)

    frameset = load_default_trace("square00")
    frame_model = FrameModel(load_default_frame_probabilities())

    edgedroid_model = EdgeDroidModel(
        frame_trace=frameset, frame_model=frame_model, timing_model=timing_model
    )

    try:
        for model_frame in edgedroid_model.play():
            guidance = lego_task.get_current_guide_illustration()
            msg = lego_task.get_current_instruction()
            ui_guidance_q.append((guidance, msg))
            ui_input_q.append(
                (
                    model_frame.frame_data,
                    f"Tag: {model_frame.frame_tag} | "
                    f"Time: {model_frame.step_frame_time:0.03f} s /"
                    f" {model_frame.step_target_time:0.03f} s",
                )
            )
            ui_guidance_q.append((guidance, msg))

            result = lego_task.submit_frame(model_frame.frame_data)
            match model_frame.frame_tag:
                case "repeat":
                    assert result == FrameResult.NO_CHANGE
                case "low_confidence":
                    assert result == FrameResult.LOW_CONFIDENCE
                case "blank":
                    assert result in (FrameResult.JUNK_FRAME, FrameResult.CV_ERROR)
                case "success" | "initial":
                    assert result == FrameResult.SUCCESS
                case _:
                    raise RuntimeError()
    except AssertionError:
        print(result)
        raise
    finally:
        done_flag.set()


if __name__ == "__main__":
    view_width = 1200
    view_height = 300

    ui_input_q = deque(maxlen=1)
    ui_guidance_q = deque(maxlen=1)

    # initial black images
    current_guidance = np.zeros((view_height, view_width // 2, 3), dtype=np.uint8)
    current_input = np.zeros((view_height, view_width // 2, 3), dtype=np.uint8)

    # process loop check
    proc_done = threading.Event()

    def resize_add_text(img: npt.NDArray, text: str) -> npt.NDArray:
        w = view_width // 2

        img = cv2.resize(img, (w, view_height), cv2.INTER_AREA)
        cv2.putText(
            img=img,
            text=f"{text[:50]}",
            color=(0, 0, 255),
            org=(view_height // 10, w // 10),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            thickness=1,
        )
        return img

    # start process thread
    pt = threading.Thread(
        target=processing_thread_loop,
        kwargs=dict(
            ui_input_q=ui_input_q, ui_guidance_q=ui_guidance_q, done_flag=proc_done
        ),
    )
    pt.start()

    # ui loop
    _ = cv2.namedWindow("DEMO")
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

        output_img = np.concatenate((current_guidance, current_input), axis=1)
        cv2.imshow("DEMO", output_img)
        cv2.waitKey(25)

    cv2.destroyAllWindows()
    pt.join()
