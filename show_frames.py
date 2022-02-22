import cv2
import pandas as pd

from edgedroid.frames import FrameModel, FrameSet


def show_frames():
    probs = pd.read_csv('./frame_probs_per_relpos.csv')
    model = FrameModel(probs)
    frameset = FrameSet.from_datafile('./square00.tgz')

    _ = cv2.namedWindow('frames')
    for frame in model.step_iterator(target_time=30):
        if frame == 'repeat':
            img = frameset.get_frame(9, 'success')
        else:
            img = frameset.get_frame(10, frame)

        cv2.putText(
            img=img,
            text=frame,
            color=(0, 0, 255),
            org=(40, 40),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=2,
            thickness=2
        )
        cv2.imshow('frames', img)
        cv2.waitKey(500)

    cv2.waitKey(5000)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    show_frames()
