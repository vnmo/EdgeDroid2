import contextlib
import socket
from typing import Callable, Literal

import click
from loguru import logger

from .common import EdgeDroidFrame, response_stream_unpack
from .. import data as e_data
from ..models import (
    EdgeDroidModel,
    EmpiricalExecutionTimeModel,
    ExecutionTimeModel,
    FrameModel,
    ModelFrame,
    TheoreticalExecutionTimeModel,
    preprocess_data,
)


class StreamSocketEmulation:
    def __init__(
        self,
        neuroticism: float,
        trace: str,
        fade_distance: int,
        model: Literal["theoretical", "empirical"] = "theoretical",
    ):
        logger.info(
            f"Initializing EdgeDroid model with neuroticism {neuroticism:0.2f} and "
            f"fade distance {fade_distance:d} steps"
        )
        logger.info(f"Model type: {model}")
        logger.info(f"Trace: {trace}")

        # should be able to use a single thread for everything

        # first thing first, prepare data
        data = preprocess_data(
            *e_data.load_default_exec_time_data(),
            transition_fade_distance=fade_distance,
        )
        frameset = e_data.load_default_trace(trace)

        # prepare models
        if model == "theoretical":
            timing_model: ExecutionTimeModel = TheoreticalExecutionTimeModel(
                data=data,
                neuroticism=neuroticism,
                transition_fade_distance=fade_distance,
            )
        else:
            timing_model: ExecutionTimeModel = EmpiricalExecutionTimeModel(
                data=data,
                neuroticism=neuroticism,
                transition_fade_distance=fade_distance,
            )

        frame_model = FrameModel(e_data.load_default_frame_probabilities())

        self._model = EdgeDroidModel(
            frame_trace=frameset, frame_model=frame_model, timing_model=timing_model
        )

    def emulate(
        self,
        sock: socket.SocketType,
        emit_cb: Callable[[ModelFrame], None] = lambda _: None,
        resp_cb: Callable[[bool], None] = lambda _: None,
    ) -> None:
        """
        # TODO: document

        Parameters
        ----------
        sock
        emit_cb
        resp_cb

        Returns
        -------

        """

        logger.warning("Starting emulation")
        with contextlib.closing(response_stream_unpack(sock)) as resp_stream:
            step = 0
            logger.info(f"Current step: {step}")
            for model_frame in self._model.play():
                # package and send the frame
                logger.debug(
                    f"Sending frame:\n"
                    f"\tSeq: {model_frame.seq}\n"
                    f"\tTag: {model_frame.frame_tag}\n"
                    f"\tStep index: {model_frame.step_index}\n"
                    f"\tFrame step seq: {model_frame.step_seq}"
                )
                payload = EdgeDroidFrame(model_frame.seq, model_frame.frame_data).pack()
                sock.sendall(payload)
                emit_cb(model_frame)

                # wait for response
                logger.debug("Waiting for response from server")
                resp = next(resp_stream)
                logger.debug("Received response from server")
                resp_cb(resp)

                if model_frame.frame_tag in ("success", "initial"):
                    if resp:
                        # if we receive a response for a success frame, advance the model
                        logger.info("Advancing to next step")
                        self._model.advance_step()
                        step += 1
                        logger.info(f"Current step: {step}")
                    else:
                        logger.error(
                            "Received unexpected unsuccessful response from server, "
                            "aborting"
                        )
                        raise click.Abort()

        logger.warning("Emulation finished")


# TODO: Move CLI to a separate module
@click.command()
@click.argument("host", type=str)
@click.argument("port", type=int)
@click.option(
    "-n",
    "--neuroticism",
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
    show_default=True,
)
@click.option("-t", "--trace", type=str, default="square00", show_default=True)
@click.option("-f", "--fade-distance", type=int, default=8, show_default=True)
@click.option(
    "-m",
    "--model",
    type=click.Choice(["empirical", "theoretical"], case_sensitive=False),
    default="theoretical",
    show_default=True,
)
@click.option("--frame-timeout-seconds", type=float, default=5.0, show_default=True)
def run_client(
    host: str,
    port: int,
    neuroticism: float,
    trace: str,
    fade_distance: int,
    model: Literal["empirical", "theoretical"],
):
    emulation = StreamSocketEmulation(
        neuroticism=neuroticism, trace=trace, fade_distance=fade_distance, model=model
    )

    # "connect" to remote
    # this is of course just for convenience, to skip adding an address to every
    # send() call, as there are no "connections" in udp.
    logger.info(f"Connecting to remote server at {host}:{port}/tcp")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(30.0)  # TODO: magic number, maybe add as an option. This is
        # just the timeout for the initial connection.
        try:
            sock.connect((host, port))
        except socket.timeout:
            logger.error(f"Timed out connecting to backend at {host}:{port}")
            raise click.Abort()
        except ConnectionRefusedError:
            logger.error(f"{host}:{port} refused connection.")
            raise click.Abort()
        except socket.error as e:
            logger.error(
                f"Encountered unspecified socket error when connecting to {host}:{port}"
            )
            logger.exception(e)
            raise click.Abort()

        sock.settimeout(None)  # blocking mode
        # these are tcp sockets, so no timeouts are needed

        emulation.emulate(sock)


if __name__ == "__main__":
    run_client()
