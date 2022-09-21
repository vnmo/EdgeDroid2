import socket
import tempfile
import time
import unittest
from multiprocessing import Pool
from typing import Optional

from click.testing import CliRunner
from loguru import logger

from ..load_generator.client.cli import edgedroid_client
from ..load_generator.client.client import StreamSocketEmulation
from ..load_generator.server import edgedroid_server
from ..load_generator.server.server import serve_LEGO_task


def run_server(
    task: str = "test",
    port: int = 5000,
    truncate: Optional[int] = None,
):
    try:
        with tempfile.NamedTemporaryFile() as tmpf:
            serve_LEGO_task(
                task_name=task,
                port=port,
                output_path=tmpf,
                bind_address="127.0.0.1",
                truncate=truncate,
            )
    except Exception as e:
        logger.exception(e)
        return e

    return None


def run_client(
    task: str = "test",
    port: int = 5000,
    truncate: Optional[int] = None,
    wait_seconds: float = 1.0,
):
    time.sleep(wait_seconds)
    try:
        emulation = StreamSocketEmulation(
            neuroticism=0.5,
            trace=task,
            # fade_distance=8,
            model="fitted-naive",
            sampling="ideal",
            truncate=truncate,
        )

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            sock.connect(("127.0.0.1", port))
            sock.settimeout(None)  # no timeouts are needed
            emulation.emulate(sock)
    except Exception as e:
        logger.exception(e)
        return e

    return None


class TestCli(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.task = "test"
        self.port = 5000
        self.truncate = (-1, 3)

    def test_server_cli(self):
        with Pool() as pool, tempfile.TemporaryDirectory() as tmpdir:
            for trunc in self.truncate:
                client_proc = pool.apply_async(
                    run_client,
                    args=(self.task, self.port, trunc, 1.0),
                )

                res = self.runner.invoke(
                    edgedroid_server,
                    [
                        "127.0.0.1",
                        str(self.port),
                        self.task,
                        "-o",
                        tmpdir,
                        "-v",
                        "--truncate",
                        f"{trunc}",
                    ],
                )
                self.assertEqual(res.exit_code, 0)

                if isinstance((cres := client_proc.get(timeout=1.0)), Exception):
                    raise cres

    def test_client_cli(self):
        with Pool() as pool, tempfile.TemporaryDirectory() as tmpdir:
            for trunc in self.truncate:
                server_proc = pool.apply_async(
                    run_server,
                    args=(self.task, self.port, trunc),
                )
                time.sleep(1.0)  # give it some time to start...

                res = self.runner.invoke(
                    edgedroid_client,
                    [
                        "127.0.0.1",
                        str(self.port),
                        self.task,
                        "-n",
                        "0.5",
                        "-m",
                        "fitted-naive",
                        "-s",
                        "ideal",
                        "-o",
                        f"{tmpdir}",
                        "-v",
                        "--truncate",
                        f"{trunc}",
                    ],
                )
                if res.exit_code != 0 and res.exception:
                    raise res.exception

                self.assertEqual(res.exit_code, 0)

                if isinstance((cres := server_proc.get(timeout=1.0)), Exception):
                    raise cres
