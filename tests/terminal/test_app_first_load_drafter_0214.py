"""The app's setup, the app's first load and the Terminal's start pick the same drafter for Auto (lane APP-LIN 0.2.14 item 7).

Before: the app prepared every DFlash2-capable model for DFlash2 (the host's own rule), so on a card where Control's Auto means the MTP the setup skipped the MTP layer and the
first load ran undrafted. Now the app asks `strata drafting plan` at the install (no run config yet: --family/--model) and at the load (--config). This pins that both answers
equal the start's own decision (drafting.choice_of, what serve/server.py's drafting.apply runs) on the card classes where they differ: a 12 GB CUDA card streaming IQ3_S (MTP)
and a 24 GB one (DFlash2).

Plain unittest, no pytest fixtures (tmp_path, capsys, monkeypatch): the Windows gate of f-0214-app reported both cases as ERROR at setup with the pytest version.
"""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli import drafting_cmd
from strata_control import drafting


def _plan(**kw) -> dict:
    a = argparse.Namespace(drafting_command="plan", data_dir=None, config=None, family=None, model=None, mode=None, no_split=False,
                           engines_unloading=False, output="json")
    for k, v in kw.items():
        setattr(a, k, v)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = drafting_cmd.run(a)
    assert code == 0, out.getvalue()
    return json.loads(out.getvalue().strip().splitlines()[-1])


class SetupFirstLoadAndStartAgree(unittest.TestCase):
    def test_on_a_12_gb_and_a_24_gb_cuda_card(self):
        for vram_mib, expected in ((12288, "mtp"), (24576, "dflash2")):
            with self.subTest(vram_mib=vram_mib), tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.object(drafting_cmd, "_hardware", return_value=("cuda", ["sm_86"])), \
                    mock.patch.object(drafting, "card_vram_mib", return_value=float(vram_mib)), \
                    mock.patch.object(drafting_cmd, "_split_expected", return_value=False), \
                    mock.patch.object(drafting_cmd, "_installed_dflash2_configs", return_value=[]):
                data = Path(tmp) / "data"
                mtp = data / "mtp" / "rt"
                mtp.mkdir(parents=True)
                # 1. the app's setup: no run config yet
                self.assertEqual(_plan(family="qwen", model="IQ3_S", mode="auto", data_dir=str(data))["choice"], expected)
                # 2. the app's first load: the run config setup wrote (Auto: no "drafting" key)
                (mtp / "experts.bin").write_bytes(b"x" * 2048)
                cfg = {"model_name": "qwen3.8-flash-next-iq3_s", "backend": "cuda",
                       "args": ["--pack", str(data / "packs" / "iq3_s"), "--native", str(Path(tmp) / "IQ3_S-00001-of-00002.gguf"), "--mtp", str(mtp), "--max-context", "131072"]}
                path = Path(tmp) / "strata-iq3_s.json"
                path.write_text(json.dumps(cfg), encoding="utf-8")
                self.assertEqual(_plan(config=str(path))["choice"], expected)
                # 3. the start (serve/server.py runs drafting.apply on the same config): the same choice
                self.assertEqual(drafting.choice_of("auto", model_name=cfg["model_name"], backend="cuda", archs=["sm_86"], args=cfg["args"], vram_mib=float(vram_mib)), expected)


if __name__ == "__main__":
    unittest.main()
