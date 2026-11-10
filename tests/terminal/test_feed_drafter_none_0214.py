"""A feed model whose entry says `drafter: none` starts with no drafter at every start, and the Terminal plans none for it (lane APP-LIN 0.2.14 item 8; Q4-P8 G14)."""
from pathlib import Path
from types import SimpleNamespace

from strata_cli.launch import runner as R
from strata_control import drafting


def _cfg(**kw):
    return {"model_name": "qwen4exp-q4_k_m", "args": ["--native", "/m/a.gguf", "--mtp", "/d/mtp/rt", "--spec-min-p", "0.5", "--max-context", "32768"], **kw}


def test_a_feed_model_with_no_draft_layer_starts_with_no_drafter():
    from serve.server import engine_args
    for extra in ({"mtp": False, "drafting": "off"}, {"mtp": False}, {"drafting": "off"}):
        d = drafting.apply(_cfg(**extra), backend="cuda", archs=["sm_86"])
        assert d.drafter == "off", extra
        started = engine_args(_cfg(**extra), announce=False)     # what the server starts: no draft layer, whichever key said so
        assert "--mtp" not in started and "--spec-min-p" not in started, extra
        assert started[started.index("--max-context") + 1] == "32768"


def test_other_models_keep_their_decision():
    cfg = _cfg()                                              # a model DFlash2 is not for, draft layer on: as before
    assert drafting.apply(cfg, backend="cuda", archs=["sm_86"]).drafter == "mtp"
    assert "--mtp" in cfg["args"]
    cfg = _cfg(drafting="dflash2")
    d = drafting.apply(cfg, backend="cuda", archs=["sm_86"])
    assert (d.drafter, d.note) == ("mtp", drafting.NOT_SUPPORTED)


def test_the_terminal_plans_no_drafter_for_it():
    fake = SimpleNamespace(s=SimpleNamespace(drafting=None, family="qwen4exp"), split_expected=lambda: False,
                           feed_without_draft_layer=lambda family=None: family == "qwen4exp")
    assert R.Runner.resolved_drafting(fake, "qwen4exp") == "off"
    fake.s.drafting = "mtp"
    assert R.Runner.resolved_drafting(fake, "qwen4exp") == "off", "the head the MTP would load is Flash-Next's, not this model's"


def test_setup_writes_drafting_off_for_it():
    src = (Path(drafting.__file__).parent / "setup_config.py").read_text(encoding="utf-8")
    assert 'if getattr(a, "feed_no_drafter", False):' in src and 'cfg["drafting"] = "off"' in src
    setup = (Path(drafting.__file__).parent.parent / "setup.py").read_text(encoding="utf-8")
    at = setup.index('fam.get("drafter") == "none" and a.mtp is None')
    assert "a.feed_no_drafter = True" in setup[at:at + 600]
