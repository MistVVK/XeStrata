"""The elastic image encoder (the vision section's "elastic": true with "vram_elastic"): the first picture makes the
expert cache keep the encoder's VRAM free and then starts it, later pictures use it as it is, an idle encoder stops and
the cache takes its VRAM back, a POST /v1/vram reserve stacks on the encoder's share, and the engine's load never starts
it on its own."""
import contextlib
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, Service, Vision
from serve.test_vram import VramEngine

ROOT = Path(__file__).resolve().parents[1]
IMAGE = {"type": "image", "source": "x.png"}


class ElasticVision:
    """An encoder process stand-in: starts, stops, and turns every picture into three image tokens."""
    def __init__(self, d, elastic=True):
        self.elastic, self.elastic_mib, self.elastic_idle_s = elastic, 2048, 60.0
        self.up, self.starts, self.unloads = False, 0, 0
        self.dir = Path(d)
        self.rows = self.dir / "img.sve"
        self.rows.write_bytes(b"rows")

    def alive(self):
        return self.up

    def restart(self):
        self.up, self.starts = True, self.starts + 1

    def unload(self):
        self.up, self.unloads = False, self.unloads + 1

    def encode(self, source):
        assert self.up, "a picture reached an encoder that was not started"
        return self.rows, 3


class Elastic(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp()
        self.tok = ByteTokenizer()
        self.engine = VramEngine(self.tok)
        self.engine.loaded = True
        self.vision = ElasticVision(d)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=self.vision)
        self.svc.free_vram_mib = lambda: None               # (not this PC's card)
        self.addCleanup(self.drop_embeddings)

    def drop_embeddings(self):
        path = getattr(getattr(self.svc, "embeddings", None), "path", None)
        if path is not None:
            path.unlink(missing_ok=True)

    def picture(self):
        self.svc.prepare([{"role": "user", "content": [{"type": "text", "text": "what is it"}, IMAGE]}], None, {})

    def test_the_first_picture_makes_room_then_starts_the_encoder(self):
        self.picture()
        self.assertEqual(self.engine.calls, [2048])         # the cache keeps the encoder's share free first
        self.assertEqual(self.vision.starts, 1)
        self.assertTrue(self.svc.vision_held)
        self.picture()                                      # the next one uses the running encoder
        self.assertEqual(self.engine.calls, [2048])
        self.assertEqual(self.vision.starts, 1)

    def test_an_idle_encoder_stops_and_the_cache_grows_back_whole(self):
        self.picture()
        self.assertEqual(self.svc._vision_idle_down(), "busy")             # a picture came just now
        self.svc.last_image_at = time.time() - 61
        with self.svc.fifo:                                                # a request holds the turn
            self.assertEqual(self.svc._vision_idle_down(), "busy")
        self.assertEqual(self.svc._vision_idle_down(), "down")
        self.assertFalse(self.vision.up)
        self.assertFalse(self.svc.vision_held)
        self.assertEqual(self.engine.calls, [2048, 0])      # 0, not None: the full size, not the start's reserve
        self.assertEqual(self.svc._vision_idle_down(), "nothing")
        self.picture()                                      # and the next picture starts it again
        self.assertEqual(self.vision.starts, 2)
        self.assertEqual(self.engine.calls, [2048, 0, 2048])

    def test_a_vram_reserve_stacks_on_the_encoders_share(self):
        self.svc.vram(4000)
        self.picture()
        self.assertEqual(self.engine.calls, [4000, 6048])
        self.svc.vram(1000)                                 # asked while the encoder runs
        self.assertEqual(self.engine.calls[-1], 3048)
        self.svc.last_image_at = 0
        self.svc._vision_idle_down()
        self.assertEqual(self.engine.calls[-1], 1000)       # back to the asked reserve, not to 0

    def test_an_engine_that_cannot_shrink_still_gets_its_encoder(self):
        self.engine.elastic = False
        self.picture()
        self.assertEqual(self.vision.starts, 1)
        self.assertFalse(self.svc.vision_held)
        self.svc.last_image_at = 0
        self.assertEqual(self.svc._vision_idle_down(), "down")
        self.assertEqual(self.engine.calls, [2048])         # the refused shrink only: nothing to give back

    def test_an_encoder_that_fails_to_start_is_a_503_not_a_dropped_connection(self):
        def fail():
            raise RuntimeError("the vision encoder did not start: cudaMalloc failed: out of memory")
        self.vision.restart = fail
        with self.assertRaises(EngineDied) as e:            # the handlers answer EngineDied with a 503
            self.picture()
        self.assertIn("the image encoder failed to start", str(e.exception))
        self.svc.last_image_at = 0
        self.assertEqual(self.svc._vision_idle_down(), "down")
        self.assertEqual(self.engine.calls, [2048, 0])      # the share made for it goes back

    def test_the_stop_is_logged_when_the_cache_never_made_room(self):
        self.engine.elastic = False                         # a layer split: the engine refuses --vram-elastic
        self.picture()
        self.svc.last_image_at = 0
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(self.svc._vision_idle_down(), "down")
        self.assertIn("the image encoder stopped after 60 s without a picture", out.getvalue())
        self.assertNotIn("expert cache", out.getvalue())

    def test_the_qwen36_engine_is_not_sent_a_command_it_does_not_have(self):
        # the Qwen3.6 engine path reports no vram_elastic and has no VRAM command: it answered "VRAM 2048" with its usage text, and the log said
        # "the expert cache did not make room for the image encoder (expected: GEN <max_new> ...)" at every picture (0.2.15 release test, boston)
        del self.engine.info["vram_elastic"]
        usage = "expected: GEN <max_new> [key=value ...] <id,id,...>"
        self.engine.vram = lambda reserve, timeout=120.0: (self.engine.calls.append(reserve), (_ for _ in ()).throw(ValueError(usage)))[1]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.picture()
            self.svc.last_image_at = 0
            self.assertEqual(self.svc._vision_idle_down(), "down")
            self.picture()
        self.assertEqual(self.engine.calls, [])             # never asked
        self.assertEqual(self.vision.starts, 2)             # the encoder starts all the same
        self.assertNotIn("expected: GEN", out.getvalue())
        self.assertNotIn("did not make room", out.getvalue())
        with self.assertRaises(ValueError) as e:            # POST /v1/vram: the server's own sentence, not the engine's usage text
            self.svc.vram(4000)
        self.assertEqual(str(e.exception), "this engine cannot resize its VRAM use")
        self.assertEqual(self.engine.calls, [])

    def test_the_load_never_starts_an_elastic_encoder(self):
        self.assertFalse(self.svc._vision_down())
        self.svc.vision = ElasticVision(tempfile.mkdtemp(), elastic=False)
        self.assertTrue(self.svc._vision_down())            # a non-elastic encoder that is down is started again

    def test_an_unload_lets_go_of_the_share(self):
        self.picture()
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertFalse(self.svc.vision_held)
        self.assertFalse(self.vision.up)

    def test_the_config_keys(self):
        v = Vision({"exe": "e", "mmproj": "m", "model": "t", "elastic": True, "elastic_mib": 3000,
                    "elastic_idle_s": 30}, lazy=False)          # elastic: nothing starts with the server
        self.assertTrue(v.elastic and v.elastic_mib == 3000 and v.elastic_idle_s == 30 and not v.alive())
        v = Vision({"exe": "e", "mmproj": "m", "model": "t"}, lazy=True)
        self.assertFalse(v.elastic)
        self.assertEqual((v.elastic_mib, v.elastic_idle_s), (2048, 120.0))

    def test_the_ask_follows_the_image_limit(self):
        """Without the person's own elastic_mib the cache keeps the reader's need at its image limit free, never less than 2,048 MiB."""
        base = {"exe": "e", "mmproj": "m", "model": "t", "gpu": True, "elastic": True}
        for limit, ask in ((None, 2048), (1024, 2048), (4096, 2048), (16384, 3674), (0, 3674)):
            cfg = dict(base) if limit is None else {**base, "max_tokens": limit}
            self.assertEqual(Vision(cfg, lazy=True).elastic_mib, ask, limit)
        self.assertEqual(Vision({**base, "max_tokens": 16384, "elastic_mib": 2500}, lazy=True).elastic_mib, 2500)
        self.assertEqual(Vision({**base, "max_tokens": 1024, "elastic_mib": 5000}, lazy=True).elastic_mib, 5000)


class OnDemandReader(unittest.TestCase):
    """The reader that starts with the first picture: no room on the card -> the CPU reader reads it, the cache grows back at once,
    and the next start tries the card again; pictures already encoded start nothing."""

    def vision(self, d, cfg=None):
        v = Vision(cfg or {"exe": "gpu-reader", "mmproj": "m", "model": "t", "gpu": True, "cpu_exe": "cpu-reader", "elastic": True}, lazy=True)
        v.started = []

        def start_reader():
            exe = v.spawn[0][0]
            v.started.append(exe)
            if exe.endswith("gpu-reader") and v.gpu_fails:
                raise RuntimeError("cudaMalloc failed: out of memory")
            v.proc = mock.Mock()
            v.proc.poll.return_value = None
            v.stopped = False
        v.gpu_fails = False
        v._start_reader = start_reader
        v.close = lambda: None
        return v

    def service(self, v, elastic_engine=True):
        tok = ByteTokenizer()
        engine = VramEngine(tok, elastic=elastic_engine)
        engine.loaded = True
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=v)
        return svc, engine

    def test_a_gpu_reader_that_does_not_start_is_the_cpu_reader_for_this_start_only(self):
        v = self.vision(tempfile.mkdtemp())
        v.gpu_fails = True
        svc, engine = self.service(v)
        svc._vision_up()
        self.assertEqual([x.split("-")[0] for x in v.started], ["gpu", "cpu"])
        self.assertTrue(v.on_cpu)
        self.assertEqual(engine.calls, [2048, 0])           # the cache made room, then took it back: the CPU reader needs none
        self.assertFalse(svc.vision_held)
        svc.last_image_at = 0
        self.assertEqual(svc._vision_idle_down(), "down")   # the idle stop ends the CPU reader too ...
        self.assertEqual(engine.calls, [2048, 0])           # ... and has no VRAM to give back
        self.assertFalse(v.on_cpu)                          # the next start tries the card again
        v.gpu_fails = False
        svc._vision_up()                                    # the card is tried again, and works now
        self.assertEqual(v.started[-1].split("-")[0], "gpu")
        self.assertFalse(v.on_cpu)
        self.assertTrue(svc.vision_held)

    def test_no_room_and_a_cache_that_cannot_shrink_goes_straight_to_the_cpu_reader(self):
        v = self.vision(tempfile.mkdtemp())
        svc, engine = self.service(v, elastic_engine=False)
        svc.free_vram_mib = lambda: 900
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            svc._vision_up()
        self.assertEqual([x.split("-")[0] for x in v.started], ["cpu"])   # the card was not tried
        self.assertIn("this picture is read on the CPU", out.getvalue())
        self.assertFalse(svc.vision_held)

    def test_room_enough_and_a_cache_that_cannot_shrink_still_tries_the_card(self):
        v = self.vision(tempfile.mkdtemp())
        svc, engine = self.service(v, elastic_engine=False)
        svc.free_vram_mib = lambda: 5000
        svc._vision_up()
        self.assertEqual([x.split("-")[0] for x in v.started], ["gpu"])
        svc2, _ = self.service(self.vision(tempfile.mkdtemp()), elastic_engine=False)
        svc2.free_vram_mib = lambda: None                   # unknown: the card is tried (and the CPU reader is there if it fails)
        svc2._vision_up()
        self.assertEqual([x.split("-")[0] for x in svc2.vision.started], ["gpu"])

    def test_a_cpu_reader_starts_without_asking_the_cache_for_vram(self):
        """An on-demand CPU reader by choice (no "gpu" in the vision section) uses no VRAM: its start asks the expert cache for nothing,
        whether the engine can shrink its cache or not, and logs no missing room.  A GPU reader still asks."""
        cpu = {"exe": "cpu-reader", "mmproj": "m", "model": "t", "elastic": True}
        for elastic_engine in (True, False):
            with self.subTest(elastic_engine=elastic_engine):
                v = self.vision(tempfile.mkdtemp(), cpu)
                svc, engine = self.service(v, elastic_engine=elastic_engine)
                svc.free_vram_mib = lambda: None            # (not this PC's card)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    svc._vision_up()
                self.assertEqual([x.split("-")[0] for x in v.started], ["cpu"])
                self.assertEqual(engine.calls, [])             # the cache was not asked
                self.assertFalse(svc.vision_held)
                self.assertIn("the image encoder started in", out.getvalue())
                self.assertIn("on the CPU", out.getvalue())
                self.assertNotIn("did not make room", out.getvalue())
                svc.last_image_at = 0
                self.assertEqual(svc._vision_idle_down(), "down")
                self.assertEqual(engine.calls, [])             # nothing to give back at the stop either
        v = self.vision(tempfile.mkdtemp())
        svc, engine = self.service(v)
        svc.free_vram_mib = lambda: None
        svc._vision_up()
        self.assertEqual([x.split("-")[0] for x in v.started], ["gpu"])
        self.assertEqual(engine.calls, [2048])                 # the GPU reader: the cache keeps its share free first
        self.assertTrue(svc.vision_held)

    def test_a_non_elastic_gpu_reader_keeps_the_cpu_reader_for_good(self):
        v = Vision({"exe": "gpu-reader", "mmproj": "m", "model": "t", "gpu": True, "cpu_exe": "cpu-reader"}, lazy=True)
        started = []

        def start_reader():
            started.append(v.spawn[0][0])
            if started[-1].endswith("gpu-reader"):
                raise RuntimeError("no kernel image")
        v._start_reader = start_reader
        v._start()
        v.unload()
        self.assertEqual([x.split("-")[0] for x in started], ["gpu", "cpu"])
        self.assertTrue(v.on_cpu)                           # as before: the card is not tried again

    def test_pictures_already_encoded_start_no_reader(self):
        d = Path(tempfile.mkdtemp())
        v = self.vision(d)
        png = bytes([0x89]) + b"PNG" + bytes([13, 10, 26, 10]) + b"rest"
        v.dir = d
        v.cache["k"] = (d / "x.sve", 3)
        import hashlib
        v.cache[hashlib.sha256(png).hexdigest()[:32]] = (d / "x.sve", 3)
        self.assertTrue(v.all_cached([png]))
        self.assertFalse(v.all_cached([png + b"x"]))
        self.assertFalse(v.all_cached(["/no/such/file.png"]))   # an error is encode()'s to report


class OnDemandByDefault(unittest.TestCase):
    """An NVIDIA GPU reader on one GPU starts on demand without the config saying so; everything else keeps its reader as it was."""
    BASE = {"args": ["--max-context", "8192", "--vision"], "gpu": 0,
            "vision": {"exe": "/e/strata-vision", "mmproj": "m", "model": "t", "gpu": True, "max_tokens": 1024}}

    def setUp(self):
        # these tests describe a PC (x86); the Spark class (Linux on Arm) has its own tests below, so a test run on a Spark does not change them
        p = mock.patch("platform.machine", return_value="x86_64")
        p.start()
        self.addCleanup(p.stop)

    def cfg(self, **over):
        import copy
        c = copy.deepcopy(self.BASE)
        for k, v in over.items():
            if k == "vision":
                c["vision"].update(v)
            else:
                c[k] = v
        return c

    def test_the_spark_class_keeps_the_reader_loaded_by_default(self):
        """Linux on Arm with an NVIDIA GPU reader on one GPU (a DGX Spark): resident, as the AMD reader and the Mac's are (owner 2026-11-11)."""
        import sys
        from serve.server import apply_vision_defaults, vision_on_demand_by_default as f
        for machine in ("aarch64", "arm64", "AARCH64"):
            with self.subTest(machine=machine), mock.patch("platform.machine", return_value=machine), mock.patch.object(sys, "platform", "linux"):
                self.assertFalse(f(self.cfg()))
                c = apply_vision_defaults(self.cfg(args=["--vision", "--vram-reserve-mib", "700"]))
                self.assertNotIn("elastic", c["vision"])                                    # no key written: the reader starts with the server
                self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "700")   # the resident path's reserve, not 2,048
                self.assertFalse(f(self.cfg(vision={"exe": "component:engine/strata-vision"})))
                self.assertFalse(f(self.cfg(backend="cuda")))

    def test_the_spark_class_still_takes_an_explicit_elastic_key_either_way(self):
        import sys
        from serve.server import apply_vision_defaults
        with mock.patch("platform.machine", return_value="aarch64"), mock.patch.object(sys, "platform", "linux"):
            c = apply_vision_defaults(self.cfg(vision={"elastic": True}))
            self.assertIs(c["vision"]["elastic"], True)                                     # the old behaviour is still reachable
            self.assertEqual(c["args"], self.BASE["args"])
            c = apply_vision_defaults(self.cfg(vision={"elastic": False}))
            self.assertIs(c["vision"]["elastic"], False)

    def test_other_hosts_keep_the_on_demand_default(self):
        import sys
        from serve.server import vision_on_demand_by_default as f
        for system, machine in (("linux", "x86_64"), ("linux", "AMD64"), ("win32", "AMD64"), ("win32", "ARM64")):    # a PC, and Windows on Arm (not this rule)
            with self.subTest(system=system, machine=machine), mock.patch("platform.machine", return_value=machine), mock.patch.object(sys, "platform", system):
                self.assertTrue(f(self.cfg()))
        with mock.patch("platform.machine", return_value="arm64"), mock.patch.object(sys, "platform", "darwin"):
            self.assertFalse(f(self.cfg()))                                                # a Mac was never on demand

    def test_the_spark_class_does_not_change_the_amd_rule(self):
        import sys
        import serve.server as S
        with mock.patch("platform.machine", return_value="aarch64"), mock.patch.object(sys, "platform", "linux"), \
                mock.patch.object(S, "VISION_ON_DEMAND_AMD_BY_DEFAULT", True):
            self.assertTrue(S.vision_on_demand_by_default(self.cfg(vision={"exe": "/e/strata-vision-hip", "cpu_exe": "/e/strata-vision"}, backend="hip")))

    def test_the_plain_nvidia_config_is_on_demand(self):
        from serve.server import apply_vision_defaults, vision_on_demand_by_default
        c = apply_vision_defaults(self.cfg())
        self.assertTrue(c["vision"]["elastic"])
        self.assertIs(c["vram_elastic"], True)                                      # owner 11-11: the cache makes the room when a picture comes
        self.assertEqual(c["args"], self.BASE["args"])                               # ... and nothing is held back before it
        c = apply_vision_defaults(self.cfg(args=["--vision", "--vram-reserve-mib", "700"]))
        self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "700")    # setup's reserve stays as written
        c = apply_vision_defaults(self.cfg(args=["--vision", "--vram-reserve-mib", "1536"]))
        self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "1536")   # a chosen reserve is kept
        # a config that names the shrinking cache itself keeps the old rule: the reader's room in the reserve
        c = apply_vision_defaults(self.cfg(vram_elastic=False, args=["--vision", "--vram-reserve-mib", "700"]))
        self.assertIs(c["vram_elastic"], False)
        self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "2048")
        c = apply_vision_defaults(self.cfg(vram_elastic=False))
        self.assertEqual(c["args"][-2:], ["--vram-reserve-mib", "2048"])
        import serve.server as S
        with mock.patch.object(S, "VISION_NO_RESERVE_BY_DEFAULT", False):              # the switch back: the old rule everywhere
            c = apply_vision_defaults(self.cfg(args=["--vision", "--vram-reserve-mib", "700"]))
            self.assertNotIn("vram_elastic", c)
            self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "2048")
        self.assertTrue(vision_on_demand_by_default(self.cfg(vision={"exe": "component:engine/strata-vision.exe"})))
        self.assertTrue(vision_on_demand_by_default(self.cfg(vision={"exe": "C:/e/strata-vision.exe"})))
        bs = chr(92)
        self.assertTrue(vision_on_demand_by_default(self.cfg(vision={"exe": "C:" + bs + "e" + bs + "strata-vision.exe"})))

    def test_the_amd_reader_with_its_cpu_reader_is_on_demand_when_asked(self):
        import serve.server as S
        f = S.vision_on_demand_by_default
        amd = self.cfg(vision={"exe": "/e/strata-vision-hip", "cpu_exe": "/e/strata-vision"}, backend="hip")
        self.assertFalse(f(amd))                                  # off by default on AMD (no slots gained: see the constant's note)
        with mock.patch.object(S, "VISION_ON_DEMAND_AMD_BY_DEFAULT", True):
            self.assertTrue(f(amd))
            got = S.apply_vision_defaults(amd)
            self.assertTrue(got["vision"]["elastic"])
            self.assertNotIn("vram_elastic", got)                 # AMD: no shrinking cache, the room stays in the reserve
            self.assertEqual(got["args"][-2:], ["--vram-reserve-mib", "2048"])
        with mock.patch.object(__import__("serve.server").server, "VISION_ON_DEMAND_AMD_BY_DEFAULT", True):
            self.assertFalse(f(self.cfg(vision={"exe": "/e/strata-vision-hip", "cpu_exe": "/e/strata-vision"}, backend="hip", gpu=[0, 1])))
        self.assertFalse(f(self.cfg(vision={"exe": "/e/strata-vision-sycl", "cpu_exe": "/e/strata-vision"}, backend="sycl")))

    def test_what_stays_as_it_was(self):
        from serve.server import vision_on_demand_by_default as f
        self.assertFalse(f(self.cfg(vision={"gpu": False})))                         # the CPU reader
        self.assertFalse(f(self.cfg(vision={"elastic": False})))                     # the person said no
        self.assertFalse(f(self.cfg(vision={"cuda_device": 1})))                     # a reader on its own card
        self.assertFalse(f(self.cfg(vision={"exe": "/e/strata-vision-hip"}, backend="hip")))              # AMD without its CPU reader
        self.assertFalse(f(self.cfg(vision={"exe": "/e/strata-vision"}, backend="hip")))
        self.assertFalse(f(self.cfg(backend="sycl", vision={"exe": "/e/strata-vision-sycl"})))
        self.assertFalse(f(self.cfg(gpu=[0, 1])))                                     # a layer split
        self.assertFalse(f(self.cfg(args=["--vision", "--kv-grow"])))
        self.assertTrue(f(self.cfg(args=["--vision", "--resident-budget-gib", "30"])))   # no cache to shrink is needed
        self.assertFalse(f(self.cfg(parallel=4)))
        c = self.cfg()
        c.pop("vision")
        self.assertFalse(f(c))
        c = self.cfg(vision={"elastic": True})                                        # already asked for: untouched
        from serve.server import apply_vision_defaults
        self.assertEqual(apply_vision_defaults(c)["args"], self.BASE["args"])   # already asked for: untouched


if __name__ == "__main__":
    unittest.main()
