"""pytest: the test_*.py files here are scripts, not pytest modules. They run their checks (and argparse) when they are imported and
need libtqenc.so, which tools/trellis/build.sh makes on a box with a compiler; collecting them would stop the whole pytest run.
Run them as  python3 tools/trellis/test_blockldl.py  and  python3 tools/trellis/test_tqenc_py.py  on that box."""
collect_ignore_glob = ["test_*.py"]
