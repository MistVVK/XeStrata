"""cfgrun.py NAME [--drop FLAG[=N]]... -- EXTRA...: runs the engine as setup's UD config starts it, plus EXTRA
(output NAME.txt, logits NAME.logits); --drop removes a flag (and its value) from the config's args.  CFG: the config
setup wrote (xestrata-unsloth-ud-q4_k_xl.json in the XeStrata folder)."""
import json
import os
import subprocess
import sys

H = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.environ["CFG"]))
name, rest = sys.argv[1], sys.argv[2:]
cut = rest.index("--") if "--" in rest else len(rest)
drops, extra = rest[:cut], rest[cut + 1:]
args = list(cfg["args"])
for i in range(0, len(drops), 2):
    flag, n = drops[i + 1].split("=") if "=" in drops[i + 1] else (drops[i + 1], 0)
    k = args.index(flag)
    del args[k:k + 1 + int(n)]
env = dict(os.environ, LD_LIBRARY_PATH=":".join(cfg["lib_dirs"]), ZES_ENABLE_SYSMAN="1")
cmd = [cfg["exe"], *args, "--tokens-file", f"{H}/chat_ids.txt", "--max-new", "64",
       "--dump-logits", f"{H}/{name}.logits", *extra]
with open(f"{H}/{name}.txt", "w") as f:
    f.write(" ".join(cmd) + "\n")
    f.flush()
    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cfg["cwd"], env=env).returncode
print(name, "rc", rc)
