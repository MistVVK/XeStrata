"""Header inventory: the first 32 MiB of each GGUF, parsed with tools/gguf_reader.py.

    python inventory.py OUT_DIR REPO@REV:PATH ...
Writes OUT_DIR/<file>.csv (every tensor) and prints, per file, the type counts and the expert types per layer.
"""
import collections, pathlib, sys, tempfile, urllib.request
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))
from gguf_reader import GGUFFile

out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for spec in sys.argv[2:]:
    repo_rev, path = spec.split(":", 1)
    repo, rev = repo_rev.split("@")
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{path}"
    name = path.split("/")[-1]
    req = urllib.request.Request(url, headers={"Range": "bytes=0-33554431", "User-Agent": "strata-inventory"})
    data = urllib.request.urlopen(req, timeout=300).read()
    with tempfile.NamedTemporaryFile(suffix=".gguf") as f:
        f.write(data); f.flush()
        g = GGUFFile(pathlib.Path(f.name))
    rows = ["name,type_id,type,shape_ggml_order,bytes"]
    for t in g.tensors:
        rows.append(f"{t.name},{t.type_id},{t.type_name},{'x'.join(map(str, t.shape))},{t.expected_bytes() or ''}")
    (out / (name + ".csv")).write_text("\n".join(rows) + "\n")
    print(f"== {name}  ({len(g.tensors)} tensors)")
    meta = {k: v for k, v in g.metadata.items() if any(s in k for s in ("expert_count", "expert_used", "architecture", "block_count"))}
    print("   meta:", meta)
    print("   types:", dict(sorted(g.by_type().items(), key=lambda kv: -kv[1])))
    exp = collections.defaultdict(dict)
    for t in g.tensors:
        for role in ("gate", "up", "down"):
            if t.name.endswith(f".ffn_{role}_exps.weight"):
                exp[int(t.name.split(".")[1])][role] = (t.type_name, "x".join(map(str, t.shape)))
    combos = collections.Counter((v.get("gate", ("-",))[0], v.get("up", ("-",))[0], v.get("down", ("-",))[0]) for v in exp.values())
    for c, n in combos.most_common():
        layers = [l for l, v in sorted(exp.items()) if (v.get("gate", ("-",))[0], v.get("up", ("-",))[0], v.get("down", ("-",))[0]) == c]
        shape = exp[layers[0]].get("gate", ("", "?"))[1]
        print(f"   experts gate/up/down {c}: {n} layers {layers[:12]}{'...' if len(layers) > 12 else ''}  gate shape {shape}")
