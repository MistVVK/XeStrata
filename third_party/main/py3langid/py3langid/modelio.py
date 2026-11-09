"""Model serialization: npz inside LZMA, no pickle.

DFA rows are deduplicated: `nextmove` holds distinct 256-byte rows,
`nextmove_row` maps state → row. Both keys required; legacy models rejected.
"""

import io
import lzma
import shutil
import tempfile
from array import array

import numpy as np


def _canonical_rows(rows, row_index):
    """Sort transition rows for reproducibility; narrow to uint16 if possible."""
    uniq, index = np.unique(np.asarray(rows).reshape(-1, 256), axis=0,
                            return_inverse=True)
    dtype = np.uint16 if len(uniq) < 1 << 16 else np.uint32
    return uniq.ravel(), index.ravel()[np.asarray(row_index)].astype(dtype)


def expand_nextmove(rows, row_index):
    """Undo row sharing: one 256-entry row per state."""
    return _to_array(np.asarray(rows).reshape(-1, 256)[np.asarray(row_index)])


def save_model(path, model):
    """Write (nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_row, tk_output)."""
    nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_row, tk_output = model
    nextmove = np.asarray(tk_nextmove)
    dtype = np.uint16 if not nextmove.size or nextmove.max() < 1 << 16 else np.uint32
    rows, row_index = _canonical_rows(nextmove, tk_row)
    out_feat = np.asarray(tk_output, dtype=np.int32)
    if len(out_feat) != len(row_index):
        raise ValueError("one output slot per DFA state")
    arrays = {
        "ptc": np.asarray(nb_ptc, dtype=np.float16).reshape(-1, len(nb_pc)),
        "pc": np.asarray(nb_pc, dtype=np.float32),
        "classes": np.array(nb_classes),
        "nextmove": rows.astype(dtype),
        "nextmove_row": row_index,
        "out_feat": out_feat,
    }
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    with open(path, "wb") as f:
        f.write(lzma.compress(buffer.getvalue(), preset=6))


_TYPECODE = {2: "H", 4: "I", 8: "L"}


def _to_array(arr):
    """NumPy unsigned int array → stdlib array('H'/'I'/'L')."""
    out = array(_TYPECODE[arr.dtype.itemsize])
    out.frombytes(memoryview(np.ascontiguousarray(arr)).cast("B"))
    return out


def load_model(path):
    """Load npz+LZMA model → (nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_row, tk_output)."""
    # stream LZMA to a temp file so the uncompressed npz is never fully resident
    with tempfile.TemporaryFile(suffix=".npz") as tmp:
        with lzma.open(path) as src:
            shutil.copyfileobj(src, tmp, length=1 << 20)
        tmp.seek(0)
        with np.load(tmp, allow_pickle=False) as data:
            missing = {"nextmove_row", "out_feat"}.difference(data.files)
            if missing:
                raise ValueError(
                    f"{path}: unsupported model layout, missing "
                    f"{sorted(missing)}; retrain with py3langid.train.train")
            return (data["ptc"], data["pc"], data["classes"].tolist(),
                    _to_array(data["nextmove"]), _to_array(data["nextmove_row"]),
                    data["out_feat"].tolist())
