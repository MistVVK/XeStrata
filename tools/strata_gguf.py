"""tools/strata_gguf.py - Strata's own GGUF reader and writer for the preparation tools (replaces llama.cpp's gguf-py in them).

It covers what the tools use, with gguf-py's names so a tool runs on either (see tools/_paths.py gguf_lib(); STRATA_GGUF_PY still selects
the vendored gguf-py for two releases):

    GGMLQuantizationType, GGML_QUANT_SIZES, GGUFValueType, GGUF_MAGIC, GGUF_VERSION, GGUF_DEFAULT_ALIGNMENT
    quant_shape_to_byte_shape / quant_shape_from_byte_shape
    GGUFReader(path)    .data (memmap of the file) .data_offset .alignment .fields[key] (.types .contents()) .tensors
                        each tensor: .name .tensor_type .shape (dims, ggml order) .n_elements .n_bytes .data_offset .data .field
    GGUFWriter(path, arch)   add_* / add_key_value / add_tensor(name, array, raw_shape=, raw_dtype=) / write_header_to_file /
                        write_kv_data_to_file / write_tensors_to_file / close; one file, little-endian, no splitting, no lazy tensors
    quants              strata_quants: dequantize / quantize

A file written here is byte-identical to gguf-py's for the same calls (tools/test_strata_gguf.py compares them). Reading handles GGUF v2 and v3,
little-endian only. The tools' lightweight header parser is tools/gguf_reader.py (GGUFFile); this module is the one that gives tensor data.
"""
from __future__ import annotations

import os
import struct
import sys
from collections import OrderedDict
from enum import IntEnum
from typing import Any, NamedTuple, Sequence

import numpy as np

import strata_quants as quants

GGUF_MAGIC = 0x46554747          # "GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32
GGML_MAX_DIMS = 4
GGUF_MAX_STRING_LENGTH = 1024 * 1024 * 1024
GGUF_MAX_ARRAY_ELEMENTS = 1024 * 1024 * 1024
READER_SUPPORTED_VERSIONS = (2, GGUF_VERSION)


class GGMLQuantizationType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    TQ1_0 = 34
    TQ2_0 = 35
    MXFP4 = 39
    NVFP4 = 40
    Q1_0 = 41
    Q2_0 = 42
    # Agention's llama.cpp fork (the APR trellis expert types; see tools/gguf_reader.py): gguf-py cannot open these, we can
    PTQ1_0 = 143
    TQ2_T = 144
    TQK6 = 145
    TQK7 = 146
    TQP5 = 147
    TQP6 = 148
    TQP7 = 149


_Q = GGMLQuantizationType
#: (elements per block, bytes per block): ggml's block structs (third_party/strata-ggml/strata-quant-common.h)
GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {
    _Q.F32: (1, 4), _Q.F16: (1, 2), _Q.Q4_0: (32, 18), _Q.Q4_1: (32, 20), _Q.Q5_0: (32, 22), _Q.Q5_1: (32, 24),
    _Q.Q8_0: (32, 34), _Q.Q8_1: (32, 36), _Q.Q2_K: (256, 84), _Q.Q3_K: (256, 110), _Q.Q4_K: (256, 144),
    _Q.Q5_K: (256, 176), _Q.Q6_K: (256, 210), _Q.Q8_K: (256, 292), _Q.IQ2_XXS: (256, 66), _Q.IQ2_XS: (256, 74),
    _Q.IQ3_XXS: (256, 98), _Q.IQ1_S: (256, 50), _Q.IQ4_NL: (32, 18), _Q.IQ3_S: (256, 110), _Q.IQ2_S: (256, 82),
    _Q.IQ4_XS: (256, 136), _Q.I8: (1, 1), _Q.I16: (1, 2), _Q.I32: (1, 4), _Q.I64: (1, 8), _Q.F64: (1, 8),
    _Q.IQ1_M: (256, 56), _Q.BF16: (1, 2), _Q.TQ1_0: (256, 54), _Q.TQ2_0: (256, 66), _Q.MXFP4: (32, 17),
    _Q.NVFP4: (64, 36), _Q.Q1_0: (128, 18), _Q.Q2_0: (64, 18),
    _Q.PTQ1_0: (128, 28), _Q.TQ2_T: (128, 34), _Q.TQK6: (128, 26), _Q.TQK7: (128, 30),
    _Q.TQP5: (128, 42), _Q.TQP6: (128, 50), _Q.TQP7: (128, 58),
}


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

    @staticmethod
    def get_type(val: Any) -> "GGUFValueType":
        if isinstance(val, (str, bytes, bytearray)):
            return GGUFValueType.STRING
        if isinstance(val, list):
            return GGUFValueType.ARRAY
        if isinstance(val, float):
            return GGUFValueType.FLOAT32
        if isinstance(val, bool):
            return GGUFValueType.BOOL
        if isinstance(val, int):
            return GGUFValueType.INT32
        raise ValueError(f"Unknown type: {type(val)}")


_V = GGUFValueType
_SCALAR = {_V.UINT8: ("B", np.uint8), _V.INT8: ("b", np.int8), _V.UINT16: ("H", np.uint16), _V.INT16: ("h", np.int16),
           _V.UINT32: ("I", np.uint32), _V.INT32: ("i", np.int32), _V.FLOAT32: ("f", np.float32), _V.UINT64: ("Q", np.uint64),
           _V.INT64: ("q", np.int64), _V.FLOAT64: ("d", np.float64), _V.BOOL: ("?", np.bool_)}


def quant_shape_to_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType) -> tuple[int, ...]:
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % block_size != 0:
        raise ValueError(f"Quantized tensor row size ({shape[-1]}) is not a multiple of {quant_type.name} block size ({block_size})")
    return (*shape[:-1], shape[-1] // block_size * type_size)


def quant_shape_from_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType) -> tuple[int, ...]:
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % type_size != 0:
        raise ValueError(f"Quantized tensor bytes per row ({shape[-1]}) is not a multiple of {quant_type.name} type size ({type_size})")
    return (*shape[:-1], shape[-1] // type_size * block_size)


# --------------------------------------------------------------------------------------------------------------------- reader
class ReaderField(NamedTuple):
    offset: int
    name: str
    types: list
    value: Any = None          # the parsed value (a str, a number, a bool, or a list)

    def contents(self, index_or_slice: int | slice = slice(None)) -> Any:
        if not self.types:
            return None
        if self.types[0] == GGUFValueType.ARRAY:
            return self.value[index_or_slice]
        return self.value


class ReaderTensor(NamedTuple):
    name: str
    tensor_type: GGMLQuantizationType
    shape: np.ndarray
    n_elements: int
    n_bytes: int
    data_offset: int
    data: np.ndarray
    field: Any


_NP_DATA = {_Q.F16: np.float16, _Q.F32: np.float32, _Q.F64: np.float64, _Q.I8: np.int8, _Q.I16: np.int16, _Q.I32: np.int32, _Q.I64: np.int64}


class GGUFReader:
    alignment: int = GGUF_DEFAULT_ALIGNMENT
    data_offset: int

    def __init__(self, path, mode: str = "r"):
        self.data = np.memmap(path, mode=mode)
        self._size = int(self.data.nbytes)
        self._buf = memoryview(self.data)            # byte offsets index this; struct reads from it
        self._pos = 0
        if self._size < 24:
            raise ValueError("not a GGUF file (too short)")
        if self._u("<I") != GGUF_MAGIC:
            raise ValueError("GGUF magic invalid")
        version = self._u("<I")
        if version & 65535 == 0:
            raise ValueError("this GGUF is big-endian, which is not supported")
        if version not in READER_SUPPORTED_VERSIONS:
            raise ValueError(f"Sorry, file appears to be version {version} which we cannot handle")
        self.fields: OrderedDict[str, ReaderField] = OrderedDict()
        self.tensors: list[ReaderTensor] = []
        tensor_count, kv_count = self._u("<Q"), self._u("<Q")
        if tensor_count > GGUF_MAX_ARRAY_ELEMENTS:
            raise ValueError(f"Tensor count {tensor_count} exceeds maximum {GGUF_MAX_ARRAY_ELEMENTS}")
        if kv_count > GGUF_MAX_ARRAY_ELEMENTS:
            raise ValueError(f"KV count {kv_count} exceeds maximum {GGUF_MAX_ARRAY_ELEMENTS}")
        self.fields["GGUF.version"] = ReaderField(4, "GGUF.version", [_V.UINT32], version)
        self.fields["GGUF.tensor_count"] = ReaderField(8, "GGUF.tensor_count", [_V.UINT64], tensor_count)
        self.fields["GGUF.kv_count"] = ReaderField(16, "GGUF.kv_count", [_V.UINT64], kv_count)
        for _ in range(kv_count):
            start = self._pos
            key = self._str()
            vt = self._u("<I")
            types, value = self._value(vt)
            if key in self.fields:
                raise KeyError(f"Duplicate {key} already in list at offset {start}")
            self.fields[key] = ReaderField(start, key, types, value)
        infos = [self._tensor_info() for _ in range(tensor_count)]
        align = self.fields.get("general.alignment")
        if align is not None:
            if align.types != [_V.UINT32]:
                raise ValueError("Bad type for general.alignment field")
            self.alignment = int(align.value)
            if self.alignment == 0 or (self.alignment & (self.alignment - 1)) != 0:
                raise ValueError("Invalid alignment: must be a non-zero power of two")
        pad = self._pos % self.alignment
        self.data_offset = self._pos + (self.alignment - pad if pad else 0)
        self._build_tensors(infos)

    # -- parsing
    def _need(self, n: int) -> None:
        if n < 0 or self._pos + n > self._size:
            raise ValueError(f"header ends early: {n} bytes needed at offset {self._pos}, {self._size - self._pos} left")

    def _u(self, fmt: str):
        n = struct.calcsize(fmt)
        self._need(n)
        (v,) = struct.unpack_from(fmt, self._buf, self._pos)
        self._pos += n
        return v

    def _str(self) -> str:
        n = self._u("<Q")
        if n > GGUF_MAX_STRING_LENGTH:
            raise ValueError(f"String length {n} exceeds maximum {GGUF_MAX_STRING_LENGTH}")
        self._need(n)
        s = bytes(self._buf[self._pos:self._pos + n]).decode("utf-8")
        self._pos += n
        return s

    def _value(self, raw_type: int):
        try:
            vt = GGUFValueType(raw_type)
        except ValueError:
            raise ValueError(f"Unknown/unhandled field type {raw_type}") from None
        if vt == _V.STRING:
            return [vt], self._str()
        if vt in _SCALAR:
            return [vt], self._u("<" + _SCALAR[vt][0])
        # an array: element type, count, then the elements
        et = GGUFValueType(self._u("<I"))
        count = self._u("<Q")
        if count > GGUF_MAX_ARRAY_ELEMENTS:
            raise ValueError(f"Array length {count} exceeds maximum {GGUF_MAX_ARRAY_ELEMENTS}")
        if et == _V.ARRAY:
            raise ValueError("nested metadata arrays are not supported")
        if et == _V.STRING:
            self._need(count * 8)
            return [vt, et], [self._str() for _ in range(count)]
        fmt, npt = _SCALAR[et]
        size = np.dtype(npt).itemsize
        self._need(count * size)
        arr = np.frombuffer(self._buf, dtype=np.dtype(npt).newbyteorder("<"), count=count, offset=self._pos)
        self._pos += count * size
        return [vt, et], arr.tolist()

    def _tensor_info(self):
        name = self._str()
        n_dims = self._u("<I")
        if n_dims > GGML_MAX_DIMS:
            raise ValueError(f"Tensor dimensions count {n_dims} exceeds GGML_MAX_DIMS ({GGML_MAX_DIMS})")
        dims = [self._u("<Q") for _ in range(n_dims)]
        raw_type = self._u("<I")
        offset = self._u("<Q")
        return name, dims, raw_type, offset

    def _build_tensors(self, infos) -> None:
        seen = set()
        for name, dims, raw_type, offset in infos:
            if name in seen:
                raise ValueError(f"Found duplicated tensor with name {name}")
            seen.add(name)
            try:
                ggml_type = GGMLQuantizationType(raw_type)
            except ValueError:
                raise ValueError(f"{raw_type} is not a valid GGMLQuantizationType") from None
            n_elems = 1
            for d in dims:
                n_elems *= int(d)
            np_dims = tuple(reversed(dims))
            block_size, type_size = GGML_QUANT_SIZES[ggml_type]
            n_bytes = n_elems * type_size // block_size
            start = self.data_offset + offset
            if ggml_type in _NP_DATA:
                item_count, item_type = n_elems, _NP_DATA[ggml_type]
            else:
                item_count, item_type = n_bytes, np.uint8
                np_dims = quant_shape_to_byte_shape(np_dims, ggml_type)
            nbytes = item_count * np.dtype(item_type).itemsize
            if start + nbytes > self._size:
                raise ValueError(f"tensor {name} ends at byte {start + nbytes}, past the end of the file ({self._size})")
            data = self.data[start:start + nbytes].view(dtype=item_type)[:item_count].reshape(np_dims)
            self.tensors.append(ReaderTensor(name, ggml_type, np.array(dims, dtype=np.uint64), n_elems, n_bytes, start, data,
                                             self.fields.get(name)))

    def get_field(self, key: str):
        return self.fields.get(key)

    def get_tensor(self, idx: int) -> ReaderTensor:
        return self.tensors[idx]


# --------------------------------------------------------------------------------------------------------------------- writer
class _TI:
    __slots__ = ("shape", "dtype", "nbytes", "tensor")

    def __init__(self, shape, dtype, nbytes, tensor=None):
        self.shape, self.dtype, self.nbytes, self.tensor = shape, dtype, nbytes, tensor


class _KV:
    __slots__ = ("value", "type", "sub_type")

    def __init__(self, value, type, sub_type=None):
        self.value, self.type, self.sub_type = value, type, sub_type


class GGUFWriter:
    """One output file, little-endian. The sequence is gguf-py's: add the KVs and tensors, then write_header_to_file(),
    write_kv_data_to_file(), write_tensors_to_file(), close(). The tensor data is held (not copied) until it is written."""

    def __init__(self, path, arch: str, use_temp_file: bool = False, endianess=None, split_max_tensors: int = 0, split_max_size: int = 0,
                 dry_run: bool = False, small_first_shard: bool = False):
        if split_max_tensors or split_max_size or small_first_shard:
            raise NotImplementedError("strata_gguf writes one file; splitting is not supported")
        if endianess not in (None, "little") and getattr(endianess, "name", "LITTLE") != "LITTLE":
            raise NotImplementedError("strata_gguf writes little-endian files only")
        self.path = os.fspath(path) if path else None
        self.arch = arch
        self.data_alignment = GGUF_DEFAULT_ALIGNMENT
        self.tensors: dict[str, _TI] = {}
        self.kv_data: dict[str, _KV] = {}
        self.fout = None
        self.state = "no_file"
        self.add_string("general.architecture", arch)

    # -- key/values
    def add_key_value(self, key: str, val: Any, vtype: GGUFValueType, sub_type: GGUFValueType | None = None) -> None:
        self.kv_data[key] = _KV(val, GGUFValueType(vtype), None if sub_type is None else GGUFValueType(sub_type))

    def remove_key(self, key: str) -> None:
        self.kv_data.pop(key, None)

    def add_uint8(self, key, val): self.add_key_value(key, val, _V.UINT8)
    def add_int8(self, key, val): self.add_key_value(key, val, _V.INT8)
    def add_uint16(self, key, val): self.add_key_value(key, val, _V.UINT16)
    def add_int16(self, key, val): self.add_key_value(key, val, _V.INT16)
    def add_uint32(self, key, val): self.add_key_value(key, val, _V.UINT32)
    def add_int32(self, key, val): self.add_key_value(key, val, _V.INT32)
    def add_float32(self, key, val): self.add_key_value(key, val, _V.FLOAT32)
    def add_uint64(self, key, val): self.add_key_value(key, val, _V.UINT64)
    def add_int64(self, key, val): self.add_key_value(key, val, _V.INT64)
    def add_float64(self, key, val): self.add_key_value(key, val, _V.FLOAT64)
    def add_bool(self, key, val): self.add_key_value(key, val, _V.BOOL)

    def add_string(self, key: str, val: str) -> None:
        if not val:
            return
        self.add_key_value(key, val, _V.STRING)

    def add_array(self, key: str, val: Sequence[Any]) -> None:
        if len(val) == 0:
            return
        self.add_key_value(key, val, _V.ARRAY)

    def add_custom_alignment(self, alignment: int) -> None:
        if alignment <= 0 or (alignment & (alignment - 1)) != 0:
            raise ValueError("Invalid alignment: must be a non-zero power of two")
        self.data_alignment = alignment
        self.add_uint32("general.alignment", alignment)

    def add_quantization_version(self, quantization_version: int) -> None:
        self.add_uint32("general.quantization_version", quantization_version)

    # -- tensors
    @staticmethod
    def ggml_pad(x: int, n: int) -> int:
        return ((x + n - 1) // n) * n

    def add_tensor_info(self, name: str, tensor_shape: Sequence[int], tensor_dtype, tensor_nbytes: int, raw_dtype=None) -> None:
        if self.state != "no_file":
            raise ValueError(f"Expected output file to be not yet opened, got {self.state}")
        if name in self.tensors:
            raise ValueError(f"Duplicated tensor name {name!r}")
        if raw_dtype is None:
            dt = np.dtype(tensor_dtype)
            known = {np.dtype(np.float16): _Q.F16, np.dtype(np.float32): _Q.F32, np.dtype(np.float64): _Q.F64, np.dtype(np.int8): _Q.I8,
                     np.dtype(np.int16): _Q.I16, np.dtype(np.int32): _Q.I32, np.dtype(np.int64): _Q.I64}
            if dt not in known:
                raise ValueError("Only F16, F32, F64, I8, I16, I32, I64 tensors are supported for now")
            dtype = known[dt]
        else:
            dtype = GGMLQuantizationType(raw_dtype)
            if np.dtype(tensor_dtype) == np.uint8:
                tensor_shape = quant_shape_from_byte_shape(tensor_shape, dtype)
        self.tensors[name] = _TI(tuple(tensor_shape), dtype, tensor_nbytes)

    def add_tensor(self, name: str, tensor: np.ndarray, raw_shape: Sequence[int] | None = None, raw_dtype=None, tensor_endianess=None) -> None:
        shape = raw_shape if raw_shape is not None else tensor.shape
        self.add_tensor_info(name, shape, tensor.dtype, tensor.nbytes, raw_dtype=raw_dtype)
        self.tensors[name].tensor = tensor

    # -- serialisation
    def _pack_val(self, val: Any, vtype: GGUFValueType, add_vtype: bool, sub_type=None) -> bytes:
        out = bytearray()
        if add_vtype:
            out += struct.pack("<I", vtype)
        sc = _SCALAR.get(vtype)
        if sc is not None:
            out += struct.pack(("" if vtype == _V.BOOL else "<") + sc[0], val)
        elif vtype == _V.STRING:
            enc = val.encode("utf-8") if isinstance(val, str) else val
            out += struct.pack("<Q", len(enc)) + enc
        elif vtype == _V.ARRAY:
            if not isinstance(val, Sequence):
                raise ValueError("Invalid GGUF metadata array, expecting sequence")
            if len(val) == 0:
                raise ValueError("Invalid GGUF metadata array. Empty array")
            if sub_type is not None:
                ltype = sub_type
            elif isinstance(val, bytes):
                ltype = _V.UINT8
            else:
                ltype = GGUFValueType.get_type(val[0])
                if not all(GGUFValueType.get_type(i) is ltype for i in val[1:]):
                    raise ValueError("All items in a GGUF array should be of the same type")
            out += struct.pack("<I", ltype) + struct.pack("<Q", len(val))
            if ltype in _SCALAR and ltype != _V.BOOL:        # a long numeric array in one call (same bytes as one pack per item)
                out += struct.pack("<%d%s" % (len(val), _SCALAR[ltype][0]), *val)
            else:
                for item in val:
                    out += self._pack_val(item, ltype, add_vtype=False)
        else:
            raise ValueError("Invalid GGUF metadata value type or value")
        return bytes(out)

    def open_output_file(self, path=None) -> None:
        if self.state == "empty" and self.fout is not None and (path is None or os.fspath(path) == self.path):
            return
        if self.state != "no_file":
            raise ValueError(f"Expected output file to be not yet opened, got {self.state}")
        if path is not None:
            self.path = os.fspath(path)
        if self.path is not None:
            self.fout = open(self.path, "wb")
            self.state = "empty"

    def write_header_to_file(self, path=None) -> None:
        self.open_output_file(path)
        if self.state != "empty":
            raise ValueError(f"Expected output file to be empty, got {self.state}")
        self.fout.write(struct.pack("<I", GGUF_MAGIC))
        self.fout.write(struct.pack("<I", GGUF_VERSION))
        self.fout.write(struct.pack("<Q", len(self.tensors)))
        self.fout.write(struct.pack("<Q", len(self.kv_data)))
        self.fout.flush()
        self.state = "header"

    def write_kv_data_to_file(self) -> None:
        if self.state != "header":
            raise ValueError(f"Expected output file to contain the header, got {self.state}")
        kv = bytearray()
        for key, val in self.kv_data.items():
            kv += self._pack_val(key, _V.STRING, add_vtype=False)
            kv += self._pack_val(val.value, val.type, add_vtype=True, sub_type=val.sub_type)
        self.fout.write(kv)
        self.fout.flush()
        self.state = "kv"

    def write_ti_data_to_file(self) -> None:
        if self.state != "kv":
            raise ValueError(f"Expected output file to contain KV data, got {self.state}")
        ti = bytearray()
        offset = 0
        for name, t in self.tensors.items():
            ti += self._pack_val(name, _V.STRING, add_vtype=False)
            n = len(t.shape)
            ti += struct.pack("<I", n)
            for j in range(n):
                ti += struct.pack("<Q", t.shape[n - 1 - j])
            ti += struct.pack("<I", int(t.dtype))
            ti += struct.pack("<Q", offset)
            offset += self.ggml_pad(t.nbytes, self.data_alignment)
        self.fout.write(ti)
        self.fout.flush()
        self.state = "ti"

    def _pad(self, n: int) -> None:
        pad = self.ggml_pad(n, self.data_alignment) - n
        if pad:
            self.fout.write(bytes(pad))

    def write_tensors_to_file(self, *, progress: bool = False) -> None:
        self.write_ti_data_to_file()
        self._pad(self.fout.tell())
        for name, t in self.tensors.items():
            if t.tensor is None or t.tensor.nbytes != t.nbytes:
                raise ValueError(f"tensor {name!r} has no data or the wrong size")
            start = self.fout.tell()
            np.ascontiguousarray(t.tensor).tofile(self.fout)
            if self.fout.tell() - start != t.nbytes:      # a short write would only show as a corrupt file at load time
                raise ValueError(f"tensor {name!r} wrote {self.fout.tell() - start} bytes, expected {t.nbytes}")
            self._pad(t.nbytes)
            t.tensor = None
        self.state = "weights"

    def flush(self) -> None:
        if self.fout is not None:
            self.fout.flush()

    def close(self) -> None:
        if self.fout is not None:
            self.fout.close()
            self.fout = None


if __name__ == "__main__":
    sys.exit("strata_gguf is a library: see tools/test_strata_gguf.py")
