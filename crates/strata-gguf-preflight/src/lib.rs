//! Bounded, constant-memory read of GGUF v3 metadata needed for Strata import preview.
//!
//! Reads tensor directory entries and checks their declared byte ranges without
//! reading tensor payloads or establishing provenance. Mirrors `strata_control/gguf_preflight.py`.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_METADATA_KEYS: u64 = 4096;
pub const MAX_STRING_BYTES: u64 = 1024 * 1024;
pub const MAX_ARRAY_ITEMS: u64 = 500_000;
pub const MAX_TENSORS: u64 = 100_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreflightError {
    pub message: String,
    pub code: i32,
}

impl PreflightError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: 5,
        }
    }
}

impl fmt::Display for PreflightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for PreflightError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreflightReport {
    pub architecture: Option<String>,
    pub projector_type: Option<String>,
    pub tokenizer_model: Option<String>,
    pub tokenizer_tokens_present: bool,
    pub tokenizer_merges_present: bool,
    pub tensor_count_reported: u64,
    pub tensor_count_scanned: u64,
    pub tensor_types: BTreeMap<String, u64>,
    pub expert_tensor_types: BTreeMap<String, u64>,
    pub header_bytes_scanned: u64,
    pub payload_bytes_required: u64,
    pub payload_extent_valid: bool,
}

impl PreflightReport {
    pub fn to_json(&self) -> String {
        fn opt_str(v: &Option<String>) -> String {
            match v {
                Some(s) => json_escape(s),
                None => "null".to_string(),
            }
        }
        fn map_json(m: &BTreeMap<String, u64>) -> String {
            let mut out = String::from("{");
            for (i, (k, v)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format!("{}:{}", json_escape(k), v));
            }
            out.push('}');
            out
        }
        format!(
            "{{\"architecture\":{},\"projector_type\":{},\"tokenizer_model\":{},\"tokenizer_tokens_present\":{},\"tokenizer_merges_present\":{},\"tensor_count_reported\":{},\"tensor_count_scanned\":{},\"tensor_types\":{},\"expert_tensor_types\":{},\"header_bytes_scanned\":{},\"payload_bytes_required\":{},\"payload_extent_valid\":{}}}",
            opt_str(&self.architecture),
            opt_str(&self.projector_type),
            opt_str(&self.tokenizer_model),
            self.tokenizer_tokens_present,
            self.tokenizer_merges_present,
            self.tensor_count_reported,
            self.tensor_count_scanned,
            map_json(&self.tensor_types),
            map_json(&self.expert_tensor_types),
            self.header_bytes_scanned,
            self.payload_bytes_required,
            self.payload_extent_valid,
        )
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn scalar_bytes(kind: u32) -> Option<u64> {
    match kind {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

/// Returns `(type_name, block_elements, block_bytes)` matching `tools/gguf_reader.py`.
fn ggml_type_geometry(type_id: u32) -> Option<(&'static str, u64, u64)> {
    match type_id {
        0 => Some(("F32", 1, 4)),
        1 => Some(("F16", 1, 2)),
        2 => Some(("Q4_0", 32, 18)),
        3 => Some(("Q4_1", 32, 20)),
        6 => Some(("Q5_0", 32, 22)),
        7 => Some(("Q5_1", 32, 24)),
        8 => Some(("Q8_0", 32, 34)),
        9 => Some(("Q8_1", 32, 36)),
        10 => Some(("Q2_K", 256, 84)),
        11 => Some(("Q3_K", 256, 110)),
        12 => Some(("Q4_K", 256, 144)),
        13 => Some(("Q5_K", 256, 176)),
        14 => Some(("Q6_K", 256, 210)),
        15 => Some(("Q8_K", 256, 292)),
        16 => Some(("IQ2_XXS", 256, 66)),
        17 => Some(("IQ2_XS", 256, 74)),
        18 => Some(("IQ3_XXS", 256, 98)),
        19 => Some(("IQ1_S", 256, 50)),
        20 => Some(("IQ4_NL", 32, 18)),
        21 => Some(("IQ3_S", 256, 110)),
        22 => Some(("IQ2_S", 256, 82)),
        23 => Some(("IQ4_XS", 256, 136)),
        24 => Some(("I8", 1, 1)),
        25 => Some(("I16", 1, 2)),
        26 => Some(("I32", 1, 4)),
        27 => Some(("I64", 1, 8)),
        28 => Some(("F64", 1, 8)),
        29 => Some(("IQ1_M", 256, 56)),
        30 => Some(("BF16", 1, 2)),
        39 => Some(("MXFP4", 32, 17)),
        40 => Some(("NVFP4", 64, 36)),
        42 => Some(("Q2_0", 64, 18)),
        143 => Some(("PTQ1_0", 128, 28)),
        144 => Some(("TQ2_T", 128, 34)),
        145 => Some(("TQK6", 128, 26)),
        146 => Some(("TQK7", 128, 30)),
        147 => Some(("TQP5", 128, 42)),
        148 => Some(("TQP6", 128, 50)),
        149 => Some(("TQP7", 128, 58)),
        _ => None,
    }
}

struct BoundedReader<R: Read + Seek> {
    stream: R,
    limit: u64,
    used: u64,
}

impl<R: Read + Seek> BoundedReader<R> {
    fn new(stream: R, file_size: u64) -> Self {
        Self {
            stream,
            limit: file_size.min(MAX_HEADER_BYTES),
            used: 0,
        }
    }

    fn check_avail(&self, count: u64) -> Result<(), PreflightError> {
        match self.used.checked_add(count) {
            Some(end) if end <= self.limit => Ok(()),
            _ => Err(PreflightError::new(
                "GGUF metadata exceeds the bounded inspection limit",
            )),
        }
    }

    fn take_exact(&mut self, buf: &mut [u8]) -> Result<(), PreflightError> {
        let count = buf.len() as u64;
        self.check_avail(count)?;
        let mut read = 0usize;
        while read < buf.len() {
            match self.stream.read(&mut buf[read..]) {
                Ok(0) => return Err(PreflightError::new("GGUF metadata is truncated")),
                Ok(n) => read += n,
                Err(_) => return Err(PreflightError::new("GGUF metadata could not be read")),
            }
        }
        self.used += count;
        Ok(())
    }

    fn skip(&mut self, count: u64) -> Result<(), PreflightError> {
        self.check_avail(count)?;
        if count <= 64 * 1024 {
            let mut left = count as usize;
            let mut scratch = [0u8; 4096];
            while left > 0 {
                let chunk = left.min(scratch.len());
                match self.stream.read(&mut scratch[..chunk]) {
                    Ok(0) => return Err(PreflightError::new("GGUF metadata is truncated")),
                    Ok(n) => left -= n,
                    Err(_) => return Err(PreflightError::new("GGUF metadata could not be read")),
                }
            }
            self.used += count;
            return Ok(());
        }
        let offset = i64::try_from(count)
            .map_err(|_| PreflightError::new("GGUF metadata exceeds the bounded inspection limit"))?;
        self.stream
            .seek(SeekFrom::Current(offset))
            .map_err(|_| PreflightError::new("GGUF metadata could not be read"))?;
        self.used += count;
        Ok(())
    }

    fn u32(&mut self) -> Result<u32, PreflightError> {
        let mut buf = [0u8; 4];
        self.take_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn u64(&mut self) -> Result<u64, PreflightError> {
        let mut buf = [0u8; 8];
        self.take_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn string(&mut self, capture: bool) -> Result<Option<String>, PreflightError> {
        let size = self.u64()?;
        if size > MAX_STRING_BYTES {
            return Err(PreflightError::new(
                "GGUF metadata string is too large to inspect",
            ));
        }
        if !capture {
            self.skip(size)?;
            return Ok(None);
        }
        let mut buf = vec![0u8; size as usize];
        self.take_exact(&mut buf)?;
        let s = String::from_utf8(buf)
            .map_err(|_| PreflightError::new("GGUF metadata has invalid UTF-8"))?;
        Ok(Some(s))
    }

    fn skip_or_capture_value(
        &mut self,
        kind: u32,
        capture_str: bool,
        capture_uint: bool,
    ) -> Result<MetaValue, PreflightError> {
        match kind {
            8 => Ok(MetaValue::String(self.string(capture_str)?)),
            9 => {
                let elem_type = self.u32()?;
                let count = self.u64()?;
                if count > MAX_ARRAY_ITEMS {
                    return Err(PreflightError::new(
                        "GGUF metadata array exceeds the inspection limit",
                    ));
                }
                if elem_type == 8 {
                    for _ in 0..count {
                        self.string(false)?;
                    }
                } else if let Some(sz) = scalar_bytes(elem_type) {
                    let total = count.checked_mul(sz).ok_or_else(|| {
                        PreflightError::new("GGUF metadata exceeds the bounded inspection limit")
                    })?;
                    self.skip(total)?;
                } else {
                    return Err(PreflightError::new(
                        "GGUF metadata has an unsupported array type",
                    ));
                }
                Ok(MetaValue::Array(elem_type, count))
            }
            _ => {
                let sz = scalar_bytes(kind).ok_or_else(|| {
                    PreflightError::new("GGUF metadata has an unsupported value type")
                })?;
                if capture_uint && (kind == 4 || kind == 10) {
                    let v = if kind == 4 {
                        self.u32()? as u64
                    } else {
                        self.u64()?
                    };
                    return Ok(MetaValue::Uint(v));
                }
                self.skip(sz)?;
                Ok(MetaValue::Skipped)
            }
        }
    }
}

enum MetaValue {
    String(Option<String>),
    Uint(u64),
    Array(u32, u64),
    Skipped,
}

pub fn real_model_file(path: &Path) -> Result<PathBuf, PreflightError> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let is_link = fs::symlink_metadata(path)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false);
    let target = fs::canonicalize(path).map_err(|_| {
        if is_link {
            PreflightError::new(format!("{name} is a link to a file that is not there"))
        } else {
            PreflightError::new(format!("{name} is missing"))
        }
    })?;
    let meta = fs::metadata(&target).map_err(|_| PreflightError::new(format!("{name} is missing")))?;
    if !meta.is_file() {
        return Err(PreflightError::new("GGUF shard must be a regular file"));
    }
    Ok(target)
}

pub fn inspect_metadata(path: &Path) -> Result<PreflightReport, PreflightError> {
    let target = real_model_file(path)?;
    let file = File::open(&target).map_err(|_| PreflightError::new("GGUF metadata could not be read"))?;
    let details = file
        .metadata()
        .map_err(|_| PreflightError::new("GGUF metadata could not be read"))?;
    if !details.is_file() {
        return Err(PreflightError::new("GGUF shard must be a regular file"));
    }
    let file_size = details.len();
    let mut reader = BoundedReader::new(BufReader::with_capacity(64 * 1024, file), file_size);

    let mut magic = [0u8; 4];
    reader.take_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(PreflightError::new("file does not begin with a GGUF header"));
    }
    let version = reader.u32()?;
    let tensors = reader.u64()?;
    let keys = reader.u64()?;
    if version != 3 {
        return Err(PreflightError::new("Strata metadata preflight requires GGUF v3"));
    }
    if tensors > MAX_TENSORS || keys > MAX_METADATA_KEYS {
        return Err(PreflightError::new(
            "GGUF tensor or metadata count exceeds the inspection limit",
        ));
    }

    let mut architecture: Option<String> = None;
    let mut projector_type: Option<String> = None;
    let mut tokenizer_model: Option<String> = None;
    let mut tokenizer_tokens_present = false;
    let mut tokenizer_merges_present = false;
    let mut alignment: Option<u64> = None;
    let mut bad_alignment = false;

    for _ in 0..keys {
        let key = reader
            .string(true)?
            .ok_or_else(|| PreflightError::new("GGUF metadata key is invalid or too long"))?;
        if key.len() > 256 {
            return Err(PreflightError::new(
                "GGUF metadata key is invalid or too long",
            ));
        }
        let value_type = reader.u32()?;
        let is_capture_str = matches!(
            key.as_str(),
            "general.architecture" | "tokenizer.ggml.model" | "clip.projector_type"
        );
        if is_capture_str && value_type != 8 {
            return Err(PreflightError::new(
                "GGUF architecture or tokenizer model is not a string",
            ));
        }
        let is_alignment = key == "general.alignment";
        let val = reader.skip_or_capture_value(value_type, is_capture_str, is_alignment)?;
        match key.as_str() {
            "general.architecture" => {
                if let MetaValue::String(s) = val {
                    architecture = s;
                }
            }
            "clip.projector_type" => {
                if let MetaValue::String(s) = val {
                    projector_type = s;
                }
            }
            "tokenizer.ggml.model" => {
                if let MetaValue::String(s) = val {
                    tokenizer_model = s;
                }
            }
            "tokenizer.ggml.tokens" => {
                if let MetaValue::Array(elem_t, count) = val {
                    tokenizer_tokens_present = elem_t == 8 && count > 0;
                }
            }
            "tokenizer.ggml.merges" => {
                if let MetaValue::Array(elem_t, count) = val {
                    tokenizer_merges_present = elem_t == 8 && count > 0;
                }
            }
            "general.alignment" => match val {
                MetaValue::Uint(v) => {
                    alignment = Some(v);
                    bad_alignment = false;
                }
                _ => {
                    bad_alignment = true;
                }
            },
            _ => {}
        }
    }

    if bad_alignment {
        return Err(PreflightError::new("GGUF tensor alignment is invalid"));
    }
    let alignment = alignment.unwrap_or(32);
    if !(1..=4096).contains(&alignment) || (alignment & (alignment - 1)) != 0 {
        return Err(PreflightError::new("GGUF tensor alignment is invalid"));
    }

    let mut tensor_types: BTreeMap<String, u64> = BTreeMap::new();
    let mut expert_tensor_types: BTreeMap<String, u64> = BTreeMap::new();
    let mut required_payload_bytes: u64 = 0;
    let mut extents: Vec<(u64, u64)> = Vec::with_capacity(tensors as usize);

    for _ in 0..tensors {
        let name = reader
            .string(true)?
            .ok_or_else(|| PreflightError::new("GGUF tensor name is invalid or too long"))?;
        if name.is_empty() || name.len() > 256 {
            return Err(PreflightError::new(
                "GGUF tensor name is invalid or too long",
            ));
        }
        let dimensions = reader.u32()?;
        if !(1..=8).contains(&dimensions) {
            return Err(PreflightError::new(
                "GGUF tensor dimension count is invalid",
            ));
        }
        let mut elements: u64 = 1;
        for _ in 0..dimensions {
            let size = reader.u64()?;
            if size == 0 || size > 1_000_000_000 || elements > (i64::MAX as u64) / size {
                return Err(PreflightError::new("GGUF tensor shape is invalid"));
            }
            elements *= size;
        }
        let type_id = reader.u32()?;
        let offset = reader.u64()?;
        let (type_name, block_elems, block_bytes) = ggml_type_geometry(type_id)
            .ok_or_else(|| PreflightError::new("GGUF tensor has an unsupported encoding or shape"))?;
        if elements % block_elems != 0 {
            return Err(PreflightError::new(
                "GGUF tensor has an unsupported encoding or shape",
            ));
        }
        if offset % alignment != 0 {
            return Err(PreflightError::new("GGUF tensor offset is not aligned"));
        }
        let byte_len = (elements / block_elems)
            .checked_mul(block_bytes)
            .ok_or_else(|| PreflightError::new("GGUF tensor shape is invalid"))?;
        let needed = offset
            .checked_add(byte_len)
            .ok_or_else(|| PreflightError::new("GGUF tensor shape is invalid"))?;
        required_payload_bytes = required_payload_bytes.max(needed);
        extents.push((offset, needed));
        *tensor_types.entry(type_name.to_string()).or_insert(0) += 1;
        if name.contains("_exps") || name.contains(".experts.") {
            *expert_tensor_types.entry(type_name.to_string()).or_insert(0) += 1;
        }
    }

    let data_start = reader
        .used
        .checked_add(alignment - 1)
        .map(|v| (v / alignment) * alignment)
        .ok_or_else(|| PreflightError::new("GGUF tensor alignment is invalid"))?;

    extents.sort_unstable();
    for pair in extents.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(PreflightError::new("GGUF tensor byte ranges overlap"));
        }
    }
    if tensors > 0
        && data_start
            .checked_add(required_payload_bytes)
            .map_or(true, |end| end > file_size)
    {
        return Err(PreflightError::new(
            "GGUF tensor payload is shorter than its directory requires",
        ));
    }
    if tensors == 1 {
        let end = data_start + required_payload_bytes;
        if file_size.saturating_sub(end) >= 64 {
            return Err(PreflightError::new(
                "single-tensor GGUF shard has unexplained trailing bytes",
            ));
        }
    }

    Ok(PreflightReport {
        architecture,
        projector_type,
        tokenizer_model,
        tokenizer_tokens_present,
        tokenizer_merges_present,
        tensor_count_reported: tensors,
        tensor_count_scanned: tensors,
        tensor_types,
        expert_tensor_types,
        header_bytes_scanned: reader.used,
        payload_bytes_required: required_payload_bytes,
        payload_extent_valid: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("strata_gguf_test_{name}_{nanos}.gguf"))
    }

    fn push_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn build_gguf(
        tensors: &[(&str, &[u64], u32, u64)],
        arch: Option<&str>,
        payload_len: usize,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        let key_count = if arch.is_some() { 2u64 } else { 0u64 };
        buf.extend_from_slice(&key_count.to_le_bytes());
        if let Some(a) = arch {
            push_str(&mut buf, "general.architecture");
            buf.extend_from_slice(&8u32.to_le_bytes());
            push_str(&mut buf, a);

            push_str(&mut buf, "tokenizer.ggml.tokens");
            buf.extend_from_slice(&9u32.to_le_bytes());
            buf.extend_from_slice(&8u32.to_le_bytes());
            buf.extend_from_slice(&2u64.to_le_bytes());
            push_str(&mut buf, "a");
            push_str(&mut buf, "b");
        }
        for (name, dims, tid, off) in tensors {
            push_str(&mut buf, name);
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in *dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&tid.to_le_bytes());
            buf.extend_from_slice(&off.to_le_bytes());
        }
        let aligned = buf.len().div_ceil(32) * 32;
        buf.resize(aligned + payload_len, 0);
        buf
    }

    #[test]
    fn valid_two_tensor_shard() {
        let p = temp_path("valid");
        // F32 4 elems = 16 B at off 0; Q4_0 32 elems = 18 B at off 32 -> needed 50 B
        let data = build_gguf(
            &[("blk.0.norm.weight", &[4], 0, 0), ("blk.0.ffn_gate_exps.weight", &[32], 2, 32)],
            Some("qwen3next"),
            50,
        );
        File::create(&p).unwrap().write_all(&data).unwrap();
        let rep = inspect_metadata(&p).unwrap();
        let _ = fs::remove_file(&p);
        assert_eq!(rep.architecture.as_deref(), Some("qwen3next"));
        assert!(rep.tokenizer_tokens_present);
        assert!(!rep.tokenizer_merges_present);
        assert_eq!(rep.tensor_count_scanned, 2);
        assert_eq!(rep.tensor_types.get("F32"), Some(&1));
        assert_eq!(rep.expert_tensor_types.get("Q4_0"), Some(&1));
        assert_eq!(rep.payload_bytes_required, 50);
    }

    #[test]
    fn rejects_overlapping_tensors() {
        let p = temp_path("overlap");
        let data = build_gguf(&[("a", &[32], 0, 0), ("b", &[32], 0, 32)], None, 256);
        File::create(&p).unwrap().write_all(&data).unwrap();
        let err = inspect_metadata(&p).unwrap_err();
        let _ = fs::remove_file(&p);
        assert_eq!(err.message, "GGUF tensor byte ranges overlap");
    }

    #[test]
    fn rejects_single_tensor_trailing_bytes() {
        let p = temp_path("trailing");
        let data = build_gguf(&[("a", &[4], 0, 0)], None, 16 + 64);
        File::create(&p).unwrap().write_all(&data).unwrap();
        let err = inspect_metadata(&p).unwrap_err();
        let _ = fs::remove_file(&p);
        assert_eq!(
            err.message,
            "single-tensor GGUF shard has unexplained trailing bytes"
        );
    }
}
