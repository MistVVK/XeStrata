// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/video/video_io.hpp - a video's frames in and out through the ffmpeg and ffprobe commands (found on the
// PATH, or named by XESTRATA_FFMPEG / XESTRATA_FFPROBE; not part of the engine: setup.py's check_ffmpeg tells the user how
// to get them), streamed a few frames at a time as RGB24, so that no video is held whole in RAM.
//
// probe() asks ffprobe what the file is: the first video stream's size as shown (a rotation in its side data or tags
// turns it, as ffmpeg's decoder applies it), its frame rate, whether that varies (r_frame_rate against avg_frame_rate:
// such a video is read at its average rate, frames dropped or repeated to it, and the result says so), the frames'
// count (the stream's, else its duration times the rate), the bits a sample and the colors' transfer.  HDR (PQ or HLG)
// is refused with the reason: mapping it to SDR changes how it looks, which is the user's call, not the reader's.
// 10-bit SDR is read as 8 bits.
//
// Writer encodes RGB24 frames: H.264 (libx264, else libopenh264) in MP4, else VP9 in WebM, whichever the ffmpeg there
// has; FFV1 in Matroska where a lossless file is asked for.  A mask video is gray (one byte a pixel).  The audio of a
// source file is copied in, or made AAC (Opus in WebM) where the container has no tag for its codec.  H.264 and VP9
// take even sides: an odd side gets its last row or column repeated.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace strata::video {

/// A file that is not a video ffmpeg reads, an HDR video, a missing ffmpeg: the message the request's answer gives.
struct VideoError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Info {
    int64_t width = 0, height = 0;   // as shown (the rotation applied)
    double fps = 0.0;                // the rate the frames are read at (the average one)
    double duration = 0.0;           // seconds
    int64_t frames = 0;              // the frames read at `fps` (estimated where the stream does not say)
    bool frames_exact = false;       // the stream gave its frame count
    bool vfr = false;                // the rate varies: read at its average (frames dropped or repeated)
    int rotation = 0;                // degrees, as the file says
    int bits = 8;                    // a sample's bits (10-bit SDR is read as 8)
    std::string codec, pix_fmt, transfer;
    bool audio = false;              // it has an audio stream
    std::string audio_codec;
};

/// The ffmpeg and ffprobe commands (XESTRATA_FFMPEG, XESTRATA_FFPROBE or their old STRATA_ names, else on the PATH);
/// empty where there is none.
std::string ffmpeg_command();
std::string ffprobe_command();
/// Whether both are there.
bool available();

/// What `path` is; VideoError for a file ffprobe cannot read, one without a video stream, or HDR.
Info probe(const std::string& path);

/// The frames of `path` from `first`, as RGB24 [h][w][3] at info.fps.
class Reader {
public:
    Reader(const std::string& path, const Info& info, int64_t first = 0);
    ~Reader();
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
    /// Up to `n` frames into `rgb` (n * w * h * 3 bytes); the frames read, fewer at the end, 0 after it.  VideoError
    /// where ffmpeg failed (its message).
    int64_t read(uint8_t* rgb, int64_t n);
    int64_t frame_bytes() const { return info_.width * info_.height * 3; }

private:
    struct Impl;
    Info info_;
    std::unique_ptr<Impl> impl_;
};

enum class Codec { automatic, h264, vp9, ffv1 };

struct WriteOptions {
    Codec codec = Codec::automatic;   // automatic: H.264 where there is an encoder for it, else VP9
    bool gray = false;                // frames of one byte a pixel (a mask)
    int quality = 18;                 // H.264's CRF, VP9's (its scale 0 .. 63, the same number)
    std::string audio_from;           // a file whose audio goes in (empty: none)
};

class Writer {
public:
    /// Starts writing w x h frames at `fps` to `stem` + the container's extension (".mp4", ".webm", ".mkv"):
    /// path() says which.
    Writer(const std::string& stem, int64_t w, int64_t h, double fps, const WriteOptions& o);
    ~Writer();   // a writer not finished is ended and its file removed
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;
    /// `n` frames of RGB24 (or gray) [h][w].  VideoError where ffmpeg failed.
    void write(const uint8_t* frames, int64_t n);
    /// Ends the file; VideoError where ffmpeg failed.
    void finish();
    const std::string& path() const { return path_; }
    /// The file's media type ("video/mp4", "video/webm", "video/x-matroska").
    std::string mime() const;
    Codec codec() const { return codec_; }

private:
    struct Impl;
    std::string path_;
    Codec codec_ = Codec::h264;
    std::unique_ptr<Impl> impl_;
};

/// The H.264 encoder the ffmpeg there has ("libx264", "libopenh264"), empty where it has none.
std::string h264_encoder();

}  // namespace strata::video
