// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/video/video_io.cpp - see include/strata/video/video_io.hpp.

#include "strata/video/video_io.hpp"

#include "strata/platform/env.hpp"
#include "strata/platform/process.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

namespace strata::video {

namespace {

// the command XESTRATA_<name> (else STRATA_<name>) names, else `program` where it is on the PATH
std::string env_or(const char* name, const char* program) {
    const char* e = strata::env_get(name);
    if (e != nullptr && *e != '\0') return e;
    return platform::Process::found(program) ? program : std::string();
}

// key=value lines (ffprobe's default writer without its section wrappers) into a map, the first of a key kept
std::map<std::string, std::string> keys(const std::string& text) {
    std::map<std::string, std::string> m;
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        m.emplace(line.substr(0, eq), line.substr(eq + 1));
    }
    return m;
}

// a number ffprobe wrote; 0 for nothing or what is not one
double num(const std::string& s) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    return end != s.c_str() ? v : 0.0;
}
int64_t inum(const std::string& s) { return (int64_t) std::strtoll(s.c_str(), nullptr, 10); }

// "30000/1001" or "29.97" -> 29.97; 0 for "0/0" or nothing
double rate(const std::string& r) {
    const size_t slash = r.find('/');
    if (slash == std::string::npos) return num(r);
    const double a = num(r.substr(0, slash)), b = num(r.substr(slash + 1));
    return b > 0.0 ? a / b : 0.0;
}

// the rate to the precision ffmpeg's fps filter and -r take
std::string rate_arg(double fps) {
    char b[64];
    std::snprintf(b, sizeof b, "%.6f", fps);
    return b;
}

std::string run_text(const std::vector<std::string>& argv, int& code, std::string& err) {
    platform::Process p(argv, false, true);
    std::string out = p.read_rest();
    code = p.wait();
    err = p.error_tail();
    return out;
}

std::string last_lines(const std::string& s, int lines = 4) {
    size_t at = s.size();
    while (at > 0 && (s[at - 1] == '\n' || s[at - 1] == '\r')) --at;
    size_t start = at;
    for (int n = 0; start > 0 && n < lines; --start)
        if (s[start - 1] == '\n') ++n;
    return s.substr(start, at - start);
}

}  // namespace

std::string ffmpeg_command() { return env_or("FFMPEG", "ffmpeg"); }
std::string ffprobe_command() { return env_or("FFPROBE", "ffprobe"); }
bool available() { return !ffmpeg_command().empty() && !ffprobe_command().empty(); }

Info probe(const std::string& path) {
    const std::string fp = ffprobe_command();
    if (fp.empty() || ffmpeg_command().empty())
        throw VideoError("video needs ffmpeg and ffprobe, which were not found (xestrata --setup says how to get "
                         "them)");
    int code = 0;
    std::string err;
    const std::string entries = "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,"
                                "duration,color_transfer,bits_per_raw_sample:stream_tags=rotate:"
                                "stream_side_data=rotation:format=duration";
    const std::string v = run_text({fp, "-v", "error", "-select_streams", "v:0", "-show_entries", entries, "-of",
                                    "default=nw=1", path},
                                   code, err);
    if (code != 0) throw VideoError("not a video ffmpeg reads: " + last_lines(err, 2));
    const auto k = keys(v);
    const auto get = [&](const char* key) {
        const auto it = k.find(key);
        return it == k.end() || it->second == "N/A" ? std::string() : it->second;
    };
    Info info;
    info.width = inum(get("width"));
    info.height = inum(get("height"));
    if (info.width <= 0 || info.height <= 0) throw VideoError("no video stream in the file");
    info.codec = get("codec_name");
    info.pix_fmt = get("pix_fmt");
    info.transfer = get("color_transfer");
    if (info.transfer == "smpte2084" || info.transfer == "arib-std-b67")
        throw VideoError("an HDR video (" + info.transfer + "): mapping it to SDR changes how it looks, and that is "
                         "not done here; convert it to SDR first");
    const std::string bits = get("bits_per_raw_sample");
    info.bits = bits.empty() ? (info.pix_fmt.find("10") != std::string::npos ? 10 : 8) : (int) inum(bits);
    const std::string rot = !get("rotation").empty() ? get("rotation") : get("TAG:rotate");
    info.rotation = (int) std::lround(num(rot));
    if (((info.rotation % 180) + 180) % 180 == 90) std::swap(info.width, info.height);
    const double r = rate(get("r_frame_rate")), avg = rate(get("avg_frame_rate"));
    info.fps = avg > 0.0 ? avg : r;
    if (info.fps <= 0.0) throw VideoError("the video's frame rate is unknown");
    info.vfr = r > 0.0 && avg > 0.0 && std::abs(r - avg) > 1e-3 * avg;
    info.duration = num(get("duration"));
    if (info.duration <= 0.0) {   // a stream without its own duration: the container's
        int c2 = 0;
        std::string e2;
        const auto f = keys(run_text({fp, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1",
                                      path},
                                     c2, e2));
        const auto d = f.find("duration");
        if (d != f.end()) info.duration = num(d->second);
    }
    const std::string nb = get("nb_frames");
    if (!nb.empty() && !info.vfr) {
        info.frames = inum(nb);
        info.frames_exact = info.frames > 0;
    }
    if (info.frames <= 0) info.frames = std::max<int64_t>(1, std::llround(info.duration * info.fps));
    const auto a = keys(run_text({fp, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name",
                                  "-of", "default=nw=1", path},
                                 code, err));
    const auto ac = a.find("codec_name");
    info.audio = ac != a.end() && !ac->second.empty();
    if (info.audio) info.audio_codec = ac->second;
    return info;
}

struct Reader::Impl {
    std::unique_ptr<platform::Process> p;
    bool done = false;
};

Reader::Reader(const std::string& path, const Info& info, int64_t first)
    : info_(info), impl_(std::make_unique<Impl>()) {
    const std::string ff = ffmpeg_command();
    if (ff.empty()) throw VideoError("video needs ffmpeg, which was not found (xestrata --setup says how to get it)");
    std::vector<std::string> a = {ff, "-v", "error", "-nostdin"};
    if (first > 0) {   // a frame-exact start: seek before it, then drop the frames up to it by number
        a.insert(a.end(), {"-ss", rate_arg(std::max(0.0, (double) first / info.fps - 1.0))});
    }
    a.insert(a.end(), {"-i", path, "-map", "0:v:0"});
    // a VFR video's frames to its average rate (dropped or repeated); a constant rate's as they are
    std::string vf = info.vfr ? "fps=" + rate_arg(info.fps) : std::string();
    if (first > 0) {
        const double start = std::max(0.0, (double) first / info.fps - 1.0);
        const int64_t skip = first - std::llround(start * info.fps);
        if (skip > 0) vf += (vf.empty() ? "" : ",") + std::string("select=gte(n\\,") + std::to_string(skip) + ")";
    }
    if (!vf.empty()) a.insert(a.end(), {"-vf", vf});
    a.insert(a.end(), {"-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"});
    impl_->p = std::make_unique<platform::Process>(a, false, true);
}

Reader::~Reader() = default;

int64_t Reader::read(uint8_t* rgb, int64_t n) {
    if (impl_->done || n <= 0) return 0;
    const size_t fb = (size_t) frame_bytes();
    const size_t got = impl_->p->read_all(rgb, fb * (size_t) n);
    const int64_t frames = (int64_t) (got / fb);
    if (got < fb * (size_t) n) {   // the end: ffmpeg's exit says whether it was the file's end or a failure
        impl_->done = true;
        if (impl_->p->wait() != 0) throw VideoError("reading the video failed: " + last_lines(impl_->p->error_tail()));
    }
    return frames;
}

std::string h264_encoder() {
    static std::once_flag once;
    static std::string enc;
    std::call_once(once, [] {
        const std::string ff = ffmpeg_command();
        if (ff.empty()) return;
        int code = 0;
        std::string err;
        const std::string list = run_text({ff, "-hide_banner", "-encoders"}, code, err);
        for (const char* e : {"libx264", "libopenh264"}) {
            const std::string word = std::string(" ") + e + " ";
            if (list.find(word) != std::string::npos) {
                enc = e;
                return;
            }
        }
    });
    return enc;
}

namespace {
bool has_encoder(const std::string& name) {
    const std::string ff = ffmpeg_command();
    if (ff.empty()) return false;
    int code = 0;
    std::string err;
    return run_text({ff, "-hide_banner", "-encoders"}, code, err).find(" " + name + " ") != std::string::npos;
}

// an audio codec an MP4 file has a tag for (else it is made AAC)
bool mp4_audio(const std::string& c) {
    for (const char* ok : {"aac", "mp3", "alac", "ac3", "eac3", "opus", "flac"})
        if (c == ok) return true;
    return false;
}
}  // namespace

struct Writer::Impl {
    std::unique_ptr<platform::Process> p;
    int64_t frame_bytes = 0;
    bool finished = false;
};

Writer::Writer(const std::string& stem, int64_t w, int64_t h, double fps, const WriteOptions& o)
    : impl_(std::make_unique<Impl>()) {
    const std::string ff = ffmpeg_command();
    if (ff.empty()) throw VideoError("video needs ffmpeg, which was not found (xestrata --setup says how to get it)");
    if (w <= 0 || h <= 0 || fps <= 0.0) throw std::invalid_argument("Writer: no frames to write");
    const std::string h264 = h264_encoder();
    codec_ = o.codec;
    if (codec_ == Codec::automatic) codec_ = !h264.empty() ? Codec::h264 : Codec::vp9;
    if (codec_ == Codec::h264 && h264.empty()) throw VideoError("this ffmpeg has no H.264 encoder");
    if (codec_ == Codec::vp9 && !has_encoder("libvpx-vp9")) throw VideoError("this ffmpeg has no VP9 encoder");
    path_ = stem + (codec_ == Codec::h264 ? ".mp4" : codec_ == Codec::vp9 ? ".webm" : ".mkv");
    impl_->frame_bytes = w * h * (o.gray ? 1 : 3);
    const std::string size = std::to_string(w) + "x" + std::to_string(h);
    std::vector<std::string> a = {ff, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt",
                                  o.gray ? "gray" : "rgb24", "-s", size, "-r", rate_arg(fps), "-i", "-"};
    std::string audio;
    if (!o.audio_from.empty()) {
        Info src;
        try {
            src = probe(o.audio_from);
        } catch (const VideoError& e) {   // a source that no longer reads: the video without its audio
            std::fprintf(stderr, "xestrata-image: the audio of %s is left out: %s\n", o.audio_from.c_str(), e.what());
        }
        if (src.audio) {
            a.insert(a.end(), {"-i", o.audio_from, "-map", "0:v:0", "-map", "1:a:0"});
            audio = src.audio_codec;
        }
    }
    const std::string q = std::to_string(o.quality);
    // 4:2:0 wants even sides: an odd one gets a column or row repeated
    const std::string even = (w % 2 || h % 2) && codec_ != Codec::ffv1 ? "pad=ceil(iw/2)*2:ceil(ih/2)*2" : "";
    if (!even.empty()) a.insert(a.end(), {"-vf", even});
    switch (codec_) {
    case Codec::h264:
        a.insert(a.end(), {"-c:v", h264, "-pix_fmt", o.gray && h264 == "libx264" ? "gray" : "yuv420p"});
        if (h264 == "libx264") a.insert(a.end(), {"-crf", q, "-preset", "medium"});
        else a.insert(a.end(), {"-b:v", "0", "-qmax", q});
        a.insert(a.end(), {"-movflags", "+faststart"});
        break;
    case Codec::vp9:
        a.insert(a.end(), {"-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-crf", q, "-b:v", "0", "-row-mt", "1"});
        break;
    default:
        a.insert(a.end(), {"-c:v", "ffv1", "-pix_fmt", o.gray ? "gray" : "rgb24", "-level", "3"});
        break;
    }
    if (!audio.empty()) {
        const bool copy = codec_ == Codec::h264 ? mp4_audio(audio)
                          : codec_ == Codec::vp9 ? (audio == "opus" || audio == "vorbis")
                                                 : true;
        if (copy) a.insert(a.end(), {"-c:a", "copy"});
        else if (codec_ == Codec::vp9) a.insert(a.end(), {"-c:a", "libopus", "-b:a", "160k"});
        else a.insert(a.end(), {"-c:a", "aac", "-b:a", "192k"});
    }
    a.push_back(path_);
    impl_->p = std::make_unique<platform::Process>(a, true, false);
}

Writer::~Writer() {
    if (!impl_->finished) {
        impl_->p->kill();
        impl_->p->wait();
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
}

void Writer::write(const uint8_t* frames, int64_t n) {
    if (n <= 0) return;
    if (!impl_->p->write(frames, (size_t) (impl_->frame_bytes * n))) {
        impl_->p->wait();
        throw VideoError("writing the video failed: " + last_lines(impl_->p->error_tail()));
    }
}

void Writer::finish() {
    impl_->p->close_in();
    if (impl_->p->wait() != 0) throw VideoError("writing the video failed: " + last_lines(impl_->p->error_tail()));
    impl_->finished = true;
}

std::string Writer::mime() const {
    return codec_ == Codec::h264 ? "video/mp4" : codec_ == Codec::vp9 ? "video/webm" : "video/x-matroska";
}

}  // namespace strata::video
