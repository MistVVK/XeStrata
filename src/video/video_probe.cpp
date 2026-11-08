// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/video/video_probe.cpp - the video input and output (video_io.hpp) by hand:
//
//   strata-video-probe info FILE                      what probe() reads
//   strata-video-probe copy FILE STEM [CHUNK] [CODEC] read in chunks of CHUNK frames and write back (h264, vp9, ffv1),
//                                                     the audio copied; the frames and the seconds
//   strata-video-probe frames FILE FIRST N OUT.rgb    N frames from FIRST as raw RGB24

#include "strata/video/video_io.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace sv = strata::video;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: xestrata-video-probe info FILE | copy FILE STEM [CHUNK] [CODEC] | "
                             "frames FILE FIRST N OUT.rgb\n");
        return 2;
    }
    const std::string mode = argv[1], file = argv[2];
    try {
        const sv::Info info = sv::probe(file);
        if (mode == "info") {
            std::printf("size=%lldx%lld fps=%.6f vfr=%d frames=%lld exact=%d duration=%.3f rotation=%d bits=%d "
                        "codec=%s pix_fmt=%s transfer=%s audio=%s h264=%s\n",
                        (long long) info.width, (long long) info.height, info.fps, (int) info.vfr,
                        (long long) info.frames, (int) info.frames_exact, info.duration, info.rotation, info.bits,
                        info.codec.c_str(), info.pix_fmt.c_str(), info.transfer.c_str(),
                        info.audio ? info.audio_codec.c_str() : "none", sv::h264_encoder().c_str());
            return 0;
        }
        if (mode == "copy" && argc >= 4) {
            const int64_t chunk = argc > 4 ? std::strtoll(argv[4], nullptr, 10) : 16;
            sv::WriteOptions o;
            const std::string c = argc > 5 ? argv[5] : "";
            o.codec = c == "h264" ? sv::Codec::h264 : c == "vp9" ? sv::Codec::vp9 : c == "ffv1" ? sv::Codec::ffv1
                                                                                                 : sv::Codec::automatic;
            o.audio_from = file;
            const auto t0 = std::chrono::steady_clock::now();
            sv::Reader r(file, info);
            sv::Writer w(argv[3], info.width, info.height, info.fps, o);
            std::vector<uint8_t> buf((size_t) (r.frame_bytes() * chunk));
            int64_t total = 0;
            for (int64_t n; (n = r.read(buf.data(), chunk)) > 0;) {
                w.write(buf.data(), n);
                total += n;
            }
            w.finish();
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("frames=%lld written=%s mime=%s seconds=%.3f\n", (long long) total, w.path().c_str(),
                        w.mime().c_str(), s);
            return 0;
        }
        if (mode == "frames" && argc >= 6) {
            const int64_t first = std::strtoll(argv[3], nullptr, 10), n = std::strtoll(argv[4], nullptr, 10);
            sv::Reader r(file, info, first);
            std::vector<uint8_t> buf((size_t) (r.frame_bytes() * n));
            const int64_t got = r.read(buf.data(), n);
            std::ofstream(argv[5], std::ios::binary).write((const char*) buf.data(), r.frame_bytes() * got);
            std::printf("frames=%lld\n", (long long) got);
            return 0;
        }
        std::fprintf(stderr, "xestrata-video-probe: unknown mode %s\n", mode.c_str());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-video-probe: %s\n", e.what());
        return 1;
    }
}
