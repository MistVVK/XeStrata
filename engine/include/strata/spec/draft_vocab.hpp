// include/strata/spec/draft_vocab.hpp - which draft-vocabulary file a DFlash2-alone start reads its draft head from.
//
// The subset is `--mtp-draft-vocab FILE`, else `<--mtp DIR>/draft_vocab.bin`.  A start without it would run the drafter's head over the whole vocabulary without a word
// (+0.7 ms a round measured on Metal), so a missing file is an error at start.  One exception: a build that ships its own default subset and can name it (the Metal
// engine: data/draft_vocab.bin beside the build, the file Window::df2_bind has always fallen back to; the Mac's MTP folder is empty by design) passes that path
// as `packaged_default`; it is used only when the person did not name a file, and only when it exists.  CUDA, HIP and SYCL pass "": a missing file stays an error.
// `whole_head_ok` is the person's own STRATA_METAL_DRAFT_VOCAB=full (the whole head, asked for on purpose).  Header only, no device types.
#pragma once

#include <functional>
#include <string>

namespace strata::spec {

struct Df2Vocab {
    bool ok = false;
    std::string path;        ///< the file the drafter binds (the wanted path, even when it is missing and the whole head was asked for)
    bool packaged = false;   ///< `path` is the build's packaged default, not <mtp>/draft_vocab.bin
    bool whole_head = false; ///< no file, on purpose: the drafter runs its head over the whole vocabulary
    std::string why;         ///< ok == false: the line for stderr
};

inline Df2Vocab df2_vocab_resolve(const std::string& mtp_dir, const std::string& explicit_file, const std::string& packaged_default, bool whole_head_ok,
                                  const std::function<bool(const std::string&)>& exists) {
    Df2Vocab r;
    const std::string wanted = explicit_file.empty() ? mtp_dir + "/draft_vocab.bin" : explicit_file;
    if (exists(wanted)) { r.ok = true; r.path = wanted; return r; }
    if (explicit_file.empty() && !packaged_default.empty() && exists(packaged_default)) { r.ok = true; r.path = packaged_default; r.packaged = true; return r; }
    if (whole_head_ok) { r.ok = true; r.path = wanted; r.whole_head = true; return r; }
    r.path = wanted;
    r.why = "--draft dflash2: the draft vocabulary " + wanted + " does not exist (--mtp DIR needs DIR/draft_vocab.bin, or --mtp-draft-vocab FILE); the drafter would run its head over the whole vocabulary";
    return r;
}

}  // namespace strata::spec
