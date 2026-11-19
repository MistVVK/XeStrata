// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/strata_image.cpp - strata-image: Qwen-Image-2.1, Mage-Flow, HunyuanImage 3.0 or MOSS-Image text to image on
// the GPU.
//
//   strata-image --dit DIT.gguf --vae VAE.safetensors --text TEXT_ENCODER.gguf [--vision MMPROJ.gguf]
//                [--budget-mib M] (--ids 1,2,3 [--drop N] [--width W] [--height H] [--seed S] [--sigmas 1,...,0]
//                [--ref IMAGE]... [--mask IMAGE] [--init IMAGE] [--strength F] [--repeat N] [--lora STRENGTH:FILE]... --out IMAGE.rgba | --serve)
//   strata-image --dit DIT.safetensors ... [--dit-q8 auto|0|1] [--edit-dit EDIT_DIT.safetensors --vision MMPROJ.gguf]
//                [--dit-keep 0|1]
//
// The DiT's file says the model: a GGUF is Qwen-Image-2.1's, a safetensors file Mage-Flow's (its bf16 weights as
// Comfy-Org has them; its blocks FP16 on the GPU when they fit the budget with the text encoder's layers, else
// quantized to 8 bits as they are read (MfDit: W8A8 or Q8_0): --dit-q8 auto, the default; 1 always 8 bits, 0 always
// FP16).  A safetensors file whose header is a HunyuanImage 3.0 save - its `format`, or the `comfy_quant` beside
// its matrices, or the model's own `model.wte.weight` beside `patch_embed` - is HunyuanImage 3.0's, and takes no
// --text: its DiT reads the ids itself (`wte`, with three rows its embedders write over), so its entry is
//   strata-image --dit QUANTIZED.safetensors --vae VAE.safetensors [--vision VISION.safetensors] (--ids ... |
//                --serve) [--out F] [--width W] [--height H] [--seed S] [--steps N] [--sigmas 1,...,0]
//                [--guidance G] [--ref IMAGE]... [--budget-mib M]
// and its guidance is a token's value (`--guidance`, the model's own 2.5 unset) rather than a second pass.  Its
// edits (--ref, REF:) need --vision, the model's bf16 shard with its `vision_model.*` tensors (SigLIP2): the ids
// then carry each reference's block before the prompt (hunyuanimage3_pipeline.hpp).  Mage-Flow's edits take
// its edit DiT (--edit-dit) and Qwen3-VL-4B's mmproj (--vision), up to three --ref; --dit-keep 1, the default, keeps the
// DiT not in use in RAM between a generation and an edit (0: read from its file again).
// MOSS-Image's DiT is its diffusers transformer folder (or a safetensors file of its weights) with the Flux2 VAE
// (--vae, its narrower decoder) and Qwen3.5-9B's GGUF (--text): its prompt in its own chat form, all of it (--drop 0),
// guidance with --neg-ids (the empty negative prompt's ids, the model's own) and --cfg (its 3), --steps (50) and
// --shift (5) its schedule's, --uncond-skip the blocks the negative pass skips (22; none), skip-layer guidance with
// --slg-layers and --slg-scale, both active from --slg-start to --slg-stop (shares of the steps, 0 and 1), and --te /
// --neg-te the chat engine's states (its ENCODE files) in place of encoding the ids.
// Anima's DiT is a safetensors file with its LLM adapter (anima_dit.hpp), with Wan 2.1's VAE (--vae, Comfy-Org's
// qwen_image_vae.safetensors) and Qwen3-0.6B-Base's GGUF (--text): its prompt's Qwen3 ids as --ids (no chat form),
// the T5 tokenizer's ids and their weights (--t5, --t5-weights: the LLM adapter's queries, the prompt's emphasis),
// guidance with --neg-ids, --neg-t5, --neg-t5-weights and --cfg, --steps and --shift (30, 3) its schedule's, and
// --sampler er_sde (the default) or euler.
//
// One image: --ids, the prompt in the model's chat form, tokenized (the server's tokenizer); --drop, the leading ids of
// the system turn; --budget-mib, the VRAM the engine may take (default: the free VRAM); --repeat, the image again N
// times in the process (the cache's effect); --sigmas, the sampler's sigmas from 1 down to the terminal 0, one step
// between each two (default Turbo's 8 steps; serve/images.py's base_sigmas for the base model); --ref, an edit's
// reference image (PNG, JPEG, BMP, GIF; its pad in --ids is one <|image_pad|>, which the engine widens to the image's
// tokens), with the vision encoder (--vision, Qwen3-VL's mmproj); --init, img2img's base image for a model without
// edits; --strength, how much of the schedule runs from the base image (diffusers' strength, above 0 and at most 1;
// 1, the default, starts from noise); --mask, a mask over the base image (the --init image, else the first --ref;
// OpenAI's: a PNG whose transparent pixels are where the image changes, the rest kept as the base has it); see
// img_start.hpp.
// Adapters (LoRA files, image_lora.hpp: the forms it reads, the plain LoRA's now) are applied at run time,
// a strength each (and a factor of it each DiT block, where the word gives them): the IMG line's
// LORA:<strength>[@<block 0>,<block 1>,...]:<path>, repeated, in the order given.  Their A and B stay on the GPU
// with the DiT's weights (the budget's fixed part; the IMAGE line says how many, lora_n, and their MiB, lora_mib), and
// a set that differs from the last request's starts the context again (the prefix's K and V were the other set's).  A
// model that applies none refuses a request that names one: HunyuanImage 3.0 (it takes no adapters).
// Writes RGBA8 rows to --out and a line of times and of the budget's split a run.
//
// --serve: the engine stays, its layers and the last prompt's states kept between requests, and takes lines on stdin
// (serve/images.py), one at a time, answering each on stdout:
//   IMG <width> <height> <seed> <drop> <out path> <ids, commas> [SIGMAS:<sigmas, commas>] [REF:<path>]...
//                            -> STAGE <refs|text|sampling|decode> as each stage starts and STEP <done> <steps> lines,
//                            then IMAGE key=value ... (times, the budget's split); SIGMAS: as --sigmas; REF: an
//                            edit's reference images; MASK:, INIT: and STRENGTH: as --mask, --init and --strength;
//                            MOSS-Image's edits: EDITFMT:aitk an ai-toolkit adapter's layout, EDITFMT:aitk-nokv one
//                            trained with its kv_cache off, EDITFMT:index the references at the image's timestep
//                            (EDITFMT:moss, the default, Moss-Image-Edit's; QiRefLayout), PIC:<path> a picture the text
//                            encoder sees (at the ids' next <|image_pad|>; its sides multiples of 32; --vision,
//                            Qwen3.5-9B's mmproj);
//                            CROP:<x>,<y>,<w>,<h>,<blend>,<width>,<height> the cut (img_crop.hpp): the mask's
//                            box of the base image drawn at <width> <height> and put back, the image at the base
//                            image's own size (its partial images too), the IMAGE line's crop_s the steps' time;
//                            ,<ref_w>,<ref_h> after it: the first reference the box at that size beside the base
//                            (an ai-toolkit MOSS-Image edit's; the base the drawing's init image); COLOR:<off|auto|
//                            full> the drawing's colors matched to the picture's as it goes back (off, the default;
//                            auto by the part kept around the paint; full by the whole cut: CropStitch::stitch);
//                            UPSCALE: the base image (INIT:, else the first REF:) Lanczos-resized to the line's size
//                            first (an upscale's second pass; its time in crop_s); UPS:<name>[:<path>] with it: that
//                            upscaler's (UPS's) first, then the drawing (its time in ups_s), UPSCOLOR:<m> its color
//                            correction (UPS's COLOR:); UPS: with CROP: the box enlarged to the drawing's size by that
//                            upscaler instead of Lanczos (crop_upscale; its time in ups_s); the upscaler's run says
//                            STAGE upscale, then PART <share done, 0..1> as it goes; TILES:'s drawing says TILE <i>
//                            <n> before each tile's STEP lines;
//                            LORA:<strength>[@<w0>,<w1>,...]:<path> an adapter (repeated, in the order given; the
//                            strength first, so that a path's own colons, as Windows' have, stand; the path's spaces
//                            and other separators %XX'd, the word's whitespace being the line's); @ the strength's
//                            factor each DiT block (one a block, LoraUse's blocks);
//                            NEG:<ids, commas> and CFG:<scale> (above 1):
//                            classifier-free guidance with that negative prompt (its pads as the prompt's); GUID:
//                            <scale> is HunyuanImage 3.0's guidance token instead (it samples once: no NEG:, no CFG:);
//                            PREVIEW:<steps, commas>: partial images after those steps, each a line PREVIEW <index>
//                            <step> <path> before the next STEP, its RGBA8 rows at <out path>.p<index>
//                            MOSS-Image's SHIFT:, USKIP:<blocks|none>, SLG:<blocks>, SLGSCALE:, SLGSTART:, SLGSTOP:
//                            as the options, and TE:<path>, NEGTE:<path> the chat engine's states
//                            Anima's T5:<ids>, T5W:<weights>, NEGT5:<ids>, NEGT5W:<weights> (commas) and
//                            SAMPLER:<euler|er_sde> (Mage-Flow's too)
//                            Qwen-Image-2.1's and Mage-Flow's guidance (guidance.hpp): CFGNORM:<strength>, CFGZERO,
//                            APG:<eta>,<norm_threshold>,<momentum>, DSLG:<blocks> with DSLGP:<scale>,<sigma_start>,
//                            <sigma_end>,<rescaling_scale> (skip layer guidance, its window in the model's sigmas),
//                            DD:<amount>,<start>,<end>,<bias>,<exponent>,<start_offset>,<end_offset>,<fade>,<smooth>,
//                            <cfg_scale_override> (Detail Daemon)
//                            Qwen-Image-2.1's ControlNet (qi_control.hpp): CNET:<path> the branch's file, CTRL:<path>
//                            the control image (a map, resized to the line's size), CTRLS:<strength>,<from>,<to> (1, 1,
//                            0: the sigmas it is on from and to), CTRLNOINPAINT the mask kept from the branch
//                            MOSS-Image's color and depth control (moss_dit.hpp's set_control): MCTL:<path> the
//                            adapter's file (its LoRA as a LORA: word beside it), MCOLOR:<path> the color map,
//                            MDEPTH:<path> the depth map (near white), MCTLENDS:<color end>,<depth end> (1, 1: the
//                            shares of the steps each is on for)
//   NEED <width> <height> <n> [REF:<path>]... [MASK:<path>] [INIT:<path>] [NEG:<n>] [LORA:<strength>:<path>]...
//                            MCTL:<path> and MCOLOR:1 / MDEPTH:1 count MOSS-Image control maps and their encoder
//                            the VRAM an image of n prompt ids needs (from the files); MASK: or INIT:, a base image
//                            (with a strength below 1 and references, MASK:<their first>); NEG: guidance's negative
//                            prompt of n ids; LORA: the request's adapters (their A and B are fixed); CROP: as IMG's
//                            (the first reference counted at the line's size, the steps' buffers added); PIC: the text
//                            encoder's pictures (the vision encoder counted; n counts their tokens); TE: the states
//                            come from the chat's engine (MOSS-Image: no text encoder counted); CNET: the ControlNet
//                            branch's (its weights, blocks and buffers counted)
//                                                                                          -> NEED least_mib=...
//                            full_mib=... held_mib=... free_mib=... cap_mib=... host_mib=... (si::QiNeed)
//                            UPS:<name>[:<path>] [UPSCOLOR:<m>] UPSIN:<width>,<height>[,<out width>,<out height>]: an
//                            upscale's first pass by that upscaler from a base image of that size to the line's size
//                            (or the out size: an upscale in tiles), before the drawing (the larger of the two counted)
//   UPS <width> <height> <in> <out> <name>[:<path>] [COLOR:<none|lab|adain|wavelet>] [SEED:<s>] [TILE:<n>]
//                            the picture <in> upscaled to width x height by the upscaler <name> (strata/upscale/
//                            upscale.hpp: lanczos, or realesrgan with its weights' <path>, %XX'd as LORA:'s), no image
//                            model run, its RGBA8 rows to <out>; COLOR: the color correction (the upscaler's own
//                            when none is named); TILE: the tiles' largest core      -> STAGE upscale, PART <share
//                            done, 0..1> as it goes, then UPSCALED
//                            width=... height=... upscale_s=... model_s=... resize_s=... color_s=... color=...
//                            tile=... tiles=...
//   UPSNEED <width> <height> <in width> <in height> <name>[:<path>] [COLOR:<m>]
//                            the VRAM that upscale needs (from the files)                   -> NEED as NEED's
//   UPSVNEED <width> <height> <in width> <in height> <name>[:<path>] [COLOR:<m>]
//                            the VRAM a video's upscale needs: UPSNEED's, a temporal model's chunk of one latent
//                            in the VAE's tiles that take the least of it with their carried frames in the RAM free
//                            now (least: past any GPU's where none fits the RAM) or of 21 frames with the frame
//                            whole (full, the default chunk) added                         -> NEED as NEED's,
//                            with ram_short=1 where the RAM is too short for every tiling
//   UPSV <width> <height> <in> <out stem> <name>[:<path>] [COLOR:<m>] [SEED:<s>] [CHUNK:<frames>] [OVERLAP:<latents>]
//        [CODEC:<h264|vp9|ffv1>] [YIELD:1]
//                            the video <in> (a file ffmpeg reads) upscaled to width x height by the upscaler, written
//                            to <out stem> + its container's extension with the source's audio (upscale.hpp's
//                            run_video: SeedVR2 in time, in chunks of CHUNK frames overlapping by OVERLAP latents, 0
//                            its own choice; the others frame by frame); the image model's room let go first.
//                            YIELD:1: YIELD at each chunk's end (a few frames' of the others), on at RESUME
//                            -> STAGE upscale-video, STEP <frames written> <frames>, then UPSVIDEO path=... mime=...
//                            frames=... width=... height=... fps=... vfr=... upscale_s=... read_s=... model_s=...
//                            resize_s=... color_s=... write_s=... color=...
// The image tools' auxiliary models (aux_net.hpp: their weights image_aux's files), on a picture (PNG, JPEG...):
//   SEGMENT <weights> <picture> <ids> <score threshold> <mask threshold> <masks>
//                            SAM 3: the instances of the words (CLIP's ids), highest score first, each mask written to
//                            <masks>.<i> (width x height bytes, 255 inside)  -> SEGMENT n=... segment_s=... boxes=...
//   SEGV <weights> <video> <prompts> <out stem> [MASKS:1] [MASK:1] [OVERLAY:1] [CODEC:<h264|vp9|ffv1>] [YIELD:1]
//                            SAM 3.1's video segmentation (sam3_video.hpp) of the prompts file's words ("W <from frame>
//                            <ids>") and clicks ("C <frame> <object or -1> <x,y,label,...>"), each frame's instances a
//                            JSON line of <out stem>.jsonl (vLLM-Omni's frames: id, score, box, COCO RLE); MASKS:1 a
//                            mask video an object (<out stem>.obj<id>), MASK:1 one of them all, OVERLAY:1 the clip with
//                            them over it (its audio kept).  YIELD:1: YIELD between frames each second, on at RESUME
//                            -> STAGE segment-video, STEP <frames> <frames>, then SEGVIDEO json=... objects=<id>:w<i>|
//                            c<i>,... [masks=<id>:<path>,...] [mask=... mime=...] [overlay=...] frames=... width=...
//                            height=... fps=... vfr=... segment_s=... read_s=... trunk_s=... detect_s=... track_s=...
//                            memory_s=... rules_s=... output_s=... write_s=...
//   FLORENCE <weights> <picture> <ids> <beams> <max new>
//                            Florence-2: the tokens it answers the task's ids (BART's) with  -> FLORENCE ids=... ...
//   POLYMASK <width> <height> <out> <x,y,x,y,...>...
//                            the polygons filled (Florence-2's answer read by the server), 255 inside  -> POLYMASK ok
//   MATTE <weights> <picture> <model width> <model height> <out> [REFINE]
//                            BiRefNet: the alpha (width x height bytes) to <out>; REFINE, the foreground's colors
//                            (RGB8) to <out>.fg too                                    -> MATTE matte_s=... ...
//   AUXNEED <sam3|sam3video|florence|birefnet> <weights> <width> <height> [<buckets> | <beams> <max new> |
//           <model width> <model height> <refine 0|1>]
//   AUXNEED control <weights or -> <width> <height> <kind> <resolution> [<weights2>]
//                            the VRAM that run needs beyond what the engine holds             -> NEED as NEED's
//   COLORMAP <weights> <picture> <out> <ids;ids;...> [<score> <mask>]
//                            MOSS-Image's color map: SAM 3's instances of each words (CLIP's ids, the words' sets
//                            split by ;) on the picture, each painted a flat color of its own (the larger first), black
//                            where none is; RGBA8 rows to <out>  -> COLORMAP width=... height=... objects=... colormap_s=...
//   CONTROL <kind> <picture> <out> [RES:<n>] [LOW:<v>] [HIGH:<v>] [THR:<v>] [SAFE:0|1] [SCORE:<v>] [DIST:<v>]
//           [BODY:0|1] [HANDS:0|1] [FACE:0|1] [W:<weights>] [W2:<weights>]
//                            a control image (cnet_prep.hpp: canny, gray, scribble, scribble_xdog, hed, scribble_hed,
//                            lineart, lineart_coarse, mlsd, depth, pose) of the picture at the shorter side RES (512),
//                            its RGBA8 rows to <out>; LOW, HIGH canny's thresholds (100, 200), THR scribble_xdog's
//                            (12), SAFE HED's steps (1), SCORE and DIST M-LSD's (0.1, 0.1), BODY, HANDS and FACE the
//                            pose's parts (1); W the network's weights (%XX'd as LORA:'s), W2 the pose's DWPose's
//                            beside YOLOX's                  -> CONTROL width=<w> height=<h> control_s=<seconds>
// Each runs in the room the image model's budget leaves (its cache given back first, then its parts, where that is too
// little; ERROR memory: ... when even that is), its scratch freed after it; its weights kept for the next while a
// quarter of the GPU's memory stays free, and let go before the next image job (IMG, NEED, RATIO, UPS, UPSNEED).
//   FREE                     the free VRAM, what the engine holds and the GPU's memory     -> FREE free_mib=... held_mib=...
//                                                                                              total_mib=...
//   W8                       whether this GPU takes the image DiTs' 8-bit form (W8A8: w8_here, timed once a GPU),
//                            for the server to choose a weights file by      -> W8 take=<1, 0, or -1: it could not time>
//   BUDGET <MiB>             the VRAM the engine may take (0: the free VRAM at its start)      -> BUDGET budget_mib=...
//   FIT <MiB>                the budget: what the engine holds and the free VRAM, less <MiB>  -> BUDGET budget_mib=...
//   RELEASE [ALL]            frees the layers, activations and buffers (the cache); ALL also the parts' fixed weights
//                            and scratch, the parts opened again on the next IMG (Mage-Flow's and MOSS-Image's DiTs
//                            keep their blocks quantized in RAM for that)                                         -> RELEASE ok
//   STOP                     (while an IMG runs) ends it after the step it is in               -> ERROR stopped
//   QUIT                     ends the engine
// A request that fails answers ERROR <message> and the engine goes on; ERROR lora: <message> when the fault is an
// adapter it names (LoraRejected: its file, its forms, its rows against the DiT's, a model that takes none).  Before
// the first line it makes a small image (its kernels built and the library's first products set up; skipped when the
// GPU has no room) and lets its buffers go, then says READY budget_mib=<M>.  A model whose sequence has a shape of its
// own (HunyuanImage 3.0, whose ids carry the image block) warms up with one of its own: the pipeline says so in
// warm_up_request.
//
//   strata-image --upscale IMAGE --upscaler NAME [--model PATH] --out IMAGE.rgba [--width W --height H | --scale S]
//                [--color none|lab|adain|wavelet] [--tile N] [--budget-mib M] [--repeat N]
// An upscaler alone (strata/upscale/upscale.hpp), no image model: the image at W x H (or S times its size), a line of
// its times a run.

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/stdin_lines.hpp"
#include "strata/program/id_list.hpp"
#include "strata/image/anima_pipeline.hpp"
#include "strata/image/birefnet.hpp"
#include "strata/image/florence2.hpp"
#include "strata/image/cnet_models.hpp"
#include "strata/image/cnet_prep.hpp"
#include "strata/image/hunyuanimage3_file.hpp"
#include "strata/image/hunyuanimage3_pipeline.hpp"
#include "strata/image/img_crop.hpp"
#include "strata/image/qi_image_io.hpp"
#include "strata/image/mf_pipeline.hpp"
#include "strata/image/moss_pipeline.hpp"
#include "strata/image/qi_pipeline.hpp"
#include "strata/image/sam3.hpp"
#include "strata/image/sam3_video.hpp"
#include "strata/image/w8_choice.hpp"
#include "strata/image/yolo_detect.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/upscale/upscale.hpp"
#include "strata/platform/downgrade.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

namespace si = strata::image;
namespace su = strata::upscale;

std::vector<int32_t> parse_ids(const std::string& s) {
    std::vector<int64_t> ids;
    strata::program::parse_id_list(s, ids, strata::program::IdListForm::Image);   // stol's exceptions, as before
    return std::vector<int32_t>(ids.begin(), ids.end());
}

std::vector<float> parse_floats(const std::string& s) {
    std::vector<float> v;
    std::stringstream ss(s);
    for (std::string tok; std::getline(ss, tok, ',');)
        if (!tok.empty()) v.push_back(std::stof(tok));
    return v;
}

// the sampler's sigmas, commas: two or more, from at most 1 down to 0, falling
std::vector<float> parse_sigmas(const std::string& s) {
    std::vector<float> sig;
    std::stringstream ss(s);
    for (std::string tok; std::getline(ss, tok, ',');)
        if (!tok.empty()) sig.push_back(std::stof(tok));
    bool ok = sig.size() >= 2 && sig.front() <= 1.0f && sig.back() == 0.0f;
    for (size_t i = 1; ok && i < sig.size(); ++i) ok = sig[i] < sig[i - 1];
    if (!ok) throw std::invalid_argument("sigmas: two or more, falling from at most 1 to 0: " + s);
    return sig;
}

// a list of block indices, commas ("none" or nothing: no block)
std::vector<int> parse_blocks(const std::string& s) {
    std::vector<int> b;
    if (s == "none") return b;
    std::stringstream ss(s);
    for (std::string tok; std::getline(ss, tok, ',');)
        if (!tok.empty()) b.push_back(std::stoi(tok));
    return b;
}

// the chat engine's states file (TextEncoder::read_states) into rows
void read_states(const std::string& path, std::vector<float>& rows, int64_t& n) {
    std::string err;
    int64_t width = 0;
    if (!si::TextEncoder::read_states(path, rows, n, width, err)) throw std::invalid_argument(err);
}

si::QiRef read_image(const std::string& path) {
    si::QiRef f;
    std::string err;
    f.rgba = si::load_rgba8(path, f.width, f.height, err);
    if (f.rgba.empty()) throw std::invalid_argument(err);
    return f;
}

// The adapters the process has read: each file once, however many requests name it (the next request of the same set
// binds nothing: QiDit::set_loras keeps the bound key)
struct Loras {
    std::map<std::string, std::shared_ptr<const si::LoraFile>> files;
    std::shared_ptr<const si::LoraFile> get(const std::string& path) {
        const auto at = files.find(path);
        if (at != files.end()) return at->second;
        std::shared_ptr<const si::LoraFile> shared;
        try {
            shared = si::LoraFile::open(path);
        } catch (const std::exception& e) {   // a file the engine cannot read or does not take: the request's
            throw si::LoraRejected(e.what());
        }
        files.emplace(path, shared);
        return shared;
    }
};

// LORA:<strength>:<path> into the request's set: the strength first, the path the rest of the word (its own colons,
// as a Windows path has, are not read as the strength's).  The line is split on whitespace, so a path's separators - a
// space, a '#' - arrive %XX'd (the server's serve/loras.py writes them so) and are decoded here; an unescaped char, a
// '%' not followed by two hexadecimal digits among them, is kept as it is.
std::string decoded_path(const std::string& s) {
    const auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size() && hex(s[i + 1]) >= 0 && hex(s[i + 2]) >= 0) {
            out += (char) (hex(s[i + 1]) * 16 + hex(s[i + 2]));
            i += 2;
        } else {
            out += s[i];
        }
    }
    return out;
}

// EDITFMT:<moss|aitk|aitk-nokv|appended>: MOSS-Image's edit adapter's layout (QiRefLayout)
si::QiRefLayout ref_layout(const std::string& word) {
    const std::string f = word.substr(8);
    if (f == "moss") return si::QiRefLayout::MossEdit;
    if (f == "aitk") return si::QiRefLayout::AitkKv;
    if (f == "aitk-nokv") return si::QiRefLayout::AitkNoKv;
    if (f == "appended") return si::QiRefLayout::Appended;
    throw std::invalid_argument("EDITFMT:<moss|aitk|aitk-nokv|appended>: " + word);
}

// A number that is the whole of `text`, finite: false where it is not.
bool whole_number(const std::string& text, float& out) {
    size_t used = 0;
    try {
        out = std::stof(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    return !text.empty() && used == text.size() && std::isfinite(out);
}

void read_lora(const std::string& word, si::QiRequest& r, Loras& loras) {
    const std::string rest = word.substr(5);
    const size_t colon = rest.find(':');
    const std::string head = rest.substr(0, colon), path = colon == std::string::npos ? "" : rest.substr(colon + 1);
    const size_t at = head.find('@');
    const std::string scale = head.substr(0, at);
    float s = 0.0f;
    if (!whole_number(scale, s))
        throw std::invalid_argument("LORA:<strength>:<path>: the strength is not a number: " + word);
    std::vector<float> blocks;
    if (at != std::string::npos) {   // @<w0>,<w1>,...: a factor each block, none left out
        const std::string list = head.substr(at + 1);
        for (size_t i = 0; i <= list.size();) {
            const size_t comma = std::min(list.find(',', i), list.size());
            float b = 0.0f;
            if (!whole_number(list.substr(i, comma - i), b))
                throw std::invalid_argument("LORA:<strength>@<w0>,<w1>,...:<path>: a block weight is not a number: " +
                                            word);
            blocks.push_back(b);
            i = comma + 1;
        }
    }
    if (path.empty()) throw std::invalid_argument("LORA:<strength>:<path>: no file: " + word);
    r.loras.push_back({loras.get(decoded_path(path)), s, std::move(blocks)});
}

// How a line reads the words it carries: an IMG line's and the command line's take the images themselves, a NEED
// line's takes their sizes alone (its estimate reads a REF:'s header, never its pixels).
enum class WordHow { Take, Size };

// One reading of the words an IMG line, a NEED line and the command line all carry (they were three): each word it
// knows sets the same field of `r`, or the caller's own below.  `crop` is where CROP:'s box goes, null where the line
// has no place for one - and then CROP: is not that line's word, as a RATIO line's reading had it ("unknown word
// CROP:...").  The rest is a NEED line's own: `sizes` the REF:'s, `pictures` the PIC:'s (what set_need_context
// takes), `mask` and `init` whether it has a base image, `n_neg` guidance's negative prompt's ids, `states` whether
// TE: or NEGTE: named the chat engine's states.  `loras` holds the adapters' files: the serve engine's one set, the
// command line's own for --lora.
struct WordRead {
    WordRead(si::QiRequest& req, Loras& files, WordHow reads = WordHow::Take, std::optional<si::CropBox>* box = nullptr)
        : r(req), loras(files), how(reads), crop(box) {}
    si::QiRequest& r;
    Loras& loras;
    WordHow how;
    std::optional<si::CropBox>* crop;
    std::vector<si::QiRef> refs;
    std::vector<std::pair<int64_t, int64_t>> sizes, pictures;
    bool mask = false, init = false, states = false;
    int64_t n_neg = 0;
};

// One word of `t`: false when it is not that reading's word (each caller words a misread its own way, as it did when
// the readings were split).  A value it cannot read throws, from the same parser as before.
bool read_word(const std::string& w, WordRead& t) {
    si::QiRequest& r = t.r;
    const bool size = t.how == WordHow::Size;
    // both readings set these the same way
    if (w.rfind("LORA:", 0) == 0) {
        read_lora(w, r, t.loras);
        return true;
    }
    if (w.rfind("EDITFMT:", 0) == 0) {
        r.refs_layout = ref_layout(w);
        return true;
    }
    if (w.rfind("UPS:", 0) == 0) {
        r.upscaler = w.substr(4);
        return true;
    }
    if (w.rfind("UPSCOLOR:", 0) == 0) {
        r.upscaler_color = w.substr(9);
        return true;
    }
    if (w.rfind("CNET:", 0) == 0) {   // the ControlNet branch (QiPipeline::control_need)
        r.control_net = decoded_path(w.substr(5));
        return true;
    }
    if (w.rfind("CROP:", 0) == 0) {
        if (t.crop == nullptr) return false;   // a line with no place for one: not its word
        *t.crop = si::parse_crop(w);
        return true;
    }
    // an image word: the image itself, or a NEED line's size of it
    if (w.rfind("REF:", 0) == 0) {
        if (size) {
            int64_t wd = 0, ht = 0;
            if (!si::image_size(w.substr(4), wd, ht)) throw std::invalid_argument("not an image: " + w);
            t.sizes.emplace_back(wd, ht);
        } else {
            t.refs.push_back(read_image(w.substr(4)));
        }
        return true;
    }
    if (w.rfind("PIC:", 0) == 0) {
        if (size) {   // the text encoder's pictures (MossPipeline::set_pictures_given)
            int64_t wd = 0, ht = 0;
            if (!si::image_size(w.substr(4), wd, ht)) throw std::invalid_argument("not an image: " + w);
            t.pictures.emplace_back(wd, ht);
        } else {
            r.pictures.push_back(read_image(w.substr(4)));
        }
        return true;
    }
    if (w.rfind("MASK:", 0) == 0) {
        if (size) t.mask = true;
        else r.mask = read_image(w.substr(5));
        return true;
    }
    if (w.rfind("INIT:", 0) == 0) {
        if (size) {
            t.mask = true;
            t.init = true;
        } else {
            r.init = read_image(w.substr(5));
        }
        return true;
    }
    if (w.rfind("TE:", 0) == 0) {
        if (size) t.states = true;
        else read_states(w.substr(3), r.te, r.te_rows);
        return true;
    }
    if (w.rfind("NEGTE:", 0) == 0) {
        if (size) t.states = true;
        else read_states(w.substr(6), r.neg_te, r.neg_te_rows);
        return true;
    }
    if (w.rfind("NEG:", 0) == 0) {
        if (size) t.n_neg = std::stoll(w.substr(4));   // guidance's negative prompt of n ids
        else r.neg_ids = parse_ids(w.substr(4));
        return true;
    }
    if (size) {
        // A control map is encoded at the drawing's size; NEED needs its presence, not its pixels.
        if (w.rfind("MCTL:", 0) == 0) { r.mctl = decoded_path(w.substr(5)); return true; }
        if (w.rfind("MCOLOR:", 0) == 0) { r.mctl_color.width = 1; return true; }
        if (w.rfind("MDEPTH:", 0) == 0) { r.mctl_depth.width = 1; return true; }
        // present is all a NEED line reads of these: the IMG line reads their values (and checks them)
        if (w.rfind("COLOR:", 0) == 0 || w == "UPSCALE" || w.rfind("TILES:", 0) == 0 || w.rfind("CTRL:", 0) == 0 ||
            w.rfind("CTRLS:", 0) == 0 || w == "CTRLNOINPAINT" || w.rfind("MCTLENDS:", 0) == 0)
            return true;
        if (w.rfind("UPSIN:", 0) == 0) {
            const std::vector<int32_t> v = parse_ids(w.substr(6));
            if ((v.size() != 2 && v.size() != 4) || std::any_of(v.begin(), v.end(), [](int32_t x) { return x <= 0; }))
                throw std::invalid_argument("UPSIN:<width>,<height>[,<out width>,<out height>]: " + w);
            r.init.width = v[0];   // only its size: the base image an upscale starts from
            r.init.height = v[1];
            if (v.size() == 4) {   // the upscale's own size, past the drawing's (TILES:)
                r.upscale = true;
                r.width = v[2];
                r.height = v[3];
            }
            return true;
        }
        return false;   // an IMG line's word: the NEED line's reader says "not an image: ..."
    }
    // the IMG line's and the command line's own words
    if (w.rfind("SIGMAS:", 0) == 0) r.sigmas = parse_sigmas(w.substr(7));
    else if (w == "UPSCALE") r.upscale = true;
    else if (w.rfind("TILES:", 0) == 0) {
        const std::vector<int32_t> v = parse_ids(w.substr(6));
        if (v.size() != 3 || v[0] < 256 || v[1] < 0 || v[2] < 0)
            throw std::invalid_argument("TILES:<tile>,<pad>,<blur> (tile 256 or more): " + w);
        r.tile = v[0];
        r.tile_pad = v[1];
        r.tile_blur = v[2];
    }
    else if (w.rfind("COLOR:", 0) == 0) {
        const std::string m = w.substr(6);
        if (m != "off" && m != "auto" && m != "full") throw std::invalid_argument("COLOR:off|auto|full: " + w);
        r.crop_color = m == "auto" ? 1 : m == "full" ? 2 : 0;
    }
    else if (w.rfind("STRENGTH:", 0) == 0) r.strength = std::stod(w.substr(9));
    else if (w.rfind("CFG:", 0) == 0) r.cfg = std::stof(w.substr(4));
    else if (w.rfind("GUID:", 0) == 0) r.guidance = std::stof(w.substr(5));
    else if (w.rfind("STEPS:", 0) == 0) r.steps = std::stoll(w.substr(6));
    else if (w.rfind("PREVIEW:", 0) == 0)
        for (const int64_t k : parse_ids(w.substr(8))) r.preview.push_back((int) k);
    else if (w.rfind("SHIFT:", 0) == 0) r.shift = std::stof(w.substr(6));
    else if (w.rfind("USKIP:", 0) == 0) {
        r.uncond_skip = parse_blocks(w.substr(6));
        r.uncond_skip_set = true;
    }
    else if (w.rfind("SLG:", 0) == 0) r.slg_layers = parse_blocks(w.substr(4));
    else if (w.rfind("SLGSCALE:", 0) == 0) r.slg_scale = std::stof(w.substr(9));
    else if (w.rfind("SLGSTART:", 0) == 0) r.slg_start = std::stof(w.substr(9));
    else if (w.rfind("SLGSTOP:", 0) == 0) r.slg_stop = std::stof(w.substr(8));
    else if (w.rfind("SCHED:", 0) == 0) r.base_schedule = w.substr(6) == "base";
    else if (w.rfind("T5:", 0) == 0) r.t5_ids = parse_ids(w.substr(3));
    else if (w.rfind("T5W:", 0) == 0) r.t5_weights = parse_floats(w.substr(4));
    else if (w.rfind("NEGT5:", 0) == 0) r.neg_t5_ids = parse_ids(w.substr(6));
    else if (w.rfind("NEGT5W:", 0) == 0) r.neg_t5_weights = parse_floats(w.substr(7));
    else if (w.rfind("SAMPLER:", 0) == 0) r.sampler = w.substr(8);
    else if (w.rfind("CFGNORM:", 0) == 0) r.guide.cfgnorm = std::stof(w.substr(8));
    else if (w == "CFGZERO") r.guide.zero_star = true;
    else if (w.rfind("APG:", 0) == 0) {
        const std::vector<float> v = parse_floats(w.substr(4));
        if (v.size() != 3) throw std::invalid_argument("APG:<eta>,<norm_threshold>,<momentum>: " + w);
        r.guide.apg = true;
        r.guide.apg_eta = v[0];
        r.guide.apg_norm_threshold = v[1];
        r.guide.apg_momentum = v[2];
    }
    else if (w.rfind("DSLG:", 0) == 0) r.guide.slg_layers = parse_blocks(w.substr(5));
    else if (w.rfind("DSLGP:", 0) == 0) {
        const std::vector<float> v = parse_floats(w.substr(6));
        if (v.size() != 4)
            throw std::invalid_argument("DSLGP:<scale>,<sigma_start>,<sigma_end>,<rescaling_scale>: " + w);
        r.guide.slg_scale = v[0];
        r.guide.slg_sigma_start = v[1];
        r.guide.slg_sigma_end = v[2];
        r.guide.slg_rescale = v[3];
    }
    else if (w.rfind("DD:", 0) == 0) {
        const std::vector<float> v = parse_floats(w.substr(3));
        if (v.size() != 10)
            throw std::invalid_argument("DD:<amount>,<start>,<end>,<bias>,<exponent>,<start_offset>,<end_offset>,"
                                        "<fade>,<smooth 0|1>,<cfg_scale_override>: " + w);
        si::GuidanceSpec& g = r.guide;
        g.dd = true;
        g.dd_amount = v[0];
        g.dd_start = v[1];
        g.dd_end = v[2];
        g.dd_bias = v[3];
        g.dd_exponent = v[4];
        g.dd_start_offset = v[5];
        g.dd_end_offset = v[6];
        g.dd_fade = v[7];
        g.dd_smooth = v[8] != 0.0f;
        g.dd_cfg = v[9];
    }
    else if (w.rfind("CTRL:", 0) == 0) r.control = read_image(decoded_path(w.substr(5)));
    else if (w.rfind("CTRLS:", 0) == 0) {
        const auto v = parse_floats(w.substr(6));
        if (v.size() != 3 || !(v[2] >= 0.0f && v[2] <= v[1] && v[1] <= 1.0f) || !std::isfinite(v[0]))
            throw std::invalid_argument("CTRLS:<strength>,<sigma from>,<sigma to>: 0 <= to <= from <= 1: " + w);
        r.control_strength = v[0];
        r.control_sigma_hi = v[1];
        r.control_sigma_lo = v[2];
    }
    else if (w == "CTRLNOINPAINT") r.control_inpaint = false;
    else if (w.rfind("MCTL:", 0) == 0) r.mctl = decoded_path(w.substr(5));
    else if (w.rfind("MCOLOR:", 0) == 0) r.mctl_color = read_image(decoded_path(w.substr(7)));
    else if (w.rfind("MDEPTH:", 0) == 0) r.mctl_depth = read_image(decoded_path(w.substr(7)));
    else if (w.rfind("MCTLENDS:", 0) == 0) {
        const auto v = parse_floats(w.substr(9));
        if (v.size() != 2 || !(v[0] >= 0.0f && v[0] <= 1.0f && v[1] >= 0.0f && v[1] <= 1.0f))
            throw std::invalid_argument("MCTLENDS:<color end>,<depth end>: each 0 .. 1: " + w);
        r.mctl_color_end = v[0];
        r.mctl_depth_end = v[1];
    }
    else return false;
    return true;
}

// The command line's options that are the IMG line's words: --steps 8 reads as the word STEPS:8, through the same
// reader.  The rest of the options (a model's file, the output, the prompt's ids, width, height, seed and drop, which
// the line carries in its header rather than as words) are the command line's own.
bool cli_word(const std::string& a, std::string& word) {
    static const std::pair<const char*, const char*> same[] = {
        {"--mask", "MASK:"},           {"--init", "INIT:"},          {"--strength", "STRENGTH:"},
        {"--neg-ids", "NEG:"},         {"--cfg", "CFG:"},            {"--sigmas", "SIGMAS:"},
        {"--steps", "STEPS:"},         {"--guidance", "GUID:"},      {"--shift", "SHIFT:"},
        {"--uncond-skip", "USKIP:"},   {"--slg-layers", "SLG:"},     {"--slg-scale", "SLGSCALE:"},
        {"--slg-start", "SLGSTART:"},  {"--slg-stop", "SLGSTOP:"},   {"--sched", "SCHED:"},   {"--t5", "T5:"},
        {"--t5-weights", "T5W:"},      {"--neg-t5", "NEGT5:"},       {"--neg-t5-weights", "NEGT5W:"},
        {"--sampler", "SAMPLER:"},     {"--lora", "LORA:"},          {"--te", "TE:"},
        {"--neg-te", "NEGTE:"},
    };
    for (const auto& [opt, w] : same)
        if (a == opt) {
            word = w;
            return true;
        }
    return false;
}

// An IMG line's words (and a RATIO line's): its REF:<path> images (and SIGMAS:, MASK:, INIT:, STRENGTH:...); CROP:
// sets `crop` (null: refused)
std::vector<si::QiRef> read_refs(std::istringstream& ls, si::QiRequest& r, Loras& loras,
                                 std::optional<si::CropBox>* crop = nullptr) {
    WordRead t(r, loras, WordHow::Take, crop);
    for (std::string word; ls >> word;)
        if (!read_word(word, t)) throw std::invalid_argument("unknown word " + word);
    return t.refs;
}

// a NEED line's: the REF:<path> images' sizes; MASK:<path> or INIT:<path> (a base image) sets `mask` (INIT: `init`
// too), NEG:<n> `n_neg`, TE:<path> (the chat engine's states) `states`, CROP: `crop`, and LORA:<strength>:<path> goes
// into `r.loras`
std::vector<std::pair<int64_t, int64_t>> ref_sizes(std::istringstream& ls, bool& mask, bool& init, int64_t& n_neg,
                                                   bool& states, std::optional<si::CropBox>& crop,
                                                   std::vector<std::pair<int64_t, int64_t>>& pictures, si::QiRequest& r,
                                                   Loras& loras) {
    WordRead t(r, loras, WordHow::Size, &crop);
    for (std::string word; ls >> word;)
        if (!read_word(word, t)) throw std::invalid_argument("not an image: " + word);
    mask = t.mask;
    init = t.init;
    n_neg = t.n_neg;
    states = t.states;
    pictures = std::move(t.pictures);
    return t.sizes;
}

std::string plan_kv(const si::QiTimes& t, const si::QiPlan& p) {
    char b[768];
    std::snprintf(b, sizeof b,
                  "text_s=%.3f sampling_s=%.3f decode_s=%.3f budget_mib=%llu blocks_mib=%llu resident_blocks=%d "
                  "layers_mib=%llu resident_layers=%d activations_mib=%llu mlp_rows=%lld vae_mib=%llu vae_tiled=%d "
                  "text_cached=%d prefix_kv_mib=%llu resident_kv=%d streamed_mib=%llu lora_n=%d lora_mib=%llu",
                  t.text, t.sampling, t.decode, (unsigned long long) (p.budget >> 20),
                  (unsigned long long) (p.blocks >> 20), p.resident_blocks, (unsigned long long) (p.layers >> 20),
                  p.resident_layers, (unsigned long long) (p.activations >> 20), (long long) p.mlp_rows,
                  (unsigned long long) (p.vae >> 20), p.vae_tiled ? 1 : 0, p.text_cached ? 1 : 0,
                  (unsigned long long) (p.prefix_kv >> 20), p.resident_kv,
                  (unsigned long long) (p.streamed >> 20), p.lora_n, (unsigned long long) (p.lora >> 20));
    return b;
}

void write_file(const std::string& path, const std::vector<uint8_t>& img) {
    std::ofstream o(path, std::ios::binary);
    o.write(reinterpret_cast<const char*>(img.data()), (std::streamsize) img.size());
    if (!o) throw std::runtime_error("cannot write " + path);
}

// The detectors the process has opened (yolo_detect.hpp), by their file: each opened once, its weights kept beside the
// image model's (a few tens of MiB) until RELEASE ALL
struct Detectors {
    std::map<std::string, std::unique_ptr<si::YoloDetector>> open;
    si::YoloDetector& get(const std::string& path) {
        auto& d = open[path];
        if (!d) {
            auto n = std::make_unique<si::YoloDetector>();
            std::string err;
            if (!n->open(path, err)) {
                open.erase(path);
                throw std::runtime_error("detector: " + err);
            }
            d = std::move(n);
        }
        return *d;
    }
    uint64_t held() const {
        uint64_t n = 0;
        for (const auto& d : open) n += d.second->weight_bytes();
        return n;
    }
    void release(bool all) {
        if (all) open.clear();
        else
            for (auto& d : open) d.second->drop_scratch();
    }
};

// The upscalers the process has opened (strata/upscale/upscale.hpp), by their name and file (UPS:'s <name>[:<path>]):
// each opened once, its weights kept beside the image model's (RealESRGAN's 33 MiB) until RELEASE ALL, its scratch let
// go after each run
struct Upscalers {
    std::map<std::string, std::unique_ptr<su::UpscaleModel>> open;
    su::UpscaleModel& get(const std::string& spec) {
        auto& m = open[spec];
        if (!m) {
            const size_t colon = spec.find(':');
            const std::string name = spec.substr(0, colon);
            const std::string path = colon == std::string::npos ? "" : decoded_path(spec.substr(colon + 1));
            try {
                m = su::open_model(name, path);
            } catch (...) {
                open.erase(spec);
                throw;
            }
        }
        return *m;
    }
    uint64_t held() const {
        uint64_t n = 0;
        for (const auto& m : open) n += m.second->held();
        return n;
    }
    void release(bool all) {
        for (auto& m : open) m.second->release(all);
        if (all) open.clear();
    }
    // every other one's weights let go (a video's run, which takes the room they hold for its length)
    void release_except(const su::UpscaleModel& keep) {
        for (auto it = open.begin(); it != open.end();) {
            if (it->second.get() == &keep) {
                ++it;
                continue;
            }
            it->second->release(true);
            it = open.erase(it);
        }
    }
};

// The VRAM an upscale may take now: what the budget leaves beside the image model's holdings, within the free VRAM
// (the upscaler's own holdings counted as its)
uint64_t ups_room(si::ImagePipeline& p, su::UpscaleModel& m) {
    size_t free = 0, total = 0;
    strata::gpu::mem_info(&free, &total);
    const uint64_t b = p.budget(), held = p.held();
    return std::min<uint64_t>((uint64_t) free + m.held(), b > held ? b - held : 0);
}

// `src` upscaled by `m` as `rq` asks, in the room the image model leaves: its cache given back first where that is
// too little for the upscaler's smallest tiles, then its parts.  The share done goes to the server as PART lines
// (serve/images.py: the Monitor's progress)
si::QiRef run_upscale(si::ImagePipeline& p, su::UpscaleModel& m, const si::QiRef& src, const su::Request& rq,
                      su::Report& rep) {
    const su::ColorFix color = rq.color_set ? rq.color : m.info().color;
    const uint64_t least = su::UpscaleRunner::least_bytes(m, src.width, src.height, rq.width, rq.height, color);
    uint64_t avail = ups_room(p, m);
    if (avail < least) {
        p.release(false);
        avail = ups_room(p, m);
    }
    if (avail < least) {
        p.release(true);
        avail = ups_room(p, m);
    }
    if (m.held() == 0) {   // its weights read first (the first run, or after it let them go): the server says so
        std::printf("STAGE load\n");
        std::fflush(stdout);
    }
    return su::UpscaleRunner::run(m, src, rq, avail, &rep, [](double f) {
        std::printf("PART %.3f\n", f);
        std::fflush(stdout);
    });
}

constexpr uint64_t kUpsReserve = 512ull << 20;   // upscale_base's

// An upscale's base image at w x h before it is drawn again: by the request's upscaler (its time added to `ups_s`),
// else Lanczos.  The drawing plans its layers on the budget, which the free VRAM at the start set: after the upscaler
// the budget is fitted again as FIT does, the server's reserve (reserve_mib, 512) kept for what the drawing's
// libraries take outside it (the RX 9060 XT's second two-pass upscale ran out with 36 MiB to spare, RealESRGAN's
// weights and its libraries' state kept).  An upscaler that holds more than the reserve (SeedVR2's gigabytes) is let
// go first, weights and all, the VRAM it gives back added to the budget; a smaller one (RealESRGAN's 64 MiB) stays
// for the next
si::QiRef upscale_base(si::ImagePipeline& p, Upscalers& ups, const si::QiRequest& r, const si::QiRef& base, int64_t w,
                       int64_t h, double& ups_s) {
    if (r.upscaler.empty()) return si::lanczos_resize(base, w, h);
    std::printf("STAGE upscale\n");   // the server's progress (serve/images.py), PART lines after it
    std::fflush(stdout);
    const auto t0 = std::chrono::steady_clock::now();
    su::Request rq;
    rq.width = w;
    rq.height = h;
    if (!r.upscaler_color.empty()) {
        rq.color = su::parse_color(r.upscaler_color);
        rq.color_set = true;
    }
    su::Report rep;
    si::QiRef out = run_upscale(p, ups.get(r.upscaler), base, rq, rep);
    if (ups.held() > kUpsReserve) {   // what it gives back goes to the budget (a FIT while it was held left it out)
        const uint64_t before = p.held() + si::free_vram();
        ups.release(true);
        const uint64_t after = p.held() + si::free_vram();
        if (after > before) p.set_budget(p.budget() + (after - before));
    }
    if (const uint64_t have = p.held() + si::free_vram(); have < p.budget() + kUpsReserve)
        p.set_budget(have > kUpsReserve ? have - kUpsReserve : 0);
    ups_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

// The auxiliary models the process has opened (SAM 3, Florence-2, BiRefNet), by their file: opened once, their weights
// loaded at their first run and kept (aux_done) while the budget leaves room for them
struct AuxNets {
    std::map<std::string, std::unique_ptr<si::AuxNet>> open;
    template <class T> T& get(const std::string& path) {
        auto& m = open[path];
        if (!m) {
            try {
                m = std::make_unique<T>(path);
            } catch (...) {
                open.erase(path);
                throw;
            }
        }
        auto* t = dynamic_cast<T*>(m.get());
        if (t == nullptr) throw std::invalid_argument(path + ": opened as another model");
        return *t;
    }
    uint64_t held() const {
        uint64_t n = 0;
        for (const auto& m : open) n += m.second->held();
        return n;
    }
    void release(bool all) {
        for (auto& m : open) {
            if (all) m.second->release();
            else m.second->drop_scratch();
        }
        if (all) open.clear();
    }
};

// An auxiliary model's run told to the server (serve/images.py: the Monitor's progress): STAGE load where its weights
// are read first, then PART <share done> as the model tells it; unset again at the end
struct AuxTold {
    si::AuxNet& m;
    explicit AuxTold(si::AuxNet& n) : m(n) {
        if (!m.loaded()) {
            std::printf("STAGE load\n");
            std::fflush(stdout);
        }
        m.progress = [](double f) {
            std::printf("PART %.3f\n", f);
            std::fflush(stdout);
        };
    }
    ~AuxTold() { m.progress = nullptr; }
    AuxTold(const AuxTold&) = delete;
    AuxTold& operator=(const AuxTold&) = delete;
};

// Room for an auxiliary model's run that needs `need` bytes beyond what it holds: what the budget leaves beside the
// image model's holdings, within the free VRAM; the image model's cache given back first where that is too little,
// then its parts, as an upscale's (run_upscale), then the other auxiliary models' weights that aux_done kept (`aux`,
// all but `keep`: a chat's find_region after its remove_background found BiRefNet's still held and Florence-2 150 MiB
// short on the B70 beside the text model, 2026-11-09), the VRAM they give back added to the budget as aux_done adds it
void aux_room(si::ImagePipeline& p, uint64_t need, const char* what, AuxNets* aux = nullptr,
              const si::AuxNet* keep = nullptr, Upscalers* ups = nullptr) {
    const auto room = [&p] {
        const uint64_t b = p.budget(), held = p.held();
        return std::min<uint64_t>(si::free_vram(), b > held ? b - held : 0);
    };
    if (room() < need) p.release(false);
    if (room() < need) p.release(true);
    if (room() < need && ups != nullptr && ups->held() > 0) {
        const uint64_t before = p.held() + si::free_vram();
        ups->release(true);
        const uint64_t after = p.held() + si::free_vram();
        if (after > before) p.set_budget(p.budget() + (after - before));
    }
    if (room() < need && aux != nullptr) {
        for (auto it = aux->open.begin(); it != aux->open.end();) {
            if (it->second.get() == keep) {
                ++it;
                continue;
            }
            const uint64_t before = p.held() + si::free_vram();
            it->second->release();
            const uint64_t after = p.held() + si::free_vram();
            if (after > before) p.set_budget(p.budget() + (after - before));
            it = aux->open.erase(it);
        }
    }
    if (const uint64_t r = room(); r < need)
        throw std::runtime_error("memory: " + std::string(what) + " needs " + std::to_string(need >> 20) +
                                 " MiB, the GPU has " + std::to_string(r >> 20) + " MiB for it");
}

// After an auxiliary model's run: its scratch freed (the image model's budget, fitted around what the engine holds,
// has no room for it); its weights kept for the next tool's run while a quarter of the GPU's memory is still free
// (reading SAM 3's 1.7 GB again took 3.2 s of a 3.3 s run on the B70, 2026-11-07), else let go as upscale_base lets a
// large upscaler go, the VRAM given back added to the budget.  The next image job lets them go first (aux_yield)
void aux_done(si::ImagePipeline& p, si::AuxNet& m) {
    m.drop_scratch();
    size_t free = 0, total = 0;
    strata::gpu::mem_info(&free, &total);
    if (m.held() > kUpsReserve && free < total / 4) {
        const uint64_t before = p.held() + si::free_vram();
        m.release();
        const uint64_t after = p.held() + si::free_vram();
        if (after > before) p.set_budget(p.budget() + (after - before));
    }
}

// Before an image job (IMG, NEED, RATIO, UPS, UPSNEED): the auxiliary models' weights let go, so the image model has
// the room it had without them (its budget fitted again by the server's FIT after its NEED)
void aux_yield(AuxNets& aux) {
    if (!aux.open.empty()) aux.release(true);
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// SEGV's prompts file: a line a prompt, "W <from frame> <ids>" (words: CLIP's tokens, commas) or "C <frame> <object>
// <x,y,label,...>" (clicks on the clip's pixels: label 1 in, 0 out, 2 and 3 a box's corners; object -1 a new one)
void read_video_prompts(const std::string& path, std::vector<si::Sam3VideoWords>& words,
                        std::vector<si::Sam3VideoClick>& clicks) {
    std::ifstream f(path);
    if (!f) throw std::invalid_argument("SEGV: cannot read " + path);
    for (std::string line; std::getline(f, line);) {
        std::istringstream ls(line);
        std::string kind, list;
        if (!(ls >> kind)) continue;
        if (kind == "W") {
            si::Sam3VideoWords w;
            ls >> w.from >> list;
            w.ids = parse_ids(list);
            if (!ls || w.from < 0 || w.ids.size() < 2) throw std::invalid_argument("SEGV: W <from> <ids>");
            words.push_back(std::move(w));
        } else if (kind == "C") {
            si::Sam3VideoClick c;
            ls >> c.frame >> c.object >> list;
            const std::vector<float> v = parse_floats(list);
            if (!ls || c.frame < 0 || v.empty() || v.size() % 3 != 0)
                throw std::invalid_argument("SEGV: C <frame> <object> <x,y,label,...>");
            for (size_t i = 0; i < v.size(); i += 3) c.points.push_back({v[i], v[i + 1], v[i + 2]});
            clicks.push_back(std::move(c));
        } else {
            throw std::invalid_argument("SEGV: a prompt line is W or C");
        }
    }
    if (words.empty() && clicks.empty()) throw std::invalid_argument("SEGV: no words and no clicks");
}

// A mask (width x height, 255 inside) as COCO's uncompressed RLE: the runs down the columns, the first counting the
// background (vLLM-Omni's rle_counts)
void rle_json(const std::vector<uint8_t>& m, int64_t w, int64_t h, std::string& out) {
    bool cur = false;
    int64_t run = 0;
    bool first = true;
    const auto put = [&](int64_t v) {
        if (!first) out += ',';
        first = false;
        out += std::to_string(v);
    };
    for (int64_t x = 0; x < w; ++x)
        for (int64_t y = 0; y < h; ++y) {
            const bool v = m[(size_t) (y * w + x)] != 0;
            if (v != cur) {
                put(run);
                run = 0;
                cur = v;
            }
            ++run;
        }
    put(run);
}

struct SegvOptions {
    bool masks = false, mask = false, overlay = false;   // a video an object, one of them all, the clip overlaid
    strata::video::Codec codec = strata::video::Codec::automatic;
    std::function<void()> pause;                         // between frames, every kSegvYieldS seconds
    std::function<bool(int64_t, int64_t)> step;          // the frames done, of about how many; false: stop
};

constexpr double kSegvYieldS = 1.0;   // the chat's wait at most, a yield to it

// SEGV: the clip `in` through SAM 3.1's video segmentation (sam3_video.hpp) a frame at a time, each frame's instances
// as a JSON line in <stem>.jsonl ({"frame_index", "instances": [{"id", "score", "box" [x0, y0, x1, y1] (the last
// pixels), "mask": {"size": [h, w], "counts"}}]}: vLLM-Omni's frames), and the videos asked for -> SEGVIDEO's line
std::string segment_video(si::Sam3& m, const std::string& in, const std::string& prompts, const std::string& stem,
                          const SegvOptions& o) {
    namespace sv = strata::video;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<si::Sam3VideoWords> words;
    std::vector<si::Sam3VideoClick> clicks;
    read_video_prompts(prompts, words, clicks);
    const sv::Info info = sv::probe(in);
    const int64_t W = info.width, H = info.height, WH = W * H;
    si::Sam3Video vs(m, W, H, info.frames, words, clicks);
    sv::Reader rd(in, info);
    const std::string json = stem + ".jsonl";
    std::ofstream js(json, std::ios::binary);
    if (!js) throw std::runtime_error("cannot write " + json);
    sv::WriteOptions gray;
    gray.codec = o.codec;
    gray.gray = true;
    std::map<int64_t, std::unique_ptr<sv::Writer>> per;   // by object
    std::unique_ptr<sv::Writer> all, over;
    if (o.mask) all = std::make_unique<sv::Writer>(stem + ".mask", W, H, info.fps, gray);
    if (o.overlay) {
        sv::WriteOptions wo;
        wo.codec = o.codec;
        wo.audio_from = in;
        over = std::make_unique<sv::Writer>(stem + ".overlay", W, H, info.fps, wo);
    }
    // the overlay's colors by id (Tableau's ten), half over the picture
    static const uint8_t kColors[10][3] = {{31, 119, 180}, {255, 127, 14}, {44, 160, 44}, {214, 39, 40},
                                           {148, 103, 189}, {140, 86, 75}, {227, 119, 194}, {127, 127, 127},
                                           {188, 189, 34}, {23, 190, 207}};
    std::deque<std::vector<uint8_t>> held;   // the frames read, until their results come out (the overlay's)
    std::vector<uint8_t> rgb((size_t) (WH * 3)), black((size_t) WH, 0), one((size_t) WH);
    std::map<int64_t, std::pair<int64_t, int64_t>> objects;   // id -> its words' or its click's (-1 the other)
    int64_t read = 0, shown = 0;
    double read_s = 0, write_s = 0;
    const auto emit = [&](const std::vector<si::Sam3Video::Frame>& fs) {
        for (const auto& f : fs) {
            const auto w0 = std::chrono::steady_clock::now();
            std::string line = "{\"frame_index\":" + std::to_string(f.index) + ",\"instances\":[";
            char b[160];
            std::fill(one.begin(), one.end(), 0);
            for (size_t i = 0; i < f.objects.size(); ++i) {
                const auto& ob = f.objects[i];
                objects.emplace(ob.id, std::make_pair(ob.words, ob.click));
                std::snprintf(b, sizeof b, "%s{\"id\":%lld,\"score\":%.4f,\"box\":[%lld,%lld,%lld,%lld],\"mask\":"
                              "{\"size\":[%lld,%lld],\"counts\":[", i ? "," : "", (long long) ob.id, ob.score,
                              (long long) ob.x0, (long long) ob.y0, (long long) (ob.x1 - 1), (long long) (ob.y1 - 1),
                              (long long) H, (long long) W);
                line += b;
                rle_json(ob.mask, W, H, line);
                line += "]}}";
                for (int64_t px = 0; px < WH; ++px) one[(size_t) px] |= ob.mask[(size_t) px];
            }
            line += "]}\n";
            js << line;
            if (o.masks) {
                for (const auto& ob : f.objects) {
                    auto& wr = per[ob.id];
                    if (!wr) {   // an object first seen here: black for the frames before
                        wr = std::make_unique<sv::Writer>(stem + ".obj" + std::to_string(ob.id), W, H, info.fps, gray);
                        for (int64_t k = 0; k < f.index; ++k) wr->write(black.data(), 1);
                    }
                }
                for (auto& [id, wr] : per) {
                    const auto it = std::find_if(f.objects.begin(), f.objects.end(),
                                                 [id = id](const auto& ob) { return ob.id == id; });
                    wr->write(it == f.objects.end() ? black.data() : it->mask.data(), 1);
                }
            }
            if (all) all->write(one.data(), 1);
            if (over) {
                std::vector<uint8_t>& pic = held.front();
                for (const auto& ob : f.objects) {
                    const uint8_t* c = kColors[ob.id % 10];
                    for (int64_t px = 0; px < WH; ++px)
                        if (ob.mask[(size_t) px])
                            for (int k = 0; k < 3; ++k)
                                pic[(size_t) (px * 3 + k)] = (uint8_t) ((pic[(size_t) (px * 3 + k)] + c[k] + 1) / 2);
                }
                over->write(pic.data(), 1);
            }
            if (!held.empty()) held.pop_front();
            write_s += seconds_since(w0);
            ++shown;
        }
    };
    auto last_yield = std::chrono::steady_clock::now();
    for (;;) {
        const auto r0 = std::chrono::steady_clock::now();
        const int64_t n = rd.read(rgb.data(), 1);
        read_s += seconds_since(r0);
        if (n <= 0) break;
        si::QiRef pic;
        pic.width = W;
        pic.height = H;
        pic.rgba.resize((size_t) (WH * 4));
        for (int64_t px = 0; px < WH; ++px) {
            for (int c = 0; c < 3; ++c) pic.rgba[(size_t) (px * 4 + c)] = rgb[(size_t) (px * 3 + c)];
            pic.rgba[(size_t) (px * 4 + 3)] = 255;
        }
        if (over) held.push_back(rgb);
        emit(vs.push(pic));
        ++read;
        if (o.step && !o.step(read, std::max(info.frames, read))) throw std::runtime_error("SEGV: the video was stopped");
        if (o.pause && seconds_since(last_yield) >= kSegvYieldS) {
            o.pause();
            last_yield = std::chrono::steady_clock::now();
        }
    }
    emit(vs.finish());
    js.close();
    if (!js) throw std::runtime_error("cannot write " + json);
    const auto w0 = std::chrono::steady_clock::now();
    std::string paths;
    for (auto& [id, wr] : per) {
        wr->finish();
        paths += (paths.empty() ? "" : ",") + std::to_string(id) + ":" + wr->path();
    }
    if (all) all->finish();
    if (over) over->finish();
    write_s += seconds_since(w0);
    std::string objs;
    for (const auto& [id, src] : objects)
        objs += (objs.empty() ? "" : ",") + std::to_string(id) +
                (src.first >= 0 ? ":w" + std::to_string(src.first) : ":c" + std::to_string(src.second));
    const auto& t = vs.times();
    char b[512];
    std::snprintf(b, sizeof b, " frames=%lld width=%lld height=%lld fps=%.6f vfr=%d segment_s=%.3f read_s=%.3f "
                  "trunk_s=%.3f detect_s=%.3f track_s=%.3f memory_s=%.3f rules_s=%.3f output_s=%.3f write_s=%.3f",
                  (long long) shown, (long long) W, (long long) H, info.fps, (int) info.vfr, seconds_since(t0), read_s,
                  t.trunk, t.detect, t.track, t.memory, t.rules, t.output, write_s);
    std::string line = "SEGVIDEO json=" + json + " objects=" + (objs.empty() ? "-" : objs);
    if (!paths.empty()) line += " masks=" + paths;
    if (all) line += " mask=" + all->path() + " mime=" + all->mime();
    if (over) line += " overlay=" + over->path();
    return line + b;
}

// DETECT's answer: "DETECT n=<count> detect_s=<seconds> boxes=<x0>,<y0>,<x1>,<y1>,<score>,<class>;..." (the boxes on
// the picture, highest score first), each mask (a segmentation head's, with `masks` given) written to <masks>.<i> as
// width x height bytes, 255 inside.  The detection's buffers are freed after it (`keep`: kept for the next, a timing's
// repeats): the image model's budget, fitted around what the engine holds, has no room for them
std::string detect_line(si::YoloDetector& d, const si::QiRef& img, float conf, const std::string& masks,
                        bool keep = false) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<uint8_t>> mk;
    const auto dets = d.detect(img, conf, masks.empty() ? nullptr : &mk);
    if (!keep) d.drop_scratch();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::string line = "DETECT n=" + std::to_string(dets.size());
    char b[160];
    std::snprintf(b, sizeof b, " detect_s=%.4f boxes=", secs);
    line += b;
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& e = dets[i];
        std::snprintf(b, sizeof b, "%s%.2f,%.2f,%.2f,%.2f,%.4f,%d", i ? ";" : "", e.x0, e.y0, e.x1, e.y1, e.score, e.cls);
        line += b;
    }
    for (size_t i = 0; i < mk.size(); ++i) {
        for (uint8_t& v : mk[i]) v = v ? 255 : 0;
        write_file(masks + "." + std::to_string(i), mk[i]);
    }
    return line;
}

// The color map of MOSS-Image's color control: each instance found (SAM 3's, every words' together) painted in a
// flat color of its own, the larger first so that a smaller one in front of it stays seen, black where none is: RGBA8
// rows.  The colors are ids only (the adapter reads regions, not hues): hues a golden-ratio step apart, never black.
std::vector<uint8_t> color_map(const std::vector<std::vector<si::Segment>>& found, int64_t W, int64_t H) {
    struct One {
        const si::Segment* s;
        size_t area;
    };
    std::vector<One> all;
    for (const auto& f : found)
        for (const auto& s : f) {
            size_t a = 0;
            for (uint8_t v : s.mask) a += v != 0;
            if (a > 0) all.push_back({&s, a});
        }
    std::stable_sort(all.begin(), all.end(), [](const One& a, const One& b) { return a.area > b.area; });
    std::vector<uint8_t> out((size_t) (W * H * 4), 0);
    for (size_t i = 3; i < out.size(); i += 4) out[i] = 255;
    for (size_t k = 0; k < all.size(); ++k) {
        const double h6 = std::fmod(0.11 + 0.6180339887 * (double) k, 1.0) * 6.0, f = h6 - std::floor(h6);
        const double v = 0.92, s = 0.8, p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
        double rgb[3];
        switch ((int) h6) {
        case 0: rgb[0] = v, rgb[1] = t, rgb[2] = p; break;
        case 1: rgb[0] = q, rgb[1] = v, rgb[2] = p; break;
        case 2: rgb[0] = p, rgb[1] = v, rgb[2] = t; break;
        case 3: rgb[0] = p, rgb[1] = q, rgb[2] = v; break;
        case 4: rgb[0] = t, rgb[1] = p, rgb[2] = v; break;
        default: rgb[0] = v, rgb[1] = p, rgb[2] = q; break;
        }
        const auto& m = all[k].s->mask;
        for (size_t px = 0; px < m.size(); ++px)
            if (m[px] != 0)
                for (int c = 0; c < 3; ++c) out[px * 4 + (size_t) c] = (uint8_t) std::lround(rgb[c] * 255.0);
    }
    return out;
}

// A control image's networks (`kind`'s, opened once in `aux` from their weights: W:'s, and the pose's W2:, DWPose's
// beside YOLOX's) into `nets` and `used`, and the VRAM a run on a picture of width x height takes beyond what the
// engine holds: their weights where they are not loaded yet, their scratch and the preprocessing's
uint64_t control_plan(AuxNets& aux, const std::string& kind, int64_t width, int64_t height, const si::ControlParams& cp,
                      const std::string& weights, const std::string& weights2, si::ControlNets& nets,
                      std::vector<si::AuxNet*>& used) {
    const si::ControlKind* k = si::ControlPrep::find(kind);
    if (k == nullptr) throw std::invalid_argument("CONTROL: no preprocessor " + kind);
    const std::string net = k->net;
    if (!net.empty() && (weights.empty() || (net == "pose" && weights2.empty())))
        throw std::invalid_argument("CONTROL " + kind + ": its weights (W:" + std::string(net == "pose" ? ", W2:" : "") +
                                    ")");
    // the processed size: the shorter side cp.resolution, the sides padded to multiples of 64
    const double side = (double) cp.resolution / (double) std::min(width, height);
    const int64_t pw = (std::lround((double) width * side) + 63) / 64 * 64;
    const int64_t ph = (std::lround((double) height * side) + 63) / 64 * 64;
    uint64_t need = si::ControlPrep::work_bytes(width, height, cp);
    const auto take = [&](auto& m, uint64_t run_bytes) {
        need += (m.loaded() ? 0 : m.file_bytes()) + run_bytes;
        used.push_back(&m);
        return &m;
    };
    if (net == "depth") {
        auto& m = aux.get<si::DepthAnything2>(weights);
        nets.depth = take(m, m.need_bytes(width, height));
    } else if (net == "normal") {
        auto& m = aux.get<si::BaeNormal>(weights);
        nets.normal = take(m, m.need_bytes(pw, ph));
    } else if (net == "segmentation") {
        auto& m = aux.get<si::UniformerSeg>(weights);
        nets.segmentation = take(m, m.need_bytes(pw, ph));
    } else if (net == "hed") {
        auto& m = aux.get<si::HedNet>(weights);
        nets.hed = take(m, m.need_bytes(pw, ph));
    } else if (net == "lineart" || net == "lineart_coarse") {
        auto& m = aux.get<si::LineartNet>(weights);
        nets.lineart = take(m, m.need_bytes(pw, ph));
    } else if (net == "mlsd") {
        auto& m = aux.get<si::MlsdNet>(weights);
        nets.mlsd = take(m, m.need_bytes(pw, ph));
    } else if (net == "pose") {
        auto& y = aux.get<si::YoloxNet>(weights);
        nets.yolox = take(y, y.need_bytes());
        auto& d = aux.get<si::DwPoseNet>(weights2);
        nets.dwpose = take(d, d.need_bytes());
    }
    return need;
}

// CONTROL's words after <kind> <picture> <out>: the map's parameters and its networks' weights (W:, and for the pose
// W2:, DWPose's beside YOLOX's), each network opened once in `aux`; `p` the image model whose room a run takes (null:
// the engine alone, --control)
std::string control_line(si::ControlPrep& prep, AuxNets& aux, si::ImagePipeline* p, const std::string& kind,
                         const si::QiRef& img, const std::string& out, std::istream& words, bool keep = false,
                         Upscalers* ups = nullptr) {
    si::ControlParams cp;
    std::string weights, weights2;
    const auto flag = [](const std::string& v) { return v != "0"; };
    for (std::string w; words >> w;) {
        if (w.rfind("RES:", 0) == 0) cp.resolution = std::stoi(w.substr(4));
        else if (w.rfind("LOW:", 0) == 0) cp.low = std::stof(w.substr(4));
        else if (w.rfind("HIGH:", 0) == 0) cp.high = std::stof(w.substr(5));
        else if (w.rfind("THR:", 0) == 0) cp.threshold = std::stof(w.substr(4));
        else if (w.rfind("SAFE:", 0) == 0) cp.safe = flag(w.substr(5));
        else if (w.rfind("SCORE:", 0) == 0) cp.score = std::stof(w.substr(6));
        else if (w.rfind("DIST:", 0) == 0) cp.distance = std::stof(w.substr(5));
        else if (w.rfind("BODY:", 0) == 0) cp.body = flag(w.substr(5));
        else if (w.rfind("HANDS:", 0) == 0) cp.hands = flag(w.substr(6));
        else if (w.rfind("FACE:", 0) == 0) cp.face = flag(w.substr(5));
        else if (w.rfind("W2:", 0) == 0) weights2 = decoded_path(w.substr(3));
        else if (w.rfind("W:", 0) == 0) weights = decoded_path(w.substr(2));
        else throw std::invalid_argument("CONTROL: an unknown word " + w);
    }
    si::ControlNets nets;
    std::vector<si::AuxNet*> used;
    const uint64_t need = control_plan(aux, kind, img.width, img.height, cp, weights, weights2, nets, used);
    const auto t0 = std::chrono::steady_clock::now();
    if (p != nullptr) aux_room(*p, need, kind.c_str(), nullptr, nullptr, ups);
    si::QiRef map;
    try {
        map = prep.run(kind, img, cp, nets);
    } catch (...) {
        prep.release();
        for (si::AuxNet* m : used) m->drop_scratch();
        throw;
    }
    if (!keep) {
        prep.release();
        for (si::AuxNet* m : used) {
            if (p != nullptr) aux_done(*p, *m);
            else m->drop_scratch();
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    write_file(out, map.rgba);
    char b[160];
    std::snprintf(b, sizeof b, "CONTROL width=%lld height=%lld control_s=%.4f", (long long) map.width,
                  (long long) map.height, secs);
    return b;
}

// The cut's box (img_crop.hpp) of a control image that is the picture's: the image at the picture's size
// (bicubic), the box cut out of it with its edges repeated where the box passes the picture
si::QiRef control_box(const si::QiRef& control, const si::CropBox& b) {
    const std::vector<uint8_t> whole = si::control_resize(control, b.width, b.height);
    si::QiRef out;
    out.width = b.w;
    out.height = b.h;
    out.rgba.resize((size_t) (b.w * b.h * 4));
    for (int64_t y = 0; y < b.h; ++y)
        for (int64_t x = 0; x < b.w; ++x) {
            const int64_t sy = std::clamp<int64_t>(b.y + y, 0, b.height - 1), sx = std::clamp<int64_t>(b.x + x, 0, b.width - 1);
            std::memcpy(&out.rgba[(size_t) ((y * b.w + x) * 4)], &whole[(size_t) ((sy * b.width + sx) * 4)], 4);
        }
    return out;
}

// An upscale past the model's size (TILES:), drawn again a tile at a time: the base image at the line's size
// (upscale_base: Lanczos, or the line's upscaler), then each tile of r.tile pixels (row by row) cut out with r.tile_pad of context around it, drawn again
// from it at the line's strength at that size (its sides multiples of 16), and put back by the tile's mask blurred by
// r.tile_blur (the cut's steps: img_crop.hpp), the next tile cut from the picture as it now is.  The tiles'
// times summed into `t`, the last one's plan in `pl`, the resizes and the cuts' time in `cut_s`.
std::vector<uint8_t> upscale_tiles(si::ImagePipeline& p, const si::QiRequest& r0, si::QiTimes& t, si::QiPlan& pl,
                                   double& cut_s, strata::core::StdinLines& in, Upscalers& ups, double& ups_s) {
    const si::QiRef* base0 = !r0.init.rgba.empty() ? &r0.init : !r0.refs.empty() ? &r0.refs[0] : nullptr;
    if (base0 == nullptr) throw std::invalid_argument("TILES: a base image (INIT: or the first REF:)");
    const int64_t W = r0.width, H = r0.height, T = r0.tile;
    auto t0 = std::chrono::steady_clock::now();
    const double ups0 = ups_s;
    si::QiRef pic = upscale_base(p, ups, r0, *base0, W, H, ups_s);
    cut_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() - (ups_s - ups0);
    const int64_t nx = (W + T - 1) / T, ny = (H + T - 1) / T;
    int index = 0;
    for (int64_t ty = 0; ty < ny; ++ty)
        for (int64_t tx = 0; tx < nx; ++tx, ++index) {
            const int64_t x0 = tx * T, y0 = ty * T, x1 = std::min(W, x0 + T), y1 = std::min(H, y0 + T);
            // the box: the tile and its context, its sides multiples of 16 (the drawing is the box's size)
            const int64_t bw = ((x1 - x0) + 2 * r0.tile_pad + 15) / 16 * 16, bh = ((y1 - y0) + 2 * r0.tile_pad + 15) / 16 * 16;
            si::CropBox box;
            box.x = x0 - (bw - (x1 - x0)) / 2;
            box.y = y0 - (bh - (y1 - y0)) / 2;
            box.w = bw;
            box.h = bh;
            box.blend = r0.tile_blur;
            box.width = W;
            box.height = H;
            si::QiRef mask;   // the tile changes (alpha 0), the rest of the picture stays
            mask.width = W;
            mask.height = H;
            mask.rgba.assign((size_t) (W * H * 4), 255);
            for (int64_t y = y0; y < y1; ++y)
                for (int64_t x = x0; x < x1; ++x) mask.rgba[(size_t) ((y * W + x) * 4 + 3)] = 0;
            si::CropStitch cs;
            t0 = std::chrono::steady_clock::now();
            cs.prepare(pic, mask, box, bw, bh);
            cut_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            si::QiRequest r = r0;
            r.width = bw;
            r.height = bh;
            r.seed = r0.seed + (uint64_t) index;
            r.upscale = false;
            r.tile = 0;
            r.refs.clear();
            r.init = cs.crop();
            r.mask = cs.regen();
            r.mask_share = cs.share();
            std::printf("TILE %d %lld\n", index + 1, (long long) nx * (long long) ny);   // the server's progress: tile i of n
            std::fflush(stdout);
            si::QiTimes tt;
            const auto img = p.generate(r, &tt, &pl, [&in](int done, int steps) {
                std::printf("STEP %d %d\n", done, steps);
                std::fflush(stdout);
                return !in.stop().load();
            }, [](const char* stage) {
                std::printf("STAGE %s\n", stage);
                std::fflush(stdout);
            });
            t.text += tt.text;
            t.sampling += tt.sampling;
            t.decode += tt.decode;
            t0 = std::chrono::steady_clock::now();
            pic.rgba = cs.stitch(img);
            cut_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
    return pic.rgba;
}

int serve(si::ImagePipeline& p) {
    Loras loras;   // the adapters' files, read once for the requests that name them
    Detectors detectors;
    Upscalers ups;
    AuxNets aux;
    si::ControlPrep control;
    {
        // the warm-up: a 256 x 256 image of one step from token ids the vocabulary has; where the GPU has no room
        // for it now (another program holds it), the first request builds the kernels instead.  A model whose
        // sequence has a shape of its own asks for its own request instead (HunyuanImage 3.0's ids carry the block).
        si::QiRequest r;
        if (!p.warm_up_request(r)) {
            for (int32_t i = 1; i <= 16; ++i) r.ids.push_back(i * 100);
            r.width = r.height = 256;
            r.sigmas = {1.0f, 0.0f};
        }
        try {
            p.generate(r);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "xestrata-image: no warm-up (%s)\n", e.what());
        }
        p.release(true);   // idle, the engine holds next to nothing on the GPU: the first request opens the parts
    }
    std::printf("READY budget_mib=%llu\n", (unsigned long long) (p.budget() >> 20));
    std::fflush(stdout);
    // the reader outlives serve(): after QUIT the server closes stdin, and the read it wakes from would lock a destroyed
    // mutex (std::system_error, the engine aborting while the GPU's memory is still its own)
    strata::core::StdinLines& in = *new strata::core::StdinLines;   // the ctor starts the detached reader
    for (std::string line; in.next(line);) {
        // the fallbacks so far (the warm-up's, the last request's), never before READY: the server reads that first
        strata::platform::downgrades_say();
        bool retried_aux = false;
    retry_aux:
        std::istringstream ls(line);
        std::string op;
        ls >> op;
        try {
            if (op == "IMG" || op == "UPSV" || op == "SEGV") in.stop() = false;
            if (op == "IMG" || op == "NEED" || op == "RATIO" || op == "UPS" || op == "UPSNEED" || op == "UPSV" ||
                op == "UPSVNEED")
                aux_yield(aux);
            if (op == "QUIT") {
                break;
            } else if (op == "IMG") {
                si::QiRequest r;
                std::string out, ids;
                ls >> r.width >> r.height >> r.seed >> r.drop >> out >> ids;
                if (!ls || out.empty()) throw std::invalid_argument("IMG <width> <height> <seed> <drop> <out> <ids>");
                r.ids = parse_ids(ids);
                std::optional<si::CropBox> crop;
                r.refs = read_refs(ls, r, loras, &crop);
                if (!r.loras.empty() && !p.takes_loras()) throw si::LoraRejected(p.no_loras_message());
                if (!r.control_net.empty() && !p.takes_control_net())
                    throw std::invalid_argument("image engine: CNET: is Qwen-Image-2.1's ControlNet; this model takes "
                                                "a control image as a reference image instead");
                if (!r.mctl.empty() && !p.takes_moss_options())
                    throw std::invalid_argument("image engine: MCTL: is MOSS-Image's");
                if (r.guide.any() && !p.takes_guidance())
                    throw std::invalid_argument("image engine: this image model takes no CFGNORM:, CFGZERO, APG:, "
                                                "DSLG: or DD:");
                // the cut (img_crop.hpp): the mask's box drawn at the line's size, the picture put back at its
                // own; the base image is the init image, else the first reference
                std::unique_ptr<si::CropStitch> cs;
                double crop_s = 0.0;   // its steps' time (the IMAGE line's crop_s)
                const auto timed = [&crop_s](const auto& f) {
                    const auto t0 = std::chrono::steady_clock::now();
                    f();
                    crop_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                };
                if (r.tile > 0) {   // an upscale past the model's size: tile by tile (upscale_tiles)
                    if (crop) throw std::invalid_argument("TILES: not with CROP:");
                    if (!r.control_net.empty()) throw std::invalid_argument("TILES: not with CNET: (a control per tile)");
                    si::QiTimes t;
                    si::QiPlan pl;
                    double cut_s = 0.0, ups_s = 0.0;
                    const auto img = upscale_tiles(p, r, t, pl, cut_s, in, ups, ups_s);
                    write_file(out, img);
                    std::printf("IMAGE %s crop_s=%.3f ups_s=%.3f\n", plan_kv(t, pl).c_str(), cut_s, ups_s);
                    std::fflush(stdout);
                    continue;
                }
                double ups_s = 0.0;   // the upscaler's time (UPS:), apart from crop_s
                if (!r.upscaler.empty() && !r.upscale && !crop) throw std::invalid_argument("UPS: with UPSCALE or CROP:");
                if (r.upscale) {   // the base image at the line's size (Lanczos, or the upscaler) before it is drawn again
                    si::QiRef* base = !r.init.rgba.empty() ? &r.init : !r.refs.empty() ? &r.refs[0] : nullptr;
                    if (base == nullptr) throw std::invalid_argument("UPSCALE: a base image (INIT: or the first REF:)");
                    timed([&] { *base = upscale_base(p, ups, r, *base, r.width, r.height, ups_s); });
                    crop_s -= ups_s;
                }
                if (crop) {
                    si::QiRef* base = !r.init.rgba.empty() ? &r.init : !r.refs.empty() ? &r.refs[0] : nullptr;
                    if (base == nullptr || r.mask.rgba.empty())
                        throw std::invalid_argument("CROP: a mask and its base image (INIT: or the first REF:)");
                    cs = std::make_unique<si::CropStitch>();
                    // UPS: with CROP: (the request's crop_upscale): the box enlarged by that upscaler, not Lanczos
                    std::optional<si::QiRef> scaled;
                    if (!r.upscaler.empty() && !r.upscale) {
                        timed([&] {
                            scaled = upscale_base(p, ups, r, si::CropStitch::cut(*base, *crop), r.width, r.height,
                                                  ups_s);
                        });
                        crop_s -= ups_s;
                    }
                    timed([&] {
                        cs->prepare(*base, r.mask, *crop, r.width, r.height, r.crop_color > 0,
                                    scaled ? &*scaled : nullptr);
                    });
                    if (crop->ref_w > 0 && base != &r.init) {   // the base the init image, the reference its own size
                        r.init = cs->crop();
                        *base = cs->ref();
                    } else {
                        *base = cs->crop();
                    }
                    r.mask = cs->regen();
                    r.mask_share = cs->share();
                    // the control image is the picture's: the same box of it (the edges repeated past the picture)
                    if (!r.control.rgba.empty()) r.control = control_box(r.control, *crop);
                }
                const int color = r.crop_color;
                r.on_preview = [&out, &cs, color](int index, int done, const std::vector<uint8_t>& rgba) {
                    const std::string path = out + ".p" + std::to_string(index);
                    write_file(path, cs ? cs->stitch(rgba, color) : rgba);
                    std::printf("PREVIEW %d %d %s\n", index, done, path.c_str());
                    std::fflush(stdout);
                };
                si::QiTimes t;
                si::QiPlan pl;
                const auto img = p.generate(r, &t, &pl, [&in](int done, int steps) {
                    std::printf("STEP %d %d\n", done, steps);
                    std::fflush(stdout);
                    return !in.stop().load();
                }, [](const char* stage) {   // the server's progress (serve/images.py)
                    std::printf("STAGE %s\n", stage);
                    std::fflush(stdout);
                });
                if (cs) {
                    std::vector<uint8_t> whole;
                    timed([&] { whole = cs->stitch(img, r.crop_color); });
                    write_file(out, whole);
                    std::printf("IMAGE %s crop_s=%.3f ups_s=%.3f\n", plan_kv(t, pl).c_str(), crop_s, ups_s);
                } else if (r.upscale) {
                    write_file(out, img);
                    std::printf("IMAGE %s crop_s=%.3f ups_s=%.3f\n", plan_kv(t, pl).c_str(), crop_s, ups_s);
                } else {
                    write_file(out, img);
                    std::printf("IMAGE %s\n", plan_kv(t, pl).c_str());
                }
            } else if (op == "UPS") {   // UPS <width> <height> <in> <out> <name>[:<path>] [COLOR:] [SEED:] [TILE:]
                su::Request rq;
                std::string src, dst, spec;
                ls >> rq.width >> rq.height >> src >> dst >> spec;
                if (!ls || spec.empty())
                    throw std::invalid_argument("UPS <width> <height> <in> <out> <name>[:<path>] [COLOR:<m>]");
                for (std::string word; ls >> word;) {
                    if (word.rfind("COLOR:", 0) == 0) {
                        rq.color = su::parse_color(word.substr(6));
                        rq.color_set = true;
                    }
                    else if (word.rfind("SEED:", 0) == 0) rq.seed = std::stoull(word.substr(5));
                    else if (word.rfind("TILE:", 0) == 0) rq.max_tile = std::stoll(word.substr(5));
                    else throw std::invalid_argument("UPS: unknown word " + word);
                }
                su::UpscaleModel& m = ups.get(spec);
                const si::QiRef pic = read_image(src);
                std::printf("STAGE upscale\n");
                std::fflush(stdout);
                const auto t0 = std::chrono::steady_clock::now();
                su::Report rep;
                const si::QiRef big = run_upscale(p, m, pic, rq, rep);
                const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                write_file(dst, big.rgba);
                const std::string parts = m.parts();
                std::printf("UPSCALED width=%lld height=%lld upscale_s=%.3f model_s=%.3f resize_s=%.3f color_s=%.3f "
                            "color=%s tile=%lld tiles=%lld%s%s\n", (long long) big.width, (long long) big.height, secs,
                            rep.model_s, rep.resize_s, rep.color_s, rep.color_applied ? su::color_name(rep.color) : "none",
                            (long long) rep.tile, (long long) rep.tiles, parts.empty() ? "" : " ", parts.c_str());
            } else if (op == "UPSV") {   // UPSV <width> <height> <in> <out stem> <name>[:<path>] [words]
                su::VideoRequest vr;
                std::string src, stem, spec;
                ls >> vr.frame.width >> vr.frame.height >> src >> stem >> spec;
                if (!ls || spec.empty())
                    throw std::invalid_argument("UPSV <width> <height> <in> <out stem> <name>[:<path>] [COLOR:<m>]");
                for (std::string word; ls >> word;) {
                    if (word.rfind("COLOR:", 0) == 0) {
                        vr.frame.color = su::parse_color(word.substr(6));
                        vr.frame.color_set = true;
                    } else if (word.rfind("SEED:", 0) == 0) {
                        vr.frame.seed = std::stoull(word.substr(5));
                    } else if (word.rfind("CHUNK:", 0) == 0) {
                        vr.chunk = std::stoll(word.substr(6));
                    } else if (word.rfind("OVERLAP:", 0) == 0) {
                        vr.overlap = std::stoll(word.substr(8));
                    } else if (word == "YIELD:1") {   // at its chunks' ends: YIELD, then on at RESUME
                        vr.pause = [&in] {
                            std::printf("YIELD\n");
                            std::fflush(stdout);
                            in.resume();
                        };
                    } else if (word.rfind("CODEC:", 0) == 0) {
                        const std::string c = word.substr(6);
                        vr.codec = c == "h264" ? strata::video::Codec::h264 : c == "vp9" ? strata::video::Codec::vp9
                                   : c == "ffv1" ? strata::video::Codec::ffv1
                                                 : throw std::invalid_argument("UPSV: CODEC: h264, vp9 or ffv1");
                    } else {
                        throw std::invalid_argument("UPSV: unknown word " + word);
                    }
                }
                su::UpscaleModel& m = ups.get(spec);
                p.release(true);   // a video's run is long: the image model's room its, and what the other upscalers
                {                  // and the detectors hold, the VRAM they give back added to the budget (aux_room's)
                    const uint64_t before = p.held() + si::free_vram();
                    ups.release_except(m);
                    detectors.release(true);
                    control.release();
                    const uint64_t after = p.held() + si::free_vram();
                    if (after > before) p.set_budget(p.budget() + (after - before));
                }
                const uint64_t avail = ups_room(p, m);
                std::printf("STAGE upscale-video\n");
                std::fflush(stdout);
                const auto t0 = std::chrono::steady_clock::now();
                const auto step = [&in](int64_t d, int64_t n) {
                    std::printf("STEP %lld %lld\n", (long long) d, (long long) n);
                    std::fflush(stdout);
                    return !in.stop().load();
                };
                // the room it was given against the most the process held through the run (the engine's log)
                size_t free0 = 0, total0 = 0;
                strata::gpu::mem_info(&free0, &total0);
                const size_t held0 = strata::gpu::device_held();
                const uint64_t ups0 = m.held();
                strata::gpu::device_peak_reset();
                const auto told_peak = [&](const char* how) {
                    std::fprintf(stderr, "xestrata-image: video upscale %s: room %llu MiB (budget %llu, the image "
                                 "model's %llu, free %llu, the upscaler's %llu at the start), the process held %llu "
                                 "MiB then and %llu at most\n", how, (unsigned long long) (avail >> 20),
                                 (unsigned long long) (p.budget() >> 20), (unsigned long long) (p.held() >> 20),
                                 (unsigned long long) (free0 >> 20), (unsigned long long) (ups0 >> 20),
                                 (unsigned long long) (held0 >> 20),
                                 (unsigned long long) (strata::gpu::device_peak_held() >> 20));
                };
                su::VideoReport vp;
                try {
                    vp = su::UpscaleRunner::run_video(m, src, stem, vr, avail, step);
                } catch (...) {
                    told_peak("failed");
                    throw;
                }
                told_peak("done");
                const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                const std::string parts = m.parts();
                std::printf("UPSVIDEO path=%s mime=%s frames=%lld width=%lld height=%lld fps=%.6f vfr=%d "
                            "upscale_s=%.3f read_s=%.3f model_s=%.3f resize_s=%.3f color_s=%.3f write_s=%.3f "
                            "color=%s%s%s\n",
                            vp.path.c_str(), vp.mime.c_str(), (long long) vp.frames, (long long) vp.width,
                            (long long) vp.height, vp.fps, (int) vp.vfr, secs, vp.read_s, vp.model_s, vp.resize_s,
                            vp.color_s, vp.write_s, su::color_name(vp.color), parts.empty() ? "" : " ", parts.c_str());
            } else if (op == "UPSVNEED") {   // UPSVNEED <width> <height> <in width> <in height> <name>[:<path>]
                int64_t w = 0, h = 0, iw = 0, ih = 0;
                std::string spec, word;
                ls >> w >> h >> iw >> ih >> spec;
                if (!ls || spec.empty())
                    throw std::invalid_argument("UPSVNEED <width> <height> <in width> <in height> <name>[:<path>]");
                su::UpscaleModel& m = ups.get(spec);
                su::ColorFix color = m.info().color;
                if (ls >> word && word.rfind("COLOR:", 0) == 0) color = su::parse_color(word.substr(6));
                size_t free = 0, total = 0;
                strata::gpu::mem_info(&free, &total);
                const int64_t mul = std::max<int64_t>(1, m.info().multiple);
                const int64_t fw = (w + mul - 1) / mul * mul, fh = (h + mul - 1) / mul * mul;
                const uint64_t vleast = m.video_bytes(fw, fh, 1, true);   // UINT64_MAX: the RAM too short
                const uint64_t least = vleast == UINT64_MAX
                                           ? UINT64_MAX
                                           : su::UpscaleRunner::least_bytes(m, iw, ih, w, h, color) + vleast;
                const uint64_t full =
                    su::UpscaleRunner::full_bytes(m, iw, ih, w, h, color) + m.video_bytes(fw, fh, 21, false);
                std::printf("NEED least_mib=%llu full_mib=%llu held_mib=%llu free_mib=%llu cap_mib=%llu host_mib=0%s\n",
                            (unsigned long long) (least >> 20), (unsigned long long) (full >> 20),
                            (unsigned long long) ((p.held() + ups.held()) >> 20), (unsigned long long) (free >> 20),
                            (unsigned long long) (total >> 20), vleast == UINT64_MAX ? " ram_short=1" : "");
            } else if (op == "UPSNEED") {   // UPSNEED <width> <height> <in width> <in height> <name>[:<path>] [COLOR:]
                int64_t w = 0, h = 0, iw = 0, ih = 0;
                std::string spec, word;
                ls >> w >> h >> iw >> ih >> spec;
                if (!ls || spec.empty())
                    throw std::invalid_argument("UPSNEED <width> <height> <in width> <in height> <name>[:<path>]");
                su::UpscaleModel& m = ups.get(spec);
                su::ColorFix color = m.info().color;
                if (ls >> word && word.rfind("COLOR:", 0) == 0) color = su::parse_color(word.substr(6));
                size_t free = 0, total = 0;
                strata::gpu::mem_info(&free, &total);
                const uint64_t least = su::UpscaleRunner::least_bytes(m, iw, ih, w, h, color);
                const uint64_t full = su::UpscaleRunner::full_bytes(m, iw, ih, w, h, color);
                // cap: the GPU's whole memory (what the upscaler could have with every other program's let go)
                std::printf("NEED least_mib=%llu full_mib=%llu held_mib=%llu free_mib=%llu cap_mib=%llu host_mib=0\n",
                            (unsigned long long) (least >> 20), (unsigned long long) (full >> 20),
                            (unsigned long long) ((p.held() + ups.held()) >> 20), (unsigned long long) (free >> 20),
                            (unsigned long long) (total >> 20));
            } else if (op == "DETECT") {   // DETECT <weights> <picture> <threshold> [<masks>]
                std::string weights, pic, masks;
                float conf = 0.5f;
                ls >> weights >> pic >> conf;
                if (!ls) throw std::invalid_argument("DETECT <weights> <picture> <threshold> [<masks>]");
                ls >> masks;
                std::printf("%s\n", detect_line(detectors.get(weights), read_image(pic), conf, masks).c_str());
            } else if (op == "SEGMENT") {   // SEGMENT <weights> <picture> <ids> <score> <mask> <masks>
                std::string weights, pic, ids, masks;
                float score = 0.5f, mask = 0.5f;
                ls >> weights >> pic >> ids >> score >> mask >> masks;
                if (!ls) throw std::invalid_argument("SEGMENT <weights> <picture> <ids> <score> <mask> <masks>");
                const auto t0 = std::chrono::steady_clock::now();
                si::Sam3& m = aux.get<si::Sam3>(weights);
                const si::QiRef img = read_image(pic);
                aux_room(p, m.need_bytes(img.width, img.height), "SAM 3", &aux, &m, &ups);
                const AuxTold told(m);
                si::Sam3::Times t;
                std::vector<si::Segment> segs;
                try {
                    segs = m.run(img, parse_ids(ids), score, mask, &t);
                } catch (...) {
                    aux_done(p, m);
                    throw;
                }
                aux_done(p, m);
                std::string line = "SEGMENT n=" + std::to_string(segs.size());
                char b[200];
                std::snprintf(b, sizeof b, " segment_s=%.4f vision_s=%.4f text_s=%.4f detect_s=%.4f masks_s=%.4f boxes=",
                              seconds_since(t0), t.vision, t.text, t.detect, t.masks);
                line += b;
                for (size_t i = 0; i < segs.size(); ++i) {
                    const auto& s = segs[i];
                    std::snprintf(b, sizeof b, "%s%.2f,%.2f,%.2f,%.2f,%.4f", i ? ";" : "", s.x0, s.y0, s.x1, s.y1,
                                  s.score);
                    line += b;
                    write_file(masks + "." + std::to_string(i), s.mask);
                }
                std::printf("%s\n", line.c_str());
            } else if (op == "COLORMAP") {   // COLORMAP <weights> <picture> <out> <ids;ids;...> [<score> <mask>]
                std::string weights, pic, out, words;
                ls >> weights >> pic >> out >> words;
                if (!ls) throw std::invalid_argument("COLORMAP <weights> <picture> <out> <ids;ids;...> [<score> <mask>]");
                float score = 0.5f, mask = 0.5f;
                ls >> score >> mask;
                std::vector<std::vector<int32_t>> ids;
                for (size_t a = 0; a <= words.size();) {
                    const size_t b = std::min(words.find(';', a), words.size());
                    if (b > a) ids.push_back(parse_ids(words.substr(a, b - a)));
                    a = b + 1;
                }
                const auto t0 = std::chrono::steady_clock::now();
                si::Sam3& m = aux.get<si::Sam3>(decoded_path(weights));
                const si::QiRef img = read_image(pic);
                aux_room(p, m.need_bytes(img.width, img.height), "SAM 3", &aux, &m, &ups);
                const AuxTold told(m);
                std::vector<std::vector<si::Segment>> found;
                try {
                    found = m.run_many(img, ids, score, mask);
                } catch (...) {
                    aux_done(p, m);
                    throw;
                }
                aux_done(p, m);
                const std::vector<uint8_t> map = color_map(found, img.width, img.height);
                write_file(out, map);
                size_t n = 0;
                for (const auto& f : found) n += f.size();
                std::printf("COLORMAP width=%lld height=%lld objects=%zu colormap_s=%.4f\n", (long long) img.width,
                            (long long) img.height, n, seconds_since(t0));
            } else if (op == "SEGV") {   // SEGV <weights> <video> <prompts> <out stem> [words]
                std::string weights, src, prompts, stem;
                ls >> weights >> src >> prompts >> stem;
                if (!ls) throw std::invalid_argument("SEGV <weights> <video> <prompts> <out stem> [MASKS:1] ...");
                SegvOptions o;
                for (std::string word; ls >> word;) {
                    if (word == "MASKS:1") o.masks = true;
                    else if (word == "MASK:1") o.mask = true;
                    else if (word == "OVERLAY:1") o.overlay = true;
                    else if (word == "YIELD:1") {
                        o.pause = [&in] {
                            std::printf("YIELD\n");
                            std::fflush(stdout);
                            in.resume();
                        };
                    } else if (word.rfind("CODEC:", 0) == 0) {
                        const std::string c = word.substr(6);
                        o.codec = c == "h264" ? strata::video::Codec::h264 : c == "vp9" ? strata::video::Codec::vp9
                                  : c == "ffv1" ? strata::video::Codec::ffv1
                                                : throw std::invalid_argument("SEGV: CODEC: h264, vp9 or ffv1");
                    } else {
                        throw std::invalid_argument("SEGV: unknown word " + word);
                    }
                }
                si::Sam3& m = aux.get<si::Sam3>(weights);
                const strata::video::Info info = strata::video::probe(src);
                aux_room(p, m.video_need_bytes(info.width, info.height, 1), "SAM 3.1's video segmentation", &aux,
                         &m, &ups);
                std::printf("STAGE segment-video\n");
                std::fflush(stdout);
                o.step = [&in](int64_t d, int64_t n) {
                    std::printf("STEP %lld %lld\n", (long long) d, (long long) n);
                    std::fflush(stdout);
                    return !in.stop().load();
                };
                std::string line;
                try {
                    line = segment_video(m, src, prompts, stem, o);
                } catch (...) {
                    aux_done(p, m);
                    throw;
                }
                aux_done(p, m);
                std::printf("%s\n", line.c_str());
            } else if (op == "FLORENCE") {   // FLORENCE <weights> <picture> <ids> <beams> <max new>
                std::string weights, pic, ids;
                int beams = 3, max_new = 1024;
                ls >> weights >> pic >> ids >> beams >> max_new;
                if (!ls || beams < 1 || max_new < 1)
                    throw std::invalid_argument("FLORENCE <weights> <picture> <ids> <beams> <max new>");
                const auto t0 = std::chrono::steady_clock::now();
                si::Florence2& m = aux.get<si::Florence2>(weights);
                const si::QiRef img = read_image(pic);
                aux_room(p, m.need_bytes(beams, max_new), "Florence-2", &aux, &m, &ups);
                const AuxTold told(m);
                si::Florence2::Times t;
                std::vector<int32_t> got;
                try {
                    got = m.run(img, parse_ids(ids), max_new, beams, &t);
                } catch (...) {
                    aux_done(p, m);
                    throw;
                }
                aux_done(p, m);
                std::string out;
                for (size_t i = 0; i < got.size(); ++i) out += (i ? "," : "") + std::to_string(got[i]);
                std::printf("FLORENCE florence_s=%.4f vision_s=%.4f encoder_s=%.4f decoder_s=%.4f steps=%d ids=%s\n",
                            seconds_since(t0), t.vision, t.encoder, t.decoder, t.steps, out.c_str());
            } else if (op == "POLYMASK") {   // POLYMASK <width> <height> <out> <x,y,...>...
                int64_t w = 0, h = 0;
                std::string out;
                ls >> w >> h >> out;
                if (!ls || w <= 0 || h <= 0) throw std::invalid_argument("POLYMASK <width> <height> <out> <x,y,...>...");
                std::vector<std::vector<float>> polys;
                for (std::string word; ls >> word;) {
                    std::vector<float> v;
                    std::stringstream ws(word);
                    for (std::string x; std::getline(ws, x, ',');) v.push_back(std::stof(x));
                    if (v.size() < 6 || v.size() % 2 != 0) throw std::invalid_argument("POLYMASK: a polygon of x,y pairs");
                    polys.push_back(std::move(v));
                }
                const si::QiRef m = si::Florence2::polygon_mask(polys, w, h);
                std::vector<uint8_t> raw((size_t) (w * h));
                for (size_t i = 0; i < raw.size(); ++i) raw[i] = (uint8_t) (255 - m.rgba[i * 4 + 3]);
                write_file(out, raw);
                std::printf("POLYMASK ok\n");
            } else if (op == "MATTE") {   // MATTE <weights> <picture> <model width> <model height> <out> [REFINE]
                std::string weights, pic, out, word;
                int64_t sw = 0, sh = 0;
                ls >> weights >> pic >> sw >> sh >> out;
                if (!ls) throw std::invalid_argument("MATTE <weights> <picture> <model width> <model height> <out> [REFINE]");
                const bool refine = (ls >> word) && word == "REFINE";
                const auto t0 = std::chrono::steady_clock::now();
                si::Birefnet& m = aux.get<si::Birefnet>(weights);
                const si::QiRef img = read_image(pic);
                aux_room(p, m.need_bytes(sw, sh, img.width, img.height, refine), "BiRefNet", &aux, &m, &ups);
                const AuxTold told(m);
                si::Birefnet::Times t;
                double refine_s = 0.0;
                try {
                    const auto alpha = m.run(img, sw, sh, &t);
                    write_file(out, alpha);
                    if (refine) {
                        const auto t1 = std::chrono::steady_clock::now();
                        write_file(out + ".fg", m.foreground(img, alpha));
                        refine_s = seconds_since(t1);
                    }
                } catch (...) {
                    aux_done(p, m);
                    throw;
                }
                aux_done(p, m);
                std::printf("MATTE matte_s=%.4f backbone_s=%.4f decoder_s=%.4f mask_s=%.4f refine_s=%.4f\n",
                            seconds_since(t0), t.backbone, t.decoder, t.mask, refine_s);
            } else if (op == "AUXNEED") {   // AUXNEED <kind> <weights> <width> <height> [...]
                std::string kind, weights;
                int64_t w = 0, h = 0;
                ls >> kind >> weights >> w >> h;
                if (!ls) throw std::invalid_argument("AUXNEED <sam3|sam3video|florence|birefnet|control> <weights> <width> <height> ...");
                uint64_t need = 0, held = 0;
                if (kind == "sam3") {
                    si::Sam3& m = aux.get<si::Sam3>(weights);
                    need = m.need_bytes(w, h), held = m.held();
                } else if (kind == "sam3video") {
                    int64_t buckets = 1;
                    ls >> buckets;
                    si::Sam3& m = aux.get<si::Sam3>(weights);
                    need = m.video_need_bytes(w, h, buckets), held = m.held();
                } else if (kind == "florence") {
                    int beams = 3, max_new = 1024;
                    ls >> beams >> max_new;
                    si::Florence2& m = aux.get<si::Florence2>(weights);
                    need = m.need_bytes(beams, max_new), held = m.held();
                } else if (kind == "birefnet") {
                    int64_t sw = 1024, sh = 1024;
                    int refine = 0;
                    ls >> sw >> sh >> refine;
                    si::Birefnet& m = aux.get<si::Birefnet>(weights);
                    need = m.need_bytes(sw, sh, w, h, refine != 0), held = m.held();
                } else if (kind == "control") {   // AUXNEED control <weights or -> <w> <h> <kind> <res> [<weights2>]
                    std::string ck, w2;
                    int res = 512;
                    ls >> ck >> res;
                    ls >> w2;
                    si::ControlParams cp;
                    cp.resolution = res;
                    si::ControlNets nets;
                    std::vector<si::AuxNet*> used;
                    need = control_plan(aux, ck, w, h, cp, weights == "-" ? "" : decoded_path(weights),
                                        w2.empty() ? "" : decoded_path(w2), nets, used);
                    for (si::AuxNet* m : used) held += m->held();
                } else {
                    throw std::invalid_argument("AUXNEED: sam3, sam3video, florence, birefnet or control, not " + kind);
                }
                size_t free = 0, total = 0;
                strata::gpu::mem_info(&free, &total);
                const uint64_t all = need + held;   // its weights and work, what it holds now counted in
                std::printf("NEED least_mib=%llu full_mib=%llu held_mib=%llu free_mib=%llu cap_mib=%llu host_mib=0\n",
                            (unsigned long long) (all >> 20), (unsigned long long) (all >> 20),
                            (unsigned long long) ((p.held() + held) >> 20), (unsigned long long) (free >> 20),
                            (unsigned long long) (total >> 20));
            } else if (op == "CONTROL") {   // CONTROL <kind> <picture> <out> [words]
                std::string kind, pic, out;
                ls >> kind >> pic >> out;
                if (!ls) throw std::invalid_argument("CONTROL <kind> <picture> <out> [RES:<n>]...");
                std::printf("%s\n", control_line(control, aux, &p, kind, read_image(pic), out, ls, false, &ups).c_str());
            } else if (op == "RATIO") {   // the size a request leaves to the model (serve/images.py)
                si::QiRequest r;
                std::string ids;
                ls >> r.drop >> ids;
                if (!ls) throw std::invalid_argument("RATIO <drop> <ids> [REF:<path>]...");
                r.ids = parse_ids(ids);
                r.refs = read_refs(ls, r, loras);
                double secs = 0.0;
                const int i = p.choose_ratio(r, &secs);
                std::printf("RATIO index=%d ratio_s=%.3f\n", i, secs);
            } else if (op == "NEED") {
                int64_t w = 0, h = 0, n = 0;
                ls >> w >> h >> n;
                if (!ls) throw std::invalid_argument("NEED <width> <height> <n> [REF:<path>]... [MASK:<path>]");
                bool mask = false, init = false, states = false;
                int64_t n_neg = 0;
                std::optional<si::CropBox> crop;
                std::vector<std::pair<int64_t, int64_t>> pictures;
                si::QiRequest lr;   // the request's adapters, which the need counts and nothing else uses
                auto sizes = ref_sizes(ls, mask, init, n_neg, states, crop, pictures, lr, loras);
                // the cut: the first reference (the base image without an init image) is drawn from at the
                // line's size, and the steps' buffers come before the drawing and after it, beside what is held
                if (crop && !init && !sizes.empty()) sizes[0] = {w, h};
                const int maps = (lr.mctl_color.width > 0 ? 1 : 0) + (lr.mctl_depth.width > 0 ? 1 : 0);
                if ((maps > 0) != !lr.mctl.empty() || (maps > 0 && !p.takes_moss_options()))
                    throw std::invalid_argument("NEED: MCTL and at least one MCOLOR or MDEPTH are MOSS-Image's");
                p.set_need_context(states, pictures, lr.refs_layout);
                p.set_need_control(maps);
                si::QiNeed nd;
                try { nd = p.need(n, w, h, sizes, mask, n_neg, lr.loras); }
                catch (...) {
                    p.set_need_control(0);
                    p.set_need_context(false, {}, si::QiRefLayout::MossEdit);
                    throw;
                }
                p.set_need_control(0);
                p.set_need_context(false, {}, si::QiRefLayout::MossEdit);
                if (!lr.upscaler.empty()) {   // the upscaler runs before the drawing, the image model's cache given back
                    su::UpscaleModel& m = ups.get(lr.upscaler);
                    const int64_t iw = lr.init.width > 0 ? lr.init.width : w, ih = lr.init.height > 0 ? lr.init.height : h;
                    const int64_t ow = lr.upscale ? lr.width : w, oh = lr.upscale ? lr.height : h;
                    const su::ColorFix c = lr.upscaler_color.empty() ? m.info().color : su::parse_color(lr.upscaler_color);
                    // its weights stay beside the drawing's where upscale_base keeps them (RealESRGAN's); SeedVR2's,
                    // past kUpsReserve, are let go before the drawing
                    const uint64_t kept = m.weight_bytes() > kUpsReserve ? 0 : m.weight_bytes();
                    nd.least = std::max(nd.least + kept, su::UpscaleRunner::least_bytes(m, iw, ih, ow, oh, c));
                    nd.full = std::max(nd.full + kept, nd.held + su::UpscaleRunner::full_bytes(m, iw, ih, ow, oh, c));
                }
                if (!lr.control_net.empty()) {
                    const auto [least, full] = p.control_need(lr.control_net, w, h);
                    nd.least += least;
                    nd.full += full;
                }
                if (crop) {
                    const uint64_t cb = si::CropStitch::bytes(crop->width, crop->height, crop->w, crop->h, w, h,
                                                              crop->ref_w, crop->ref_h);
                    nd.least += cb;
                    nd.full += cb;
                }
                std::printf("NEED least_mib=%llu full_mib=%llu held_mib=%llu free_mib=%llu cap_mib=%llu host_mib=%llu\n",
                            (unsigned long long) (nd.least >> 20), (unsigned long long) (nd.full >> 20),
                            (unsigned long long) (nd.held >> 20), (unsigned long long) (nd.free >> 20),
                            (unsigned long long) (nd.cap >> 20), (unsigned long long) (nd.host >> 20));
            } else if (op == "FREE") {
                // held_mib: all the engine holds, the image model's and the tools' models' kept between requests (the
                // server's VRAM ledger enters it); a detection's scratch and the control steps' buffers are not counted
                size_t free = 0, total = 0;
                strata::gpu::mem_info(&free, &total);
                const uint64_t held = p.held() + ups.held() + aux.held() + detectors.held();
                std::printf("FREE free_mib=%llu held_mib=%llu total_mib=%llu\n", (unsigned long long) (free >> 20),
                            (unsigned long long) (held >> 20), (unsigned long long) (total >> 20));
            } else if (op == "W8") {
                std::printf("W8 take=%d\n", si::w8_here(nullptr));
            } else if (op == "BUDGET") {
                uint64_t mib = 0;
                ls >> mib;
                p.set_budget(mib << 20);
                std::printf("BUDGET budget_mib=%llu\n", (unsigned long long) (p.budget() >> 20));
            } else if (op == "FIT") {
                uint64_t mib = 0;
                ls >> mib;
                p.fit(mib << 20);
                std::printf("BUDGET budget_mib=%llu\n", (unsigned long long) (p.budget() >> 20));
            } else if (op == "RELEASE") {
                std::string what;
                ls >> what;
                p.release(what == "ALL");
                detectors.release(what == "ALL");
                ups.release(what == "ALL");
                aux.release(what == "ALL");
                control.release();
                std::printf("RELEASE ok\n");
            } else if (!op.empty()) {
                throw std::invalid_argument("unknown request " + op);
            }
        } catch (const si::QiStopped&) {
            std::printf("ERROR stopped\n");
        } catch (const std::exception& e) {
            if (op == "IMG") {   // a request that failed (short of memory) gives its cache back
                try {
                    p.release(false);
                } catch (const std::exception& e2) {
                    std::fprintf(stderr, "xestrata-image: release after a failed request: %s\n", e2.what());
                }
            }
            std::string m = e.what();
            const bool auxiliary = op == "MATTE" || op == "SEGMENT" || op == "SEGV" || op == "FLORENCE" ||
                                   op == "CONTROL" || op == "DETECT" || op == "COLORMAP";
            const bool refused = m.find("alloc_device:") != std::string::npos && m.find("refused") != std::string::npos;
            // a video's upscale too: refused past its plan (another program took the VRAM meanwhile), it starts again
            // once with every weight let go, its file written anew
            if ((auxiliary || op == "UPSV") && refused && !retried_aux) {
                std::fprintf(stderr, "xestrata-image: %s allocation refused, releasing resident weights: %s\n",
                             op == "UPSV" ? "a video's" : "auxiliary", m.c_str());
                p.release(true);
                detectors.release(true);
                ups.release(true);
                aux.release(true);
                control.release();
                retried_aux = true;
                goto retry_aux;
            }
            if ((auxiliary || op == "UPSV") && refused)
                m = "memory: the GPU refused an allocation; free GPU memory and try again";
            for (char& c : m)
                if (c == '\n') c = ' ';
            // an adapter the request names and the engine does not take is the request's fault (the server's 400)
            const bool lora = dynamic_cast<const si::LoraRejected*>(&e) != nullptr;
            std::printf("ERROR %s%s\n", lora ? "lora: " : "", m.c_str());
        }
        std::fflush(stdout);
    }
    return 0;
}

// --upscale: an upscaler alone, no image model (tools, checks and timings)
int upscale_main(int argc, char** argv) {
    std::string in, out, name, model;
    su::Request rq;
    double scale = 0.0;
    uint64_t budget = 0;
    int repeat = 1;
    bool video = false;
    int64_t chunk = 0, overlap = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return argv[++i];
        };
        if (a == "--upscale") in = val();
        else if (a == "--video") video = true;
        else if (a == "--chunk") chunk = std::stoll(val());
        else if (a == "--overlap") overlap = std::stoll(val());
        else if (a == "--upscaler") name = val();
        else if (a == "--model") model = val();
        else if (a == "--out") out = val();
        else if (a == "--width") rq.width = std::stoll(val());
        else if (a == "--height") rq.height = std::stoll(val());
        else if (a == "--scale") scale = std::stod(val());
        else if (a == "--color") {
            rq.color = su::parse_color(val());
            rq.color_set = true;
        }
        else if (a == "--tile") rq.max_tile = std::stoll(val());
        else if (a == "--seed") rq.seed = std::stoull(val());
        else if (a == "--budget-mib") budget = std::stoull(val()) << 20;
        else if (a == "--repeat") repeat = std::stoi(val());
        else throw std::invalid_argument("unknown option " + a);
    }
    if (in.empty() || out.empty() || name.empty()) {
        std::fprintf(stderr, "usage: %s --upscale IMAGE --upscaler lanczos|realesrgan [--model WEIGHTS] --out "
                     "IMAGE.rgba [--width W --height H | --scale S] [--color none|lab|adain|wavelet] [--tile N] "
                     "[--budget-mib M] [--repeat N] [--video [--chunk FRAMES] [--overlap LATENTS]]\n"
                     "  --video: IMAGE a video and --out its file's stem (the container's extension added)\n", argv[0]);
        return 2;
    }
    const std::unique_ptr<su::UpscaleModel> m = su::open_model(name, model);
    if (video) {
        const strata::video::Info info = strata::video::probe(in);
        su::VideoRequest vr;
        vr.frame = rq;
        vr.chunk = chunk;
        vr.overlap = overlap;
        if (vr.frame.width <= 0 || vr.frame.height <= 0) {
            const double s = scale > 0.0 ? scale : (m->info().scale > 0 ? m->info().scale : 2.0);
            vr.frame.width = (int64_t) std::lround((double) info.width * s);
            vr.frame.height = (int64_t) std::lround((double) info.height * s);
        }
        size_t free = 0, total = 0;
        strata::gpu::mem_info(&free, &total);
        const uint64_t room = budget > 0 ? budget : (uint64_t) free;
        const auto t0 = std::chrono::steady_clock::now();
        const su::VideoReport vp = su::UpscaleRunner::run_video(*m, in, out, vr, room, [](int64_t d, int64_t n) {
            std::fprintf(stderr, "\rframe %lld / %lld", (long long) d, (long long) n);
            return true;
        });
        std::fprintf(stderr, "\n");
        std::printf("UPSVIDEO path=%s frames=%lld width=%lld height=%lld fps=%.3f vfr=%d upscale_s=%.3f read_s=%.3f "
                    "model_s=%.3f resize_s=%.3f color_s=%.3f write_s=%.3f color=%s %s\n",
                    vp.path.c_str(), (long long) vp.frames, (long long) vp.width, (long long) vp.height, vp.fps,
                    (int) vp.vfr, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                    vp.read_s, vp.model_s, vp.resize_s, vp.color_s, vp.write_s, su::color_name(vp.color),
                    m->parts().c_str());
        return 0;
    }
    const si::QiRef src = read_image(in);
    if (rq.width <= 0 || rq.height <= 0) {
        const double s = scale > 0.0 ? scale : (m->info().scale > 0 ? m->info().scale : 2.0);
        rq.width = (int64_t) std::lround((double) src.width * s);
        rq.height = (int64_t) std::lround((double) src.height * s);
    }
    for (int k = 0; k < std::max(1, repeat); ++k) {
        size_t free = 0, total = 0;
        strata::gpu::mem_info(&free, &total);
        const uint64_t room = (uint64_t) free + m->held();
        const uint64_t avail = budget > 0 ? std::min<uint64_t>(budget, room) : room;
        const auto t0 = std::chrono::steady_clock::now();
        su::Report rep;
        const si::QiRef big = su::UpscaleRunner::run(*m, src, rq, avail, &rep);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        write_file(out, big.rgba);
        const std::string parts = m->parts();
        std::printf("UPSCALED width=%lld height=%lld upscale_s=%.3f model_s=%.3f resize_s=%.3f color_s=%.3f color=%s "
                    "tile=%lld tiles=%lld%s%s\n", (long long) big.width, (long long) big.height, secs, rep.model_s,
                    rep.resize_s, rep.color_s, rep.color_applied ? su::color_name(rep.color) : "none",
                    (long long) rep.tile, (long long) rep.tiles, parts.empty() ? "" : " ", parts.c_str());
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    strata::core::route_stop_signals();   // a stop signal ends the engine after its GPU queues are drained
    // oneMath's rocBLAS backend times hipBLASLt's first algorithms on a shape's first product and keeps the fastest:
    // an image model's few shapes repeat every step (RX 9060 XT: its int8 products 20 to 40% faster), where the text
    // engine's change with each prompt (it leaves the timing off)
    if (std::getenv("ONEMATH_HIPBLASLT_TUNE") == nullptr) {
#if defined(_WIN32)
        _putenv_s("ONEMATH_HIPBLASLT_TUNE", "1");
#else
        setenv("ONEMATH_HIPBLASLT_TUNE", "1", 0);
#endif
    }
    try {
        if (argc >= 2 && std::string(argv[1]) == "--upscale") return upscale_main(argc, argv);
        // --detect <weights> <picture> [<threshold> [<masks>]]: the detector alone, no image model (tools and checks)
        if (argc >= 4 && std::string(argv[1]) == "--detect") {
            Detectors d;
            si::YoloDetector& det = d.get(argv[2]);
            const si::QiRef img = read_image(argv[3]);
            const float conf = argc >= 5 ? std::stof(argv[4]) : 0.5f;
            const char* rep = strata::env_get("DETECT_REPEAT");   // a timing's repeats
            const int repeat = rep != nullptr ? (int) std::strtol(rep, nullptr, 10) : 1;
            for (int i = 0; i < std::max(1, repeat); ++i)
                std::printf("%s\n", detect_line(det, img, conf, argc >= 6 ? argv[5] : "", i + 1 < repeat).c_str());
            return 0;
        }
        // --control <kind> <picture> <out> [words]: a control image alone, no image model (tools and checks)
        if (argc >= 5 && std::string(argv[1]) == "--control") {
            si::ControlPrep prep;
            AuxNets nets;
            const si::QiRef img = read_image(argv[3]);
            std::string words;
            for (int i = 5; i < argc; ++i) words.append(argv[i]).append(" ");
            const char* rep = strata::env_get("CONTROL_REPEAT");   // a timing's repeats
            const int repeat = rep != nullptr ? (int) std::strtol(rep, nullptr, 10) : 1;
            for (int i = 0; i < std::max(1, repeat); ++i) {
                std::istringstream ws(words);
                std::printf("%s\n", control_line(prep, nets, nullptr, argv[2], img, argv[4], ws, i + 1 < repeat).c_str());
            }
            return 0;
        }
        std::string dit, vae, text, vision, ratio_head, out, ids;
        std::vector<std::string> ref_paths;
        std::vector<int> block_map;   // --lora-block-map: an expanded Anima's release blocks
        bool lora_bake = true;        // --lora-bake 0: the adapters added to each product, not baked in
        Loras cli_loras;              // --lora's files
        si::QiRequest r;
        WordRead words(r, cli_loras); // the options that are the IMG line's words read through it (one reading)
        uint64_t budget = 0;
        int repeat = 1;
        std::string edit_dit;
        int dit_q8 = -1;   // Mage-Flow: by the budget
        bool serving = false, dit_keep = true, ratio = false;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto val = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
                return argv[++i];
            };
            if (a == "--dit") dit = val();
            else if (a == "--vae") vae = val();
            else if (a == "--text") text = val();
            else if (a == "--vision") vision = val();
            else if (a == "--ratio-head") ratio_head = val();
            else if (a == "--ratio") ratio = true;
            else if (a == "--ref") ref_paths.push_back(val());
            else if (a == "--out") out = val();
            else if (a == "--ids") ids = val();
            else if (a == "--drop") r.drop = std::stoll(val());
            else if (a == "--width") r.width = std::stoll(val());
            else if (a == "--height") r.height = std::stoll(val());
            else if (a == "--seed") r.seed = std::stoull(val());
            else if (a == "--lora-block-map")
                for (const int32_t b : parse_ids(val())) block_map.push_back((int) b);
            else if (a == "--lora-bake") lora_bake = std::stoi(val()) != 0;
            else if (a == "--budget-mib") budget = std::stoull(val()) << 20;
            else if (a == "--repeat") repeat = std::stoi(val());
            else if (a == "--serve") serving = true;
            else if (a == "--dit-q8") {
                const std::string v = val();
                dit_q8 = v == "auto" ? -1 : v != "0";
            }
            else if (a == "--edit-dit") edit_dit = val();
            else if (a == "--dit-keep") dit_keep = val() != "0";
            else {   // the rest are the IMG line's words: --steps 8 reads as the word STEPS:8, one reading of both
                std::string word;
                if (!cli_word(a, word) || !read_word(word + val(), words))
                    throw std::invalid_argument("unknown option " + a);
            }
        }
        // the model is in the DiT's file, not in its extension: both Mage-Flow and HunyuanImage 3.0 are
        // safetensors, and HunyuanImage 3.0's says so in its header (hunyuanimage3_is_dit_file reads the header
        // alone, no weights)
        const bool safetensors = dit.size() > 12 && dit.compare(dit.size() - 12, 12, ".safetensors") == 0;
        const bool hunyuanimage3 = safetensors && si::hunyuanimage3_is_dit_file(dit);
        const bool moss = !hunyuanimage3 && si::moss_is_dit(dit);
        const bool anima = safetensors && !hunyuanimage3 && !moss && si::anima_is_dit(dit);
        const bool mage = safetensors && !hunyuanimage3 && !moss && !anima;
        const bool need_text = !hunyuanimage3;
        if (dit.empty() || vae.empty() || (need_text && text.empty()) ||
            (!serving && ((out.empty() && !ratio) || (ids.empty() && r.te_rows == 0)))) {
            std::fprintf(stderr, "usage: %s --dit F --vae F --text F [--vision F] [--budget-mib M] (--ids 1,2,3 "
                         "[--drop N] [--width W] [--height H] [--seed S] [--sigmas 1,...,0] [--ref IMAGE]... "
                         "[--mask IMAGE] [--init IMAGE] [--strength F] [--repeat N] [--lora STRENGTH:FILE]... --out IMAGE.rgba | --serve) "
                         "[--dit-q8 auto|0|1] [--edit-dit F] "
                         "[--dit-keep 0|1]\n",
                         argv[0]);
            std::fprintf(stderr, "       %s: HunyuanImage 3.0 (a quantized .safetensors --dit) takes no --text "
                         "(its DiT reads the ids itself); [--steps N] [--guidance G] stand for [--sigmas] [--cfg]; "
                         "its edits (--ref) take --vision, the model's shard with vision_model.*; --ratio with "
                         "--ratio-head F prints the size it chooses (the ids ending with <boi> <img_size_N>) in "
                         "place of an image\n",
                         argv[0]);
            std::fprintf(stderr, "       %s: MOSS-Image (--dit its transformer folder, --text Qwen3.5-9B's GGUF) takes "
                         "[--neg-ids I] [--cfg C] [--steps N] [--shift S] [--uncond-skip B,...|none] [--slg-layers "
                         "B,...] [--slg-scale F] [--slg-start F] [--slg-stop F] [--te F] [--neg-te F] [--init IMAGE] "
                         "[--strength F] [--mask IMAGE]\n",
                         argv[0]);
            std::fprintf(stderr, "       %s: Anima (--dit its safetensors, --vae Wan 2.1's, --text Qwen3-0.6B-Base's "
                         "GGUF) takes --t5 I --t5-weights W [--neg-ids I --neg-t5 I --neg-t5-weights W] [--cfg C] "
                         "[--steps N] [--shift S] [--sampler er_sde|euler] [--init IMAGE] [--strength F] "
                         "[--mask IMAGE]\n",
                         argv[0]);
            return 2;
        }
        // the products' library loaded on a thread of its own while the model opens (as the text engines start), not
        // at the first image's first product
        strata::prefill::Gemm::prepare();
        std::unique_ptr<si::ImagePipeline> pipe;
        std::string err;
        if (hunyuanimage3) {
            auto h = std::make_unique<si::HunyuanImage3Pipeline>();
            if (!h->open(dit, vae, err, 0, vision, ratio_head)) throw std::runtime_error(err);
            pipe = std::move(h);
        } else if (moss) {
            auto m = std::make_unique<si::MossPipeline>();
            if (!m->open(dit, vae, text, err, dit_q8, vision)) throw std::runtime_error(err);
            pipe = std::move(m);
        } else if (anima) {
            auto a = std::make_unique<si::AnimaPipeline>();
            if (!a->open(dit, vae, text, err, dit_q8)) throw std::runtime_error(err);
            a->set_lora_block_map(block_map);
            pipe = std::move(a);
        } else if (mage) {
            auto m = std::make_unique<si::MfPipeline>();
            if (!m->open(dit, vae, text, err, dit_q8, edit_dit, vision, dit_keep)) throw std::runtime_error(err);
            pipe = std::move(m);
        } else {
            auto q = std::make_unique<si::QiPipeline>();
            if (!q->open(dit, vae, text, err, vision)) throw std::runtime_error(err);
            pipe = std::move(q);
        }
        si::ImagePipeline& p = *pipe;
        p.set_lora_bake(lora_bake);
        if (budget > 0) p.set_budget(budget);
        if (serving) return serve(p);
        r.ids = parse_ids(ids);
        for (const std::string& path : ref_paths) {
            si::QiRef f;
            f.rgba = si::load_rgba8(path, f.width, f.height, err);
            if (f.rgba.empty()) throw std::runtime_error(err);
            r.refs.push_back(std::move(f));
        }
        if (ratio) {
            double secs = 0.0;
            const int i = p.choose_ratio(r, &secs);
            std::printf("RATIO index=%d ratio_s=%.3f", i, secs);
            for (size_t k = 0; k < p.ratio_logits().size(); ++k)
                std::printf("%s%.6g", k == 0 ? " logits=" : ",", p.ratio_logits()[k]);
            std::printf("\n");
            return 0;
        }
        for (int k = 0; k < repeat; ++k) {
            si::QiTimes t;
            si::QiPlan pl;
            write_file(out, p.generate(r, &t, &pl));
            std::printf("%s\n", plan_kv(t, pl).c_str());
            std::fflush(stdout);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image: %s\n", e.what());
        return 1;
    }
    return 0;
}
