// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/lora_names.hpp - the names a LoRA file's keys carry for the weights a model adapts, one table a
// model.  The DiTs' build_lora_targets (qi_dit.cpp, moss_dit.cpp, mf_dit.cpp, anima_dit.cpp) and the trainers' file
// names (qi_trainer.cpp, moss_trainer.cpp, mf_trainer.cpp, anima_trainer.cpp) read the same table; a block's names go
// under the model's block path, and the trainer's products are the table's first entries, in its product order
// (QiTrainProduct, MossTrainProduct, AnimaTrainProduct; Mage-Flow's two streams side by side).
//
// serve/loras.py's TARGETS holds the same names for its checks of a file's keys; serve/test_image_lora.py checks the
// two against each other, table by table.
#pragma once

namespace strata::image::lora {

// Qwen-Image-2.1: the block's six products (the fifth the fused [gate | up]), the two names a file may carry for that
// weight's halves instead, and the weights outside the blocks.
inline constexpr const char* kQiBlock[6] = {"attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                                            "img_mlp.gate_up", "img_mlp.out"};
inline constexpr const char* kQiGateHalf[2] = {"img_mlp.gate_layer", "img_mlp.proj"};
inline constexpr const char* kQiOutside[8] = {"img_in", "time_text_embed.timestep_embedder.linear_1",
                                              "time_text_embed.timestep_embedder.linear_2", "modulation.1",
                                              "norm_out.linear", "proj_out", "txt_in.in_layer", "txt_in.out_layer"};

// MOSS-Image: the block's four products (one qkv of q, k and v; one [up | gate]), the names a file may carry for their
// parts instead, the text embedder's block (the same four), and the weights outside the blocks.
inline constexpr const char* kMossBlock[4] = {"proj.qkv", "proj.o", "proj.ffn_up_gate", "proj.ffn_down"};
inline constexpr const char* kMossQkv[4] = {"proj.q", "proj.k", "proj.v", "proj.qkv"};
inline constexpr const char* kMossUpGate[3] = {"proj.ffn_up_gate", "proj.ffn_up", "proj.ffn_gate"};
inline constexpr const char* kMossTextBlock[4] = {"qkv", "proj", "ffn_up_gate", "ffn_down"};
inline constexpr const char* kMossTextUpGate[3] = {"ffn_up_gate", "ffn_up", "ffn_gate"};
inline constexpr const char* kMossOutside[7] = {"video_embedder", "time_embedding.0", "time_embedding.2",
                                                "shared_block_modulation", "video_head_modulation",
                                                "final_linear_video", "text_embedder.proj"};

// Mage-Flow: the two streams' six products each, the blocks' two modulations, and the weights outside the blocks.
inline constexpr const char* kMageBlock[2][6] = {
    {"attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.net.0.proj", "img_mlp.net.2"},
    {"attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out", "txt_mlp.net.0.proj",
     "txt_mlp.net.2"}};
inline constexpr const char* kMageMod[2] = {"img_mod.1", "txt_mod.1"};
inline constexpr const char* kMageOutside[6] = {"img_in", "txt_in", "time_text_embed.timestep_embedder.linear_1",
                                                "time_text_embed.timestep_embedder.linear_2", "norm_out.linear",
                                                "proj_out"};

// Anima: the block's ten products, its six modulation rows (a name and ".1" or ".2"), the LLM adapter's ten products,
// and the weights outside the blocks.
inline constexpr const char* kAnimaBlock[10] = {"self_attn.q_proj",  "self_attn.k_proj",          "self_attn.v_proj",
                                                "self_attn.output_proj", "cross_attn.q_proj",     "cross_attn.k_proj",
                                                "cross_attn.v_proj", "cross_attn.output_proj",   "mlp.layer1",
                                                "mlp.layer2"};
inline constexpr const char* kAnimaMod[3] = {"adaln_modulation_self_attn", "adaln_modulation_cross_attn",
                                             "adaln_modulation_mlp"};
inline constexpr const char* kAnimaAdapter[10] = {"self_attn.q_proj", "self_attn.k_proj",   "self_attn.v_proj",
                                                  "self_attn.o_proj", "cross_attn.q_proj",  "cross_attn.k_proj",
                                                  "cross_attn.v_proj", "cross_attn.o_proj", "mlp.0",
                                                  "mlp.2"};
inline constexpr const char* kAnimaOutside[7] = {"x_embedder.proj.1", "t_embedder.1.linear_1", "t_embedder.1.linear_2",
                                                 "final_layer.linear", "final_layer.adaln_modulation.1",
                                                 "final_layer.adaln_modulation.2", "llm_adapter.out_proj"};

}  // namespace strata::image::lora
