// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Model, base, size class, measured total and text rows, open, least-form min and process peak, whole estimate (MiB).
// Rank 32; the greatest B70 / RTX 4070 / RX 9060 XT observation. MageTrail and whole-form peaks are estimates.
TRAIN_VRAM("qwen-image-2.1", "int8", 768, 2337, 33, 2640, 6243, 6104, 15360)
TRAIN_VRAM("qwen-image-2.1", "bf16", 768, 2337, 33, 2087, 5852, 5714, 21504)
TRAIN_VRAM("qwen-image-2.1", "int8", 1024, 4065, 33, 2640, 7537, 7957, 15360)
TRAIN_VRAM("qwen-image-2.1", "bf16", 1024, 4065, 33, 2087, 7143, 7559, 21504)
TRAIN_VRAM("mage-flow", "int8", 768, 2269, 29, 6583, 6737, 6649, 10240)
TRAIN_VRAM("mage-flow", "bf16", 768, 2269, 29, 6604, 6717, 6639, 14848)
TRAIN_VRAM("mage-flow", "int8", 1024, 4024, 29, 6583, 8145, 8090, 10240)
TRAIN_VRAM("mage-flow", "bf16", 1024, 4024, 29, 6604, 8059, 8015, 14848)
TRAIN_VRAM("magetrail", "int8", 1024, 4024, 29, 2048, 8145, 10240, 10240)
TRAIN_VRAM("magetrail", "bf16", 1024, 4024, 29, 2048, 11264, 12288, 14848)
TRAIN_VRAM("moss-image", "int8", 768, 2274, 34, 1958, 5156, 5027, 12288)
TRAIN_VRAM("moss-image", "bf16", 768, 2274, 34, 1900, 4914, 4786, 16384)
TRAIN_VRAM("moss-image", "int8", 1024, 4029, 34, 1958, 7039, 6820, 12288)
TRAIN_VRAM("moss-image", "bf16", 1024, 4029, 34, 1900, 6863, 6645, 16384)
TRAIN_VRAM("anima", "bf16", 768, 2752, 512, 1998, 4204, 4122, 9216)
TRAIN_VRAM("anima", "bf16", 1024, 4507, 512, 1998, 5136, 4999, 9216)
