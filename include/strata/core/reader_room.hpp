#pragma once
// The picture reader's room when the expert cache cannot give it (owner 11-11, item "picture reader on demand, no reserve up front").
//
// On one NVIDIA card the server starts the GPU picture reader with the first picture and asks the engine's expert cache to make the
// reader's room then (`VRAM <reserve>`, --vram-elastic); before the first picture nothing is held back beyond the reserve setup wrote
// (700 MiB).  In the resident RAM mode (the RAM copy of the experts the cache does not hold: --resident-experts, --resident-budget-gib,
// or the start's own fallback when RAM is short for the arena) the engine refuses that command ("VRAM: not with the resident low-RAM
// mode"), so the reader would find about 0.4 GiB free and go to the processor (measured on l620, IQ3_S on a card held to 32 GB:
// pictures 1.4-7.7 s on the processor against 0.4-2.1 s on the card).  The engine knows that mode before it sizes the cache, so there it
// keeps the reader's room in the reserve, as before the change: the server then finds it free and starts the reader on the card.
// The mode is decided once, at start (the arena / file pass loop); nothing enters it later.

namespace strata::core {

/// The reader's room, the same number as the server's VISION_RESERVE_MIB (serve/server.py) and the reader's elastic_mib.
inline constexpr int kReaderRoomMib = 2048;

/// The VRAM reserve the expert cache is sized with.  `vision`: the engine serves pictures (--vision); `vram_elastic`: the cache was
/// asked to give the reader its room on demand (--vram-elastic); `ram_copy_mode`: the resident RAM mode, in which the cache refuses
/// to resize.  Only all three together raise the reserve, and never above what was asked when that is larger.
inline int reader_room_reserve_mib(int reserve_mib, bool vision, bool vram_elastic, bool ram_copy_mode) {
    if (vision && vram_elastic && ram_copy_mode && reserve_mib < kReaderRoomMib) return kReaderRoomMib;
    return reserve_mib;
}

}  // namespace strata::core
