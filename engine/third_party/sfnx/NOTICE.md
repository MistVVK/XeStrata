# sFNX, the split-every-layer engine for Qwen3.8-Flash-Next

The files under `src/split/` come from https://github.com/johnfink8/sfnx (commit f9ba1e9), its engine (`src/`), the engine
process (`tools/sfnx.cpp`, `tools/sfnx_input.hpp`), the CPU unit tests it ships (`tools/*_test.cpp`) and two data files
(`data/draft_vocab_en.bin`, `data/expert-routes.bin`). sFNX splits every layer of the model across all the cards (experts by
feed-forward width, recurrent layers by value heads, attention K/V by position, the output head by vocabulary) instead of
giving each card some layers.

Copyright (c) 2026 John Fink. MIT License, the full text is in `LICENSE` beside this file. The notice stays with every copy.

sFNX's own front end (`front/`) was ported from Strata's `serve/` (MIT, Niko1221 and the Strata contributors). It is not part of
this copy: Strata's own front end drives the engine process.

Changes made while porting into Strata are listed in the git history of the `src/split` directory (the first commit of each
file is the unmodified upstream copy).
