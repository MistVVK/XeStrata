"""What the entry points are ALLOWED to do differently, with the reason.  Everything else must be identical in every machine class x model set:
`strata launch`, the app's plain Load and the app's Models-screen Load start the engine with the same arguments and the same environment.

The golden files are the data; `test_entry_points_agree.py` applies this list to them.  A new difference has to be added here WITH a reason, which is the review."""

#: Run-config keys that differ between the Terminal and the app by design (app vs launch).
BY_DESIGN = {
    "api_key": "the app generates its own key; the Terminal asks for none unless --api-key",
    "aliases": "the app lists the API model names clients may use",
    "port": "the app's engine listens on 18500, the Terminal's on 8080 (or --port)",
    "fit_max_tokens": "the app asks the engine to shorten a requested answer to the room left in the context (the agent asks for long answers)",
    "mcp": "the app keeps its tool-server settings in the run config",
    "log": "the app writes the engine log into its install folder, the Terminal into the folder it was started in",
    "engine_ready_s": "how long the app waits for the engine to answer; the Terminal derives its own wait from the model's size",
    "args_derived": "launch's own bookkeeping stamp of the arguments it derived (a copy of args)",
    "args_written": "launch's own bookkeeping stamp of the arguments it wrote (a copy of args)",
}

#: Environment variables the engine start adds, which differ by design.
ENV_BY_DESIGN = {
}

#: Intended differences of the Models-screen Load alone.
MODELS_SCREEN = {
    "drafting": "the Models screen's Load saves the person's Drafting choice ('auto' unless they changed it) into the model's settings; launch and the plain Load save none",
}

#: The request-slot rule (CLEAN-1A item 1a.6, owner decision 11-07).  Not a difference to allow but a fact the golden files pin: with the plain Load following Auto, NVIDIA 72 GB
#: runs DFlash2, which answers one request at a time, so the app writes no `parallel` and no `--batch ... --batch-mtp` there, exactly as the Models-screen Load always did.
#: (On main the plain Load ran the MTP and wrote `parallel: 3`; that was a consequence of the missing drafter, not a policy.)  The app's own default of several request slots
#: (apps/host/server/native-parallel.mjs) applies to a run that is not DFlash2; no cell of the matrix needs it.
REQUEST_SLOTS = "no cell writes `parallel` from either app entry point"
