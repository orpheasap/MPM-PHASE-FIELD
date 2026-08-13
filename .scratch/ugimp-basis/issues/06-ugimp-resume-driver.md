# 06 — uGIMP resume driver

**What to build:** A researcher can extend a uGIMP run in batches, exactly the way they already
extend a hat-basis run — run a batch, get the checkpoint and VTK series updated, be asked
whether to continue, and repeat — without relearning anything or redefining any physics.

A new resume driver mirrors the existing Yvonnet resume driver in structure and behaviour, but
reads only the uGIMP checkpoint. It rebuilds the material model, padded grid, basis and solver
constants entirely from the parameters recorded in that checkpoint, so the resumed run can
never drift from the run that produced it. It carries over the batch-and-prompt workflow, the
checkpoint save, and the appending of the VTK series and history arrays, so a long run reads as
one continuous simulation.

It must refuse the hat-basis checkpoint. That checkpoint has no particle domain or pad width
recorded, so continuing from it would mean reconstructing the basis by assumption — and a run
whose basis changes midway through is an unpleasant thing to have to explain later. One run,
one basis.

**Blocked by:** 05.

**Status:** ready-for-agent

- [x] Running the driver after the fresh driver continues the simulation from where it stopped,
      with time and step count picking up rather than restarting
- [x] The basis, particle domain half-widths, pad width and grid are rebuilt from the recorded
      parameters, not redefined in the script
- [x] The VTK series and the history arrays are appended to, not overwritten
- [x] The batch-and-prompt workflow behaves as the existing resume driver's does
- [x] A clear error is raised when the uGIMP checkpoint is absent, naming the driver that
      produces it
- [x] The hat-basis checkpoint is refused rather than loaded
- [x] Running the fresh driver, then this driver, then this driver again produces a continuous
      VTK series with monotonically increasing time
- [x] All four existing drivers still run unchanged, and the backbone diff shows additions only

## Comments

- `resume_Yvonnet_gimp.py` at the repo root, mirroring `resume_Yvonnet.py`; the diff is confined to
  the checkpoint path, the basis, the padded-grid rebuild, the refusal check and comments.
- Verified end to end: fresh driver to step 100, then this driver twice at `PF_NSTEPS=10`, giving
  steps 100-109 and 110-119. Time and step count pick up rather than restart, the history arrays
  reach 120 entries with strictly increasing time, and `simulation.pvd` holds three frames at
  0, 1.9959e-07 and 3.9917e-07 s -- appended, not overwritten.
- Refusal verified by copying the hat-basis checkpoint into the uGIMP output directory: the driver
  raises before loading any particle state, naming both the correct resume driver and the fresh
  one. The missing-checkpoint error names `Yvonnet_plate_gimp.py`.
- Worth knowing: the separate output directory is the primary defence, since the hat-basis driver
  never writes to the uGIMP path. The missing-key check is what catches a checkpoint that has been
  copied or renamed by hand.
- The batch-and-prompt loop, the checkpoint save helper and the VTK/history appending are carried
  over unchanged in behaviour.
- Note for redirected output: the batch banner inherits the en-dash and arrow characters from
  `resume_Yvonnet.py`, so piping stdout to a file on a cp1252 console needs
  `PYTHONIOENCODING=utf-8`. This is pre-existing behaviour of the hat-basis driver, kept so the
  two files diff cleanly.
