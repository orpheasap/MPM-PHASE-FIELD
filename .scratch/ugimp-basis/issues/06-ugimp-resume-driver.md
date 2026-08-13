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

- [ ] Running the driver after the fresh driver continues the simulation from where it stopped,
      with time and step count picking up rather than restarting
- [ ] The basis, particle domain half-widths, pad width and grid are rebuilt from the recorded
      parameters, not redefined in the script
- [ ] The VTK series and the history arrays are appended to, not overwritten
- [ ] The batch-and-prompt workflow behaves as the existing resume driver's does
- [ ] A clear error is raised when the uGIMP checkpoint is absent, naming the driver that
      produces it
- [ ] The hat-basis checkpoint is refused rather than loaded
- [ ] Running the fresh driver, then this driver, then this driver again produces a continuous
      VTK series with monotonically increasing time
- [ ] All four existing drivers still run unchanged, and the backbone diff shows additions only
