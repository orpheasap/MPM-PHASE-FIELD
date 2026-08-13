# 01 — Project glossary

**What to build:** A researcher opening this repo — or an agent working in it — can look up one
agreed meaning for the terms the MPM phase-field code uses, instead of inferring them from
variable names. A glossary lives at the repo root and defines the vocabulary this feature
introduces or leans on.

Terms to cover: background grid, particle mesh, particle domain, support, connectivity,
pseudo-mass, bulk damage, interfacial damage, pad ring, notch.

This is a glossary and nothing else. No implementation detail, no spec content, no
architecture decisions, no file paths, no formulae beyond what is needed to distinguish one
term from another.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [x] A glossary file exists at the repo root
- [x] Every term listed above has an entry
- [x] Each entry distinguishes the term from the ones it is easily confused with — in
      particular background grid vs particle mesh, and bulk damage vs interfacial damage
- [x] No entry contains implementation detail, code, or a file path
- [x] No existing file is modified

## Comments

- Implemented as `CONTEXT.md` at the repo root — the glossary filename this repo's domain-doc
  convention (`docs/agents/domain.md`) already points every skill at. No existing file was
  modified.
