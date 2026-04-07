This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

## Architecture 


## Critical Conventions

### CRITICAL: HPC environment rules
### Filesystem
- **NEVER** use `find /`, `find /ocean`, or scan outside the project directory.
  The HPC filesystem has millions of files and these commands will hang forever.
- **GPU diagnostic scripts**: `diags/` — write reusable standalone scripts for targeted investigations (Differentiability, GPU speed up, performance optimization). 
### GPU access
You are running **directly on a GPU compute node** with cuda.
Run python and pytest directly — no wrapper needed. 

### load CUDA and other modules for GPU model runs 
Always run simulations and test on GPU for faster execution and to catch GPU-specific issues. Use the following module commands to set up the environment:

## Commit and Push Policy

Commit and push often after every meaningful unit of work. Keep commits focused: one logical change per commit. This creates a recoverable history if something goes wrong, makes progress visible, and prevents work from being lost if a compute allocation runs out mid-session.

***Rules***
- Each commit implements one thing (one function, one module, one bugfix).
- Avoid large commits that change multiple modules at once.


```bash
git add <specific files>
git commit -m "<short description>"
git push origin main
```

Use descriptive commit messages. Prefix with the affected module or feature (e.g. `MLCanopyFluxes: fix stomatal conductance under low PAR`).

## keep CHANGELOG.md current (agent orientation)

Maintain a `CHANGELOG.md` at the project root to preserve cross-session context. Update it at the end of any session that makes meaningful progress — or immediately when a dead end is identified. `CHANGELOG.md` is the shared memory. Without it, agents waste time re-discovering what's done and what's broken.

**Rules:**
- Update `CHANGELOG.md` after every meaningful unit of work.
- Check off completed items with dates.
- Note what worked, what didn't, what's blocked.
- **Record failed approaches** so they aren't re-attempted. E.g.:
  "Tried ... failed because ... Switched to ..."
- Add new tasks discovered during implementation.
- When stuck, maintain a running doc of attempts in PROGRESS.md.

Keep entries in reverse-chronological order (newest first). Do not delete old entries — they are the record of what has been tried.


## Structure work for parallelism
Parallelism is easy when there are many independent failing tests (each agent picks a different one), but hard when
there's one giant failing task (all agents hit the same bug and overwrite each other). Break the problem into sub-tests.

**Task claiming:** When working in parallel, note your task in CHANGELOG.md
(e.g., "IN PROGRESS: background.py (@agent-1)"). Check CHANGELOG.md before
starting to avoid duplicate work.

### Specialized agent roles: 
- **Implementer agents**: Write module code.
- **Test quality agent**: Reviews and improves the test harness. Adds edge
  cases, improves error messages, catches gaps in coverage.
- **Performance agent**: Profiles the code, identifies bottlenecks, optimizes
  JIT compilation time, reduces memory usage.
- **Code quality agent**: Looks for duplicated code, inconsistent patterns,
  missing type hints, unclear variable names. Refactors.
- **Documentation agent**: Keeps CHANGELOG.md, docstrings, and CLAUDE.md
  in sync with actual code.

## Agent teams (use liberally)
Agent teams are enabled. Use them to parallelize independent work:
Each teammate gets its own context window and can read/write files independently.
Assign different files to different teammates to avoid conflicts.
Teams are especially valuable for this project because:
- GPU diagnostic runs take a long time — use that time for parallel investigation
- Multiple failing tests can be worked on simultaneously
- Different agents can focus on different roles (implementer, tester, performance, documentation)

## Orientation (read this first when starting a session )
When you start a new session, orient yourself:
1. Read `CHANGELOG.md` to see what's done and what's next.
2. Pick the next failing test or unchecked item from CHANGELOG.md.
3. When you finish a unit of work, update CHANGELOG.md before stopping.

## Testing workflow

Each module gets two types of tests:
1. **Value tests**: compare output arrays against clm-ml-fortran reference data
2. **Gradient tests**: compare `jax.grad` output against finite differences

Always write the test first, then make it pass.