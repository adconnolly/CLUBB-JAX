# Codex Adversarial Review

Target: clubb_jax/src (JAX port source)
Verdict: needs-attention

No-ship: the JAX source still has paths that silently change physics under autodiff and under unsupported radiation timing, so the port can report finite results for the wrong model.

Findings:
- [high] Microphysics is silently disabled under jax.grad (clubb_jax/src/Microphys/kk_microphys_step.py:87-91)
  Both KK and Morrison return early whenever key state arrays are tracers. In a multi-step rollout this is not just a gradient approximation: the concrete model stores microphysics tendencies at the end of step N and applies them to forcings at step N+1, but the traced path returns before storing them. Whole-driver jax.grad can therefore differentiate a different model with KK/Morrison feedback removed while still producing finite gradients.
  Recommendation: Do not silently skip microphysics during tracing. Either make the tendency computation tracer-compatible end to end, or fail loudly for gradient runs when microphys_scheme is KK/Morrison and max_steps exceeds the single-step case where the tendency is truly dead.
- [high] BUGSrad radiation is silently detached for multi-step gradients (clubb_jax/src/Radiation/radiation_module.py:191-198)
  The BUGSrad branch returns immediately when thlm/rcm/rtm are tracers. The comment acknowledges radht feeds the next step; that means traced multi-step runs keep incoming radht instead of computing the current step's radiative forcing for the next step. This can make jax.grad finite while omitting radiative feedback from the differentiated trajectory.
  Recommendation: Make BUGSrad differentiability explicit: implement a tracer-compatible/lightweight differentiable radiation path, or reject whole-driver gradients for multi-step BUGSrad cases instead of returning a physically different rollout.
- [medium] Radiation cadence truncates invalid dt ratios (clubb_jax/src/advance_clubb_to_end.py:28-35)
  rad_interval is computed with int(dt_rad / dt_main) and then used in modulo scheduling. There is no guard that dt_rad is a positive integer multiple of dt_main. A non-multiple such as 90/60 is truncated to 1, running radiation every main step instead of the requested cadence; dt_rad < dt_main produces rad_interval == 0 and crashes at modulo. Soil/vegetation is also inside this gated radiation call, so cadence mistakes directly perturb surface physics.
  Recommendation: Validate dt_rad >= dt_main and exactly divisible by dt_main during initialization, or replace the integer-ratio scheduler with an accumulated-time scheduler that matches the Fortran oracle for nonstandard cadences.
- [medium] Library import mutates process-global JAX precision (clubb_jax/src/CLUBB_core/advance_clubb_core_module.py:14-16)
  Importing the core module unconditionally calls jax.config.update("jax_enable_x64", True). That is hidden process-global state: importing CLUBB-JAX can change dtype behavior and compilation policy for unrelated JAX code in the same process, while numerical parity of this port depends on when the import happens. This is especially risky for callers embedding the model in training or sensitivity workflows.
  Recommendation: Move x64 configuration to an explicit entrypoint/configuration step before any JAX work, and have library modules assert the expected precision mode rather than mutating global JAX config on import.

Next steps:
- Block shipping until traced multi-step microphysics/radiation behavior is made physically explicit and radiation cadence validation is added.
