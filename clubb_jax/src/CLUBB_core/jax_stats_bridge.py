"""JAX-side stats event log that replays to the existing CLUBB API.

Porting deviations:
  * This file has no direct Fortran counterpart.  It is bridge infrastructure
    for calling the Fortran-backed stats API from a JAX core that cannot do
    Python/Fortran side effects while jitted.
  * Fortran routines call ``stats_update``, ``stats_begin_budget``,
    ``stats_update_budget``, and ``stats_finalize_budget`` immediately.
    JAX records fixed-size events inside the pytree and replays them to
    ``clubb_python.clubb_api`` after the jitted timestep returns.
  * Stats payloads are stopped with ``jax.lax.stop_gradient``.  Stats writes
    are diagnostics, not part of the differentiable model state.
  * Fortran formatted diagnostic output is not reproduced here.  Event-log
    overflow or invalid payload metadata is raised as a Python error when the
    log is replayed.
"""

from __future__ import annotations

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

OP_NONE = 0
OP_UPDATE = 1
OP_BEGIN_BUDGET = 2
OP_UPDATE_BUDGET = 3
OP_FINALIZE_BUDGET = 4

RANK_SCALAR = 0
RANK_1D = 1
RANK_2D = 2

NO_INDEX = -1

# Setting this low helps keep memory usage small, especially important with lots of stats
# Most variables only have 1 sample, some have 2, and a few have 3 or more. So we could
# probably get away with 2, but let's just do 3 for safety in case someone runs with mostly
# high sample count stats
EVENTS_PER_STATS_VARIABLE = 3


def _clean_stats_string(value) -> str:
    """Normalize Fortran/f2py character metadata returned by ``clubb_api``."""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore")
    elif isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "U"}:
            text = "".join(np.asarray(value).astype(str).reshape(-1).tolist())
        else:
            text = str(value)
    else:
        text = str(value)
    return text.replace("\x00", "").strip()


@jax.tree_util.register_pytree_node_class
class JaxStats:
    """Functional stats recorder for JIT regions.

    Static metadata mirrors the active Fortran-backed stats registry. Dynamic
    leaves are fixed-size arrays so JAX code can record update events without
    host callbacks or Python side effects.
    """

    def __init__(
        self,
        *,
        l_sample: bool,
        names: tuple[str, ...],
        ncol: int,
        max_nlev: int,
        max_events: int,
        count,
        op,
        var_id,
        rank,
        nlev,
        icol,
        level,
        l_count_sample,
        payload_scalar,
        payload_1d,
        payload_2d,
        overflow,
    ):
        self.l_sample = bool(l_sample)
        self.names = tuple(names)
        self.name_to_id = {name: i for i, name in enumerate(self.names)}
        self.ncol = int(ncol)
        self.max_nlev = int(max_nlev)
        self.max_events = int(max_events)
        self.count = count
        self.op = op
        self.var_id = var_id
        self.rank = rank
        self.nlev = nlev
        self.icol = icol
        self.level = level
        self.l_count_sample = l_count_sample
        self.payload_scalar = payload_scalar
        self.payload_1d = payload_1d
        self.payload_2d = payload_2d
        self.overflow = overflow

    @classmethod
    def empty(
        cls,
        *,
        l_sample: bool,
        names: tuple[str, ...],
        ncol: int,
        max_nlev: int,
        max_events: int = 1024,
    ):
        return cls(
            l_sample=l_sample,
            names=tuple(names),
            ncol=ncol,
            max_nlev=max_nlev,
            max_events=max_events,
            count=jnp.asarray(0, dtype=jnp.int32),
            op=jnp.full((max_events,), OP_NONE, dtype=jnp.int32),
            var_id=jnp.full((max_events,), NO_INDEX, dtype=jnp.int32),
            rank=jnp.full((max_events,), NO_INDEX, dtype=jnp.int32),
            nlev=jnp.zeros((max_events,), dtype=jnp.int32),
            icol=jnp.full((max_events,), NO_INDEX, dtype=jnp.int32),
            level=jnp.full((max_events,), NO_INDEX, dtype=jnp.int32),
            l_count_sample=jnp.ones((max_events,), dtype=bool),
            payload_scalar=jnp.zeros((max_events,), dtype=jnp.float64),
            payload_1d=jnp.zeros((max_events, max_nlev), dtype=jnp.float64),
            payload_2d=jnp.zeros((max_events, ncol, max_nlev), dtype=jnp.float64),
            overflow=jnp.asarray(False),
        )

    @classmethod
    def from_api(
        cls,
        *,
        ngrdcol: int,
        nzm: int,
        nzt: int,
        max_events: int | None = None,
    ):
        """Create bridge metadata from the active Fortran-backed stats registry."""
        from clubb_python import clubb_api
        cfg = clubb_api.get_stats_config()
        l_enabled = bool(cfg[0])
        nvars = int(cfg[2]) if l_enabled else 0
        l_sample = bool(cfg[7]) if l_enabled else False

        names = []
        max_nlev = max(int(nzm), int(nzt), 1)
        for ivar in range(nvars):
            name, _grid, _units, _long_name, _grid_id, nz = clubb_api.get_stats_var_meta(ivar)
            clean_name = _clean_stats_string(name)
            if clean_name:
                names.append(clean_name)
            max_nlev = max(max_nlev, int(nz))

        if max_events is None:
            max_events = max(1, EVENTS_PER_STATS_VARIABLE * len(names))

        return cls.empty(
            l_sample=l_sample,
            names=tuple(names),
            ncol=int(ngrdcol),
            max_nlev=max_nlev,
            max_events=max_events,
        )

    @classmethod
    def from_writer(
        cls,
        stats_writer,
        *,
        ngrdcol: int,
        nzm: int,
        nzt: int,
        max_events: int | None = None,
    ):
        """Create bridge metadata from the pure-Python :class:`StatsWriter`.

        f2py-free analogue of :meth:`from_api`: the enabled-variable set and the
        sample flag come from the writer's parsed registry instead of the
        Fortran ``get_stats_config``/``get_stats_var_meta`` calls.
        """
        names = tuple(stats_writer.registry.keys())
        max_nlev = max(int(nzm), int(nzt), 1)
        if max_events is None:
            max_events = max(1, EVENTS_PER_STATS_VARIABLE * len(names))
        return cls.empty(
            l_sample=bool(stats_writer.l_sample),
            names=names,
            ncol=int(ngrdcol),
            max_nlev=max_nlev,
            max_events=max_events,
        )

    def tree_flatten(self):
        children = (
            self.count,
            self.op,
            self.var_id,
            self.rank,
            self.nlev,
            self.icol,
            self.level,
            self.l_count_sample,
            self.payload_scalar,
            self.payload_1d,
            self.payload_2d,
            self.overflow,
        )
        aux_data = (
            self.l_sample,
            self.names,
            self.ncol,
            self.max_nlev,
            self.max_events,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        l_sample, names, ncol, max_nlev, max_events = aux_data
        (
            count,
            op,
            var_id,
            rank,
            nlev,
            icol,
            level,
            l_count_sample,
            payload_scalar,
            payload_1d,
            payload_2d,
            overflow,
        ) = children
        return cls(
            l_sample=l_sample,
            names=names,
            ncol=ncol,
            max_nlev=max_nlev,
            max_events=max_events,
            count=count,
            op=op,
            var_id=var_id,
            rank=rank,
            nlev=nlev,
            icol=icol,
            level=level,
            l_count_sample=l_count_sample,
            payload_scalar=payload_scalar,
            payload_1d=payload_1d,
            payload_2d=payload_2d,
            overflow=overflow,
        )

    def reset(self, *, l_sample: bool | None = None):
        return type(self).empty(
            l_sample=self.l_sample if l_sample is None else bool(l_sample),
            names=self.names,
            ncol=self.ncol,
            max_nlev=self.max_nlev,
            max_events=self.max_events,
        )

    def var_on_stats_list(self, name: str) -> bool:
        """Mirror ``var_on_stats_list`` for code that guards optional stats."""
        return bool(name) and name.strip() in self.name_to_id

    def update(self, name: str, values, *, icol: int | None = None, level: int | None = None):
        """Record a future ``stats_update`` call."""
        return self._record(OP_UPDATE, name, values, icol=icol, level=level)

    def begin_budget(self, name: str, values, *, icol: int | None = None):
        """Record a future ``stats_begin_budget`` call."""
        return self._record(OP_BEGIN_BUDGET, name, values, icol=icol)

    def update_budget(self, name: str, values, *, icol: int | None = None, level: int | None = None):
        """Record a future ``stats_update_budget`` call."""
        return self._record(OP_UPDATE_BUDGET, name, values, icol=icol, level=level)

    def finalize_budget(
        self,
        name: str,
        values,
        *,
        icol: int | None = None,
        l_count_sample: bool = True,
    ):
        """Record a future ``stats_finalize_budget`` call."""
        return self._record(
            OP_FINALIZE_BUDGET,
            name,
            values,
            icol=icol,
            l_count_sample=l_count_sample,
        )

    def _record(
        self,
        op_code: int,
        name: str,
        values,
        *,
        icol: int | None = None,
        level: int | None = None,
        l_count_sample: bool = True,
    ):
        clean_name = name.strip()
        if (not self.l_sample) or (not clean_name) or clean_name not in self.name_to_id:
            return self

        arr = jnp.asarray(values, dtype=jnp.float64)
        if arr.ndim > 2:
            raise ValueError(f"JaxStats supports rank <= 2 payloads; got rank {arr.ndim}.")

        idx = jnp.minimum(self.count, self.max_events - 1)
        zero_idx = jnp.asarray(0, dtype=idx.dtype)
        do_set = self.count < self.max_events
        new_count = jnp.where(do_set, self.count + 1, self.count)
        new_overflow = self.overflow | jnp.logical_not(do_set)

        op = self.op.at[idx].set(jnp.where(do_set, op_code, self.op[idx]))
        var_id = self.var_id.at[idx].set(
            jnp.where(do_set, self.name_to_id[clean_name], self.var_id[idx])
        )
        rank = self.rank.at[idx].set(jnp.where(do_set, arr.ndim, self.rank[idx]))
        icol_val = NO_INDEX if icol is None else int(icol)
        level_val = NO_INDEX if level is None else int(level)
        icol_arr = self.icol.at[idx].set(jnp.where(do_set, icol_val, self.icol[idx]))
        level_arr = self.level.at[idx].set(jnp.where(do_set, level_val, self.level[idx]))
        count_sample_arr = self.l_count_sample.at[idx].set(
            jnp.where(do_set, bool(l_count_sample), self.l_count_sample[idx])
        )

        payload_scalar = self.payload_scalar
        payload_1d = self.payload_1d
        payload_2d = self.payload_2d

        if arr.ndim == 0:
            nlev_value = 1
            payload_scalar = payload_scalar.at[idx].set(
                jnp.where(do_set, arr.reshape(()), payload_scalar[idx])
            )
        elif arr.ndim == 1:
            nlev_value = arr.shape[0]
            if nlev_value > self.max_nlev:
                raise ValueError(
                    f"Stats payload length {nlev_value} exceeds max_nlev={self.max_nlev}."
            )
            current = jax.lax.dynamic_slice(
                payload_1d, (idx, zero_idx), (1, nlev_value),
            )
            update = jnp.where(
                do_set,
                jax.lax.stop_gradient(arr)[None, :],
                current,
            )
            payload_1d = jax.lax.dynamic_update_slice(
                payload_1d, update, (idx, zero_idx),
            )
        else:
            ncol_value, nlev_value = arr.shape
            if ncol_value > self.ncol:
                raise ValueError(
                    f"Stats payload ncol {ncol_value} exceeds ncol={self.ncol}."
                )
            if nlev_value > self.max_nlev:
                raise ValueError(
                    f"Stats payload nlev {nlev_value} exceeds max_nlev={self.max_nlev}."
            )
            current = jax.lax.dynamic_slice(
                payload_2d, (idx, zero_idx, zero_idx), (1, ncol_value, nlev_value),
            )
            update = jnp.where(
                do_set,
                jax.lax.stop_gradient(arr)[None, :, :],
                current,
            )
            payload_2d = jax.lax.dynamic_update_slice(
                payload_2d, update, (idx, zero_idx, zero_idx),
            )

        nlev = self.nlev.at[idx].set(jnp.where(do_set, nlev_value, self.nlev[idx]))

        return type(self)(
            l_sample=self.l_sample,
            names=self.names,
            ncol=self.ncol,
            max_nlev=self.max_nlev,
            max_events=self.max_events,
            count=new_count,
            op=op,
            var_id=var_id,
            rank=rank,
            nlev=nlev,
            icol=icol_arr,
            level=level_arr,
            l_count_sample=count_sample_arr,
            payload_scalar=payload_scalar,
            payload_1d=payload_1d,
            payload_2d=payload_2d,
            overflow=new_overflow,
        )

    def _replay(self):
        """Decode recorded events into (op_code, name, values, icol, level, l_count_sample) tuples.

        Shared by :meth:`to_api` (Fortran) and :meth:`to_writer` (pure-Python) so the
        two back-ends apply byte-identical payloads in identical order.
        """
        if bool(np.asarray(self.overflow)):
            raise RuntimeError(
                f"JaxStats event log overflowed max_events={self.max_events}."
            )

        count = int(np.asarray(self.count))
        op = np.asarray(self.op)
        var_id = np.asarray(self.var_id)
        rank = np.asarray(self.rank)
        nlev = np.asarray(self.nlev)
        icol = np.asarray(self.icol)
        level = np.asarray(self.level)
        l_count_sample = np.asarray(self.l_count_sample)
        payload_scalar = np.asarray(self.payload_scalar)
        payload_1d = np.asarray(self.payload_1d)
        payload_2d = np.asarray(self.payload_2d)

        for i in range(count):
            op_code = int(op[i])
            if op_code == OP_NONE:
                continue
            name = self.names[int(var_id[i])]
            rank_i = int(rank[i])
            nlev_i = int(nlev[i])
            icol_arg = None if int(icol[i]) == NO_INDEX else int(icol[i])
            level_arg = None if int(level[i]) == NO_INDEX else int(level[i])
            if rank_i == RANK_SCALAR:
                values = float(payload_scalar[i])
            elif rank_i == RANK_1D:
                values = payload_1d[i, :nlev_i]
            elif rank_i == RANK_2D:
                values = payload_2d[i, :, :nlev_i]
            else:
                raise RuntimeError(f"Invalid JaxStats payload rank {rank_i}.")
            yield op_code, name, values, icol_arg, level_arg, bool(l_count_sample[i])

    def to_writer(self, stats_writer):
        """Replay recorded events to the pure-Python :class:`StatsWriter` (no f2py).

        Maps the recorded ops onto the writer's methods with the same semantics the
        Fortran-backed API applies: per-column scalar updates route to ``update_col``;
        budget ops are guarded by the writer's active-budget state.
        """
        for op_code, name, values, icol_arg, level_arg, l_count_sample in self._replay():
            if op_code == OP_UPDATE:
                if icol_arg is not None and np.ndim(values) == 0:
                    stats_writer.update_col(name, float(values), icol=icol_arg)
                else:
                    stats_writer.update(name, values)
            elif op_code == OP_BEGIN_BUDGET:
                stats_writer.begin_budget(name, values)
            elif op_code == OP_UPDATE_BUDGET:
                stats_writer.update_budget(name, values)
            elif op_code == OP_FINALIZE_BUDGET:
                stats_writer.finalize_budget(name, values, l_count_sample=l_count_sample)
            else:
                raise RuntimeError(f"Invalid JaxStats op code {op_code}.")

    def to_api(self):
        """Replay recorded events to the existing CLUBB stats API."""
        from clubb_python import clubb_api
        if bool(np.asarray(self.overflow)):
            raise RuntimeError(
                f"JaxStats event log overflowed max_events={self.max_events}."
            )

        count = int(np.asarray(self.count))
        op = np.asarray(self.op)
        var_id = np.asarray(self.var_id)
        rank = np.asarray(self.rank)
        nlev = np.asarray(self.nlev)
        icol = np.asarray(self.icol)
        level = np.asarray(self.level)
        l_count_sample = np.asarray(self.l_count_sample)
        payload_scalar = np.asarray(self.payload_scalar)
        payload_1d = np.asarray(self.payload_1d)
        payload_2d = np.asarray(self.payload_2d)

        for i in range(count):
            op_code = int(op[i])
            if op_code == OP_NONE:
                continue

            name = self.names[int(var_id[i])]
            rank_i = int(rank[i])
            nlev_i = int(nlev[i])
            icol_arg = None if int(icol[i]) == NO_INDEX else int(icol[i])
            level_arg = None if int(level[i]) == NO_INDEX else int(level[i])

            if rank_i == RANK_SCALAR:
                values = float(payload_scalar[i])
            elif rank_i == RANK_1D:
                values = payload_1d[i, :nlev_i]
            elif rank_i == RANK_2D:
                values = payload_2d[i, :, :nlev_i]
            else:
                raise RuntimeError(f"Invalid JaxStats payload rank {rank_i}.")

            if op_code == OP_UPDATE:
                clubb_api.stats_update(name, values, icol=icol_arg, level=level_arg)
            elif op_code == OP_BEGIN_BUDGET:
                clubb_api.stats_begin_budget(name, values, icol=icol_arg)
            elif op_code == OP_UPDATE_BUDGET:
                clubb_api.stats_update_budget(name, values, icol=icol_arg, level=level_arg)
            elif op_code == OP_FINALIZE_BUDGET:
                clubb_api.stats_finalize_budget(
                    name,
                    values,
                    icol=icol_arg,
                    l_count_sample=bool(l_count_sample[i]),
                )
            else:
                raise RuntimeError(f"Invalid JaxStats op code {op_code}.")
