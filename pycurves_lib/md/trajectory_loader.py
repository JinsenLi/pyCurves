from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from pycurves_lib.data.modified_bases import is_known_modified_base


@dataclass
class TrajectoryFrame:
    index: int
    time: Optional[float]
    coordinates: np.ndarray


@dataclass
class TrajectoryBatch:
    indices: list[int]
    times: list[Optional[float]]
    coordinates: np.ndarray


class TrajectoryLoader:
    """Small adapter layer around common MD trajectory readers."""

    @staticmethod
    def iter_frames(topology_file: str, trajectory_file: Optional[str] = None, frame_selector=None) -> Iterator[TrajectoryFrame]:
        if frame_selector is None:
            frame_selector = lambda i: True

        if trajectory_file is None:
            yield from TrajectoryLoader._iter_multimodel_pdb(topology_file, frame_selector)
            return

        try:
            yield from TrajectoryLoader._iter_mdanalysis(topology_file, trajectory_file, frame_selector)
            return
        except ImportError:
            pass

        try:
            yield from TrajectoryLoader._iter_mdtraj(topology_file, trajectory_file, frame_selector)
            return
        except ImportError:
            pass

        raise ImportError(
            "Trajectory input requires either MDAnalysis or mdtraj. "
            "Install the full optional dependencies with `pip install '.[all]'`, "
            "or provide a multi-model PDB as the topology input with no trajectory file."
        )

    @staticmethod
    def iter_batches(
        topology_file: str,
        trajectory_file: Optional[str] = None,
        frame_selector=None,
        batch_size: int = 256,
    ) -> Iterator[TrajectoryBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if frame_selector is None:
            frame_selector = lambda i: True

        if trajectory_file is None:
            yield from TrajectoryLoader._batch_frames(
                TrajectoryLoader._iter_multimodel_pdb(topology_file, frame_selector),
                batch_size,
            )
            return

        try:
            yield from TrajectoryLoader._iter_mdanalysis_batches(
                topology_file, trajectory_file, frame_selector, batch_size
            )
            return
        except ImportError:
            pass

        try:
            yield from TrajectoryLoader._batch_frames(
                TrajectoryLoader._iter_mdtraj(topology_file, trajectory_file, frame_selector),
                batch_size,
            )
            return
        except ImportError:
            pass

        raise ImportError(
            "Trajectory input requires either MDAnalysis or mdtraj. "
            "Install the full optional dependencies with `pip install '.[all]'`, "
            "or provide a multi-model PDB as the topology input with no trajectory file."
        )

    @staticmethod
    def _selected_indices(frame_count: int, frame_selector) -> Iterable[int]:
        explicit_indices = getattr(frame_selector, "explicit_indices", None)
        if explicit_indices is not None:
            return (
                int(index) for index in explicit_indices
                if 0 <= int(index) < frame_count and frame_selector(int(index))
            )

        frame_range = getattr(frame_selector, "mdtraj_range", None)
        if frame_range is not None:
            start, stop, step = frame_range
            start = int(start)
            step = int(step)
            first = max(0, start)
            remainder = (first - start) % step
            if remainder:
                first += step - remainder
            end = frame_count if stop is None else min(frame_count, int(stop))
            return range(first, max(first, end), step)

        return (index for index in range(frame_count) if frame_selector(index))

    @staticmethod
    def _batch_frames(frames: Iterable[TrajectoryFrame], batch_size: int) -> Iterator[TrajectoryBatch]:
        coordinates = []
        indices = []
        times = []
        for frame in frames:
            coordinates.append(frame.coordinates)
            indices.append(int(frame.index))
            times.append(None if frame.time is None else float(frame.time))
            if len(coordinates) == batch_size:
                yield TrajectoryBatch(indices, times, np.asarray(coordinates, dtype=float))
                coordinates, indices, times = [], [], []
        if coordinates:
            yield TrajectoryBatch(indices, times, np.asarray(coordinates, dtype=float))

    @staticmethod
    def _iter_mdanalysis(topology_file: str, trajectory_file: str, frame_selector) -> Iterator[TrajectoryFrame]:
        import MDAnalysis as mda

        universe = mda.Universe(topology_file, trajectory_file)
        for i in TrajectoryLoader._selected_indices(universe.trajectory.n_frames, frame_selector):
            ts = universe.trajectory[i]
            yield TrajectoryFrame(
                index=int(ts.frame),
                time=float(ts.time) if ts.time is not None else None,
                coordinates=np.asarray(universe.atoms.positions, dtype=float).copy(),
            )

    @staticmethod
    def _iter_mdanalysis_batches(
        topology_file: str,
        trajectory_file: str,
        frame_selector,
        batch_size: int,
    ) -> Iterator[TrajectoryBatch]:
        import MDAnalysis as mda

        universe = mda.Universe(topology_file, trajectory_file)
        coordinates = np.empty((batch_size, universe.atoms.n_atoms, 3), dtype=float)
        indices = []
        times = []
        for i in TrajectoryLoader._selected_indices(universe.trajectory.n_frames, frame_selector):
            ts = universe.trajectory[i]
            coordinates[len(indices)] = ts.positions
            indices.append(int(ts.frame))
            times.append(float(ts.time) if ts.time is not None else None)
            if len(indices) == batch_size:
                yield TrajectoryBatch(indices, times, coordinates)
                coordinates = np.empty_like(coordinates)
                indices, times = [], []
        if indices:
            yield TrajectoryBatch(indices, times, coordinates[:len(indices)])

    @staticmethod
    def _iter_mdtraj(topology_file: str, trajectory_file: str, frame_selector) -> Iterator[TrajectoryFrame]:
        import mdtraj as md

        explicit_indices = getattr(frame_selector, "explicit_indices", None)
        mdtraj_range = getattr(frame_selector, "mdtraj_range", None)
        if explicit_indices is not None:
            for idx in explicit_indices:
                if not frame_selector(idx):
                    continue
                traj = md.load_frame(trajectory_file, int(idx), top=topology_file)
                time = float(traj.time[0]) if getattr(traj, "time", None) is not None and len(traj.time) else None
                yield TrajectoryFrame(
                    index=int(idx),
                    time=time,
                    coordinates=np.asarray(traj.xyz[0], dtype=float) * 10.0,
                )
            return

        stop = mdtraj_range[1] if mdtraj_range is not None else None
        next_index = 0
        # Some large XTC offset tables fail random seeking. Sequential chunks
        # keep arbitrary start/step selections correct without using `skip=`.
        for chunk in md.iterload(trajectory_file, top=topology_file, chunk=50):
            times = getattr(chunk, "time", None)
            for offset in range(chunk.n_frames):
                idx = next_index + offset
                if stop is not None and idx >= stop:
                    return
                if not frame_selector(idx):
                    continue
                time = float(times[offset]) if times is not None and len(times) > offset else None
                yield TrajectoryFrame(
                    index=idx,
                    time=time,
                    coordinates=np.asarray(chunk.xyz[offset], dtype=float) * 10.0,
                )
            next_index += chunk.n_frames

    @staticmethod
    def _iter_multimodel_pdb(pdb_file: str, frame_selector) -> Iterator[TrajectoryFrame]:
        path = Path(pdb_file)
        if path.suffix.lower() not in {".pdb", ".brk"}:
            raise ImportError("Built-in trajectory fallback only supports multi-model PDB files.")

        frames = []
        current = []
        saw_model = False
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                record = line[:6]
                if record.startswith("MODEL"):
                    saw_model = True
                    current = []
                    continue
                if record.startswith("ENDMDL"):
                    if current:
                        frames.append(np.asarray(current, dtype=float))
                    current = []
                    continue
                if record == "HETATM" and not is_known_modified_base(line[17:20]):
                    continue
                if record not in {"ATOM  ", "HETATM"}:
                    continue
                try:
                    current.append([
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ])
                except ValueError:
                    continue

        if not saw_model:
            raise ImportError("No MODEL/ENDMDL records found; this is not a multi-model PDB trajectory.")
        if current:
            frames.append(np.asarray(current, dtype=float))

        for idx, coords in enumerate(frames):
            if frame_selector(idx):
                yield TrajectoryFrame(index=idx, time=None, coordinates=coords)

