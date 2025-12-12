import math
from typing import List, Tuple, Optional, Iterable

class ChunkPlanner:
    """
    Plan and analyze fixed-size chunking with overlap over a linear sequence
    """

    def __init__(self, length: int, chunk_size: Optional[int] = None, overlap: int = 0) -> None:
        if length <= 0:
            raise ValueError("length must be a positive integer.")
        if overlap < 0:
            raise ValueError("overlap must be a non-negative integer.")
        if chunk_size is not None:
            if chunk_size <= 0:
                raise ValueError("chunk_size must be a positive integer.")
            if overlap >= chunk_size:
                raise ValueError("overlap must be strictly less than chunk_size.")

        self.length = length
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ---------- Core formulas ----------

    @staticmethod
    def num_chunks_static(length: int, chunk_size: int, overlap: int) -> int:
        """Static version: N = ceil((L - O) / (C - O))."""
        if length <= 0 or chunk_size <= 0:
            raise ValueError("length and chunk_size must be positive.")
        if not (0 <= overlap < chunk_size):
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size.")
        step = chunk_size - overlap
        if chunk_size >= length:
            return 1
        return math.ceil((length - overlap) / step)

    def num_chunks(self) -> int:
        """Compute number of chunks using current planner settings."""
        if self.chunk_size is None:
            raise ValueError("chunk_size is not set.")
        return self.num_chunks_static(self.length, self.chunk_size, self.overlap)

    @staticmethod
    def max_chunk_size(length: int, overlap: int, max_chunks: int) -> int:
        """
        Maximize chunk size C subject to N <= max_chunks.
        Derived: C = (L - O) / N_max + O, then floor to integer.
        """
        if length <= 0:
            raise ValueError("length must be a positive integer.")
        if overlap < 0:
            raise ValueError("overlap must be a non-negative integer.")
        if max_chunks <= 0:
            raise ValueError("max_chunks must be a positive integer.")

        c_cont = (length - overlap) / max_chunks + overlap
        c_max = math.floor(c_cont)
        if c_max <= overlap:
            raise ValueError(
                "No valid chunk_size exists (computed C_max <= overlap). "
                "Increase max_chunks or decrease overlap."
            )
        return c_max

    @staticmethod
    def max_overlap_for_Nmax(length: int, chunk_size: int, Nmax: int) -> int:
        """
        Largest integer overlap O in [0, C-1] such that N <= Nmax.
        Uses bound: O <= (Nmax*C - L) / (Nmax - 1) for Nmax >= 2.
        Special case Nmax=1: requires C >= L; then O_max = C-1.
        """
        if length <= 0 or chunk_size <= 0 or Nmax <= 0:
            raise ValueError("length, chunk_size, Nmax must be positive.")
        if Nmax == 1:
            if chunk_size < length:
                raise ValueError("Nmax=1 impossible unless chunk_size >= length.")
            return chunk_size - 1

        bound = (Nmax * chunk_size - length) / (Nmax - 1)
        Omax = math.floor(bound)
        Omax = max(0, min(Omax, chunk_size - 1))
        # Verify
        while Omax >= 0 and ChunkPlanner.num_chunks_static(length, chunk_size, Omax) > Nmax:
            Omax -= 1
        if Omax < 0:
            raise ValueError("No non-negative overlap satisfies N <= Nmax.")
        return Omax

    @staticmethod
    def overlaps_for_exact_N(length: int, chunk_size: int, N: int) -> List[int]:
        """
        Return all integer overlaps O in [0, C-1] that yield exactly N chunks.
        Uses inequality interval; falls back to checking when N < 3.
        """
        if length <= 0 or chunk_size <= 0 or N <= 0:
            raise ValueError("length, chunk_size, N must be positive.")

        valid: List[int] = []
        if N >= 3:
            lower = ((N - 1) * chunk_size - length) / (N - 2)  # O > lower
            upper = (N * chunk_size - length) / (N - 1)        # O <= upper
            start = max(0, math.floor(lower) + 1)
            end = min(chunk_size - 1, math.floor(upper))
            for O in range(start, end + 1):
                if ChunkPlanner.num_chunks_static(length, chunk_size, O) == N:
                    valid.append(O)
            return valid

        # Edge cases N=1 or N=2
        for O in range(0, chunk_size):
            if ChunkPlanner.num_chunks_static(length, chunk_size, O) == N:
                valid.append(O)
        return valid

    # ---------- Indices & splitting ----------

    @staticmethod
    def chunk_indices(length: int, chunk_size: int, overlap: int) -> List[Tuple[int, int]]:
        """
        Return start/end indices (end-exclusive) for each chunk covering [0, length).
        """
        n = ChunkPlanner.num_chunks_static(length, chunk_size, overlap)
        step = chunk_size - overlap
        indices: List[Tuple[int, int]] = []
        start = 0
        for _ in range(n):
            end = min(start + chunk_size, length)
            indices.append((start, end))
            start += step
            if start >= length:
                break
        return indices

    def indices(self) -> List[Tuple[int, int]]:
        """Instance method variant using current settings."""
        if self.chunk_size is None:
            raise ValueError("chunk_size is not set.")
        return self.chunk_indices(self.length, self.chunk_size, self.overlap)

    @staticmethod
    def split_sequence(seq: Iterable, chunk_size: int, overlap: int) -> List[List]:
        """
        Split a generic sequence into overlapping chunks using indices.
        Works for lists, strings, etc.
        """
        if hasattr(seq, "__len__"):
            length = len(seq)  # type: ignore
        else:
            raise TypeError("seq must be a sized iterable (e.g., list or string).")

        ranges = ChunkPlanner.chunk_indices(length, chunk_size, overlap)
        result: List[List] = []
        # Convert to list for slicing generically
        seq_list = list(seq)
        for s, e in ranges:
            result.append(seq_list[s:e])
        return result

    def split_text(self, text: str) -> List[str]:
        """
        Split a string 'text' using current planner's C and O.
        """
        if self.chunk_size is None:
            raise ValueError("chunk_size is not set.")
        # Use indices to slice text
        parts = []
        for s, e in self.indices():
            parts.append(text[s:e])
        return parts

    # ---------- Convenience setters ----------

    def set_chunk_size(self, chunk_size: int) -> None:
        """Update chunk_size with validation against current overlap."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.overlap >= chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size.")
        self.chunk_size = chunk_size

    def set_overlap(self, overlap: int) -> None:
        """Update overlap with validation against current chunk_size (if set)."""
        if overlap < 0:
            raise ValueError("overlap must be non-negative.")
        if self.chunk_size is not None and overlap >= self.chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size.")
        self.overlap = overlap