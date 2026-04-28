"""
SER file I/O — reader and writer for the SER video format used by FireCapture.

SER format reference:
    Header: 178 bytes (fixed)
    Frames: Width × Height × bytes_per_pixel per frame, contiguous
    Trailer: 8 bytes × FrameCount (int64 UTC timestamps, optional)

ColorID mapping (subset relevant to mono / Bayer):
    0  = MONO
    8  = BayerRGGB
    9  = BayerGRBG
    10 = BayerGBRG
    11 = BayerBGGR
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


# Bayer ColorID → OpenCV demosaic code producing RGB output (ch0=R, ch2=B).
# OpenCV naming is counterintuitive: COLOR_BayerRG2RGB on an RGGB input
# actually yields BGR (ch0=B, ch2=R).  The "opposite" name produces true RGB:
#   RGGB (ID=8)  → COLOR_BayerBG2RGB  (or equivalently BayerRG2BGR)
#   GRBG (ID=9)  → COLOR_BayerGB2RGB
#   GBRG (ID=10) → COLOR_BayerGR2RGB
#   BGGR (ID=11) → COLOR_BayerRG2RGB
_BAYER_TO_RGB: Dict[int, int] = {
    8:  cv2.COLOR_BayerBG2RGB,
    9:  cv2.COLOR_BayerGB2RGB,
    10: cv2.COLOR_BayerGR2RGB,
    11: cv2.COLOR_BayerRG2RGB,
}


class SERReader:
    """Read frames and metadata from a SER file.

    Keeps the file handle open for random-access frame reads.
    Call ``close()`` (or use as a context manager) when done.
    """

    def __init__(self, file_path: Path | str) -> None:
        self.file_path = Path(file_path)
        self.header: Dict[str, object] = {}
        self.frame_size: int = 0
        self._f = open(self.file_path, "rb")
        self._read_header()

    # ── Context manager support ────────────────────────────────────────────────

    def __enter__(self) -> "SERReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Header ────────────────────────────────────────────────────────────────

    def _read_header(self) -> None:
        """Parse the 178-byte SER header."""
        self._f.seek(0)
        data = self._f.read(178)
        self.header["FileID"]      = data[0:14].decode("ascii", errors="replace")
        self.header["LuID"]        = struct.unpack("<I", data[14:18])[0]
        self.header["ColorID"]     = struct.unpack("<I", data[18:22])[0]
        self.header["LittleEndian"] = struct.unpack("<I", data[22:26])[0]
        self.header["Width"]       = struct.unpack("<I", data[26:30])[0]
        self.header["Height"]      = struct.unpack("<I", data[30:34])[0]
        self.header["PixelDepth"]  = struct.unpack("<I", data[34:38])[0]
        self.header["FrameCount"]  = struct.unpack("<I", data[38:42])[0]

        bytes_per_pixel = 1 if self.header["PixelDepth"] <= 8 else 2
        self.frame_size = (
            self.header["Width"] * self.header["Height"] * bytes_per_pixel
        )

    # ── Frame access ──────────────────────────────────────────────────────────

    def get_frame(self, index: int) -> np.ndarray:
        """Return raw (mono/Bayer) frame as a numpy array."""
        if index >= self.header["FrameCount"]:
            raise IndexError(
                f"Frame index {index} out of range "
                f"(FrameCount={self.header['FrameCount']})"
            )
        offset = 178 + index * self.frame_size
        self._f.seek(offset)
        data = self._f.read(self.frame_size)
        dtype = np.uint8 if self.header["PixelDepth"] <= 8 else np.uint16
        frame = np.frombuffer(data, dtype=dtype).reshape(
            (self.header["Height"], self.header["Width"])
        )
        # NOTE: The LittleEndian header field is ignored intentionally.
        # All major capture software (SharpCap, FireCapture, ZWO) writes
        # little-endian data regardless of what this field says (it is almost
        # always 0 even for LE files). PIPP and AS!4 share this behavior.
        return frame

    def to_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Demosaic a raw Bayer frame to RGB (no-op for MONO frames)."""
        color_id = self.header["ColorID"]
        if color_id in _BAYER_TO_RGB:
            return cv2.cvtColor(frame, _BAYER_TO_RGB[color_id])
        return frame

    def get_frame_rgb(self, index: int) -> np.ndarray:
        """Read frame at *index* and return as RGB (or mono if ColorID=0)."""
        return self.to_rgb(self.get_frame(index))

    # ── Timestamps ────────────────────────────────────────────────────────────

    def get_all_timestamps(self) -> List[int]:
        """Extract per-frame UTC timestamps from the file trailer.

        Returns an empty list if the trailer is absent.
        """
        ts_offset = 178 + self.header["FrameCount"] * self.frame_size
        file_size = os.path.getsize(self.file_path)
        timestamps: List[int] = []
        if file_size >= ts_offset + self.header["FrameCount"] * 8:
            self._f.seek(ts_offset)
            for _ in range(self.header["FrameCount"]):
                raw = struct.unpack("<Q", self._f.read(8))[0]
                timestamps.append(raw)
        return timestamps

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        if hasattr(self, "_f") and not self._f.closed:
            self._f.close()


class SERWriter:
    """Write frames to a new SER file.

    Usage::

        writer = SERWriter(path, source_header, width, height)
        for frame, ts in zip(frames, timestamps):
            writer.write_frame(frame, ts)
        writer.close()
    """

    def __init__(
        self,
        file_path: Path | str,
        source_header: Dict[str, object],
        width: int,
        height: int,
    ) -> None:
        self.file_path = Path(file_path)
        self._header = source_header.copy()
        self._header["Width"] = width
        self._header["Height"] = height
        self._header["FrameCount"] = 0
        self._timestamps: List[int] = []
        self._f = open(self.file_path, "wb")
        # Reserve header space (filled in on close)
        self._f.write(b"\x00" * 178)

    def write_frame(self, frame: np.ndarray, timestamp: int = 0) -> None:
        """Append one frame (and its timestamp) to the file."""
        self._f.write(frame.tobytes())
        self._timestamps.append(timestamp)
        self._header["FrameCount"] += 1

    def close(self) -> None:
        """Write timestamp trailer, then seek back and finalise the header."""
        # Trailer
        for ts in self._timestamps:
            self._f.write(struct.pack("<Q", ts))

        # Header
        file_id = self._header.get("FileID", "LUCAM-RECORDER")
        if isinstance(file_id, str):
            file_id = file_id.encode("ascii")
        file_id = file_id[:14].ljust(14, b"\x00")

        header_bytes = struct.pack(
            "<14sIIIIIII40s40s40sQQ",
            file_id,
            int(self._header.get("LuID", 0)),
            int(self._header.get("ColorID", 0)),
            int(self._header.get("LittleEndian", 0)),
            int(self._header["Width"]),
            int(self._header["Height"]),
            int(self._header.get("PixelDepth", 8)),
            int(self._header["FrameCount"]),
            b"",   # Observer
            b"",   # Instrument
            b"",   # Telescope
            0,     # DateTime
            0,     # DateTimeUTC
        )
        # Pad / truncate to exactly 178 bytes
        header_bytes = (header_bytes + b"\x00" * 178)[:178]
        self._f.seek(0)
        self._f.write(header_bytes)
        self._f.close()
