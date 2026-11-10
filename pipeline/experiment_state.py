"""Internal state of the well plate: which wells were filled, when, with how much."""
from dataclasses import asdict, dataclass, field
from typing import Optional

import settings as S
from plate import well_name


def format_time(seconds):
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


@dataclass
class DispenseEvent:
    time_s: float
    frame_idx: int
    well: str
    row: int                    # grid cell (see plate.grid_cell_at)
    col: int
    volume_ul: Optional[float]  # None if the volume counter couldn't be read
    press_p: Optional[float]

    def to_dict(self):
        d = asdict(self)
        d["time"] = format_time(self.time_s)
        return d


@dataclass
class WellPlateState:
    rows: int = S.GRID_ROWS
    cols: int = S.GRID_COLS
    events: list = field(default_factory=list)   # DispenseEvent, in order

    def record_dispense(self, time_s, frame_idx, row, col, volume_ul, press_p) -> DispenseEvent:
        ev = DispenseEvent(time_s, frame_idx, well_name(row, col), int(row), int(col),
                           volume_ul, press_p)
        self.events.append(ev)
        return ev

    def wells(self):
        """well -> {"dispenses", "total_volume_ul", "unknown_volume_dispenses"} for filled wells."""
        out = {}
        for ev in self.events:
            w = out.setdefault(ev.well, {"dispenses": 0, "total_volume_ul": 0.0,
                                         "unknown_volume_dispenses": 0})
            w["dispenses"] += 1
            if ev.volume_ul is None:
                w["unknown_volume_dispenses"] += 1
            else:
                w["total_volume_ul"] += ev.volume_ul
        return out

    def to_dict(self, recent_events=5):
        """Compact summary (for the VLM and the report JSON)."""
        wells = self.wells()
        return {
            "plate_format": f"{self.rows * self.cols}-well ({self.rows} rows A-"
                            f"{chr(ord('A') + self.rows - 1)}, {self.cols} columns)",
            "n_wells_filled": len(wells),
            "n_dispenses": len(self.events),
            "filled_wells": wells,
            "recent_dispenses": [ev.to_dict() for ev in self.events[-recent_events:]],
        }

    def grid_markdown(self):
        """Plate map: total µL per well ("?" when a volume couldn't be read)."""
        wells = self.wells()
        header = "|   | " + " | ".join(str(c + 1) for c in range(self.cols)) + " |"
        lines = [header, "|---" * (self.cols + 1) + "|"]
        for r in range(self.rows):
            letter = chr(ord("A") + r)
            cells = []
            for c in range(self.cols):
                w = wells.get(f"{letter}{c + 1}")
                if w is None:
                    cells.append("·")
                elif w["unknown_volume_dispenses"] == w["dispenses"]:
                    cells.append("?")
                else:
                    cells.append(f"{w['total_volume_ul']:g}"
                                 + ("+?" if w["unknown_volume_dispenses"] else ""))
            lines.append(f"| **{letter}** | " + " | ".join(cells) + " |")
        return "\n".join(lines)
