"""
The running experiment report. The narrative parts (summary, current activity,
procedure steps, materials, notes) come from the VLM; the plate map and the
dispense log are rendered directly from the tracked well-plate state, so the
numbers in the report are always the measured ones.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from experiment_state import WellPlateState, format_time


@dataclass
class ExperimentReport:
    recording: str
    summary: str = ""
    current_activity: str = ""
    steps: list = field(default_factory=list)       # {"time", "description"}
    materials: list = field(default_factory=list)   # str
    notes: list = field(default_factory=list)       # {"time", "text"}
    updated_at_s: Optional[float] = None
    n_vlm_updates: int = 0

    def apply(self, update, time_s):
        """Merge one VLM ReportUpdate (see vlm_reporter.ReportUpdate)."""
        t = format_time(time_s)
        self.summary = update.summary.strip() or self.summary
        self.current_activity = update.current_activity.strip()
        known_steps = {s["description"].lower() for s in self.steps}
        for desc in update.new_steps:
            desc = desc.strip()
            if desc and desc.lower() not in known_steps:
                self.steps.append({"time": t, "description": desc})
                known_steps.add(desc.lower())
        known_materials = {m.lower() for m in self.materials}
        for m in update.new_materials:
            m = m.strip()
            if m and m.lower() not in known_materials:
                self.materials.append(m)
                known_materials.add(m.lower())
        if update.note.strip():
            self.notes.append({"time": t, "text": update.note.strip()})
        self.updated_at_s = time_s
        self.n_vlm_updates += 1

    def recent_steps_text(self, n):
        if not self.steps:
            return "(none yet)"
        return "\n".join(f"- [{s['time']}] {s['description']}" for s in self.steps[-n:])

    def to_markdown(self, plate: WellPlateState):
        updated = "-" if self.updated_at_s is None else format_time(self.updated_at_s)
        md = [f"# Experiment report", "",
              f"_Recording: {self.recording} · last updated at {updated} · "
              f"{self.n_vlm_updates} VLM updates_", "",
              "## Summary", self.summary or "_Not enough observations yet._", "",
              "## Current activity", self.current_activity or "-", "",
              "## Procedure log"]
        md += ([f"- **{s['time']}** {s['description']}" for s in self.steps]
               or ["_No steps recorded yet._"])
        md += ["", "## Well plate",
               f"{len(plate.wells())} wells filled, {len(plate.events)} dispenses "
               "(µL per well; ? = volume not read)", "", plate.grid_markdown(), "",
               "### Dispense log"]
        if plate.events:
            md += ["| Time | Well | Volume (µL) |", "|---|---|---|"]
            md += [f"| {format_time(e.time_s)} | {e.well} | "
                   f"{'?' if e.volume_ul is None else f'{e.volume_ul:g}'} |" for e in plate.events]
        else:
            md.append("_No dispenses detected yet._")
        md += ["", "## Materials and equipment"]
        md += [f"- {m}" for m in self.materials] or ["-"]
        md += ["", "## Notes"]
        md += [f"- **{n['time']}** {n['text']}" for n in self.notes] or ["-"]
        return "\n".join(md) + "\n"

    def save(self, out_dir, plate: WellPlateState):
        out_dir = Path(out_dir)
        (out_dir / "report.md").write_text(self.to_markdown(plate))
        (out_dir / "report.json").write_text(json.dumps(
            {"report": asdict(self), "plate": plate.to_dict(recent_events=len(plate.events)),
             "dispenses": [e.to_dict() for e in plate.events]}, indent=2))
