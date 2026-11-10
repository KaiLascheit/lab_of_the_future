# Pipeline

Aria VRS recording → 5 Hz frames → objects, pipette/plate tracking, press, volume →
well-plate state → running VLM experiment report (1 Hz).

```
cd pipeline
python main.py --vrs ../streams/<recording>.vrs [--start 3:20 --end 4:30] [--no-vlm]
```

All settings are in `settings.py`. Paths there are relative to the project root
(the folder containing `pipeline/`): `streams/`, `dataset_building/scripts/`,
`volume_templates/`, `rigid_body_calibration.json`, and `.env` with `GEMINI_API_KEY`.

| File | Role |
|---|---|
| `main.py` | command line, frame loop |
| `experiment_pipeline.py` | per-frame orchestration (order of steps, gating, logging) |
| `settings.py` | every path, rate, threshold, model choice |
| `vrs_reader.py` | RGB frames at 5 Hz + camera calibration from the VRS |
| `object_detector.py` | YOLO-World labels + MobileSAM masks → objects for the VLM, pipette gate |
| `apriltag_tracker.py` | per frame: tags → pipette pose → plate box → ray → hit cell |
| `markers.py` | AprilTag detection, single-tag poses |
| `rigid_body.py` | multi-tag pipette: calibration (cached / online) and fused pose |
| `plate.py` | plate box, ray–box intersection, grid cell, well names |
| `geometry.py` | Aria camera model helpers |
| `press_classifier.py` | plunger crop + classifier, online press → dispense events |
| `volume_reader.py` | volume counter reading (also standalone CLI for `--collect`) |
| `experiment_state.py` | well-plate state: dispenses, volumes per well |
| `report.py` | running report, rendered to `report.md` / `report.json` |
| `vlm_reporter.py` + `prompts/report_update.txt` | Gemini call with frame + objects + plate state |
| `visualisation.py` | debug overlays (`annotated/`) |

Outputs go to `outputs/<recording>/`: `report.md`, `report.json`, `frames.jsonl`,
`frames/`, `annotated/`.
