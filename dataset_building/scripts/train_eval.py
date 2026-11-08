"""
Train a pressed / not-pressed classifier on frozen pretrained features and evaluate it.

Install:  pip install torch torchvision scikit-learn matplotlib pandas joblib
Usage:    python train_eval.py [--backbone dinov2_vits14|resnet18|resnet50|pixels] [--curve]
                               [--train-sessions S ...] [--test-sessions S ...] [--run-name NAME]
          e.g. compare p1000-only vs mixed training, both scored on the p1000 test frames:
            python train_eval.py --train-sessions p1000_154354 --test-sessions p1000_154354 --run-name p1000_only
            python train_eval.py --test-sessions p1000_154354 --run-name mixed

Pipeline
  1. Image features for every crop that is used (cached in data/features/<backbone>.npz).
  2. StandardScaler + logistic regression (class-balanced) on the train_pick frames;
     the regularisation strength is chosen by cross-validation grouped by EVENT, so
     near-identical frames of one press never sit on both sides of a fold.
  3. Test evaluation:
       frame level  - usable test frames, raw and temporally smoothed probabilities
       event level  - predicted press events (runs of smoothed prob >= threshold) matched
                      against labelled press events: recall, precision, onset error
  4. --curve: learning curve over the number of training events per class.

Output: data/results/<run name or backbone>/ metrics.json (incl. per-session), predictions.csv, model.joblib,
        timeline_<session>.png, learning_curve.csv (with --curve)
"""
import argparse
import json
from collections import defaultdict

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C

# ----------------------------------------------------------------------------- SETTINGS
INPUT_HW = (336, 224)          # model input (height, width); keeps the crops' 3:2 aspect
BATCH_SIZE = 32
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
THRESHOLD = 0.5                # probability threshold for "pressed"
SMOOTH_WINDOW = 5              # median filter length in frames (odd; 1 = off)
MIN_EVENT_FRAMES = 3           # shorter predicted press runs are discarded
CURVE_EVENTS = [3, 5, 10, 20, 40]   # training events per class for the learning curve
CURVE_REPEATS = 5
SEED = 0
# ----------------------------------------------------------------------------- /SETTINGS


# ----------------------------------------------------------------------------- features
def load_backbone(name):
    if name == "pixels":
        return None, None
    import torch
    if name.startswith("dinov2"):
        model = torch.hub.load("facebookresearch/dinov2", name)
    elif name in ("resnet18", "resnet50"):
        import torchvision
        model = getattr(torchvision.models, name)(weights="DEFAULT")
        model.fc = torch.nn.Identity()
    else:
        raise SystemExit(f"Unknown backbone '{name}'")
    model.eval()
    torch.set_grad_enabled(False)
    return model, torch


def preprocess(img_bgr):
    h, w = INPUT_HW
    img = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return img.transpose(2, 0, 1).astype(np.float32)


def pixel_features(img_bgr):
    """Tiny raw-pixel baseline (no deep learning): 24 x 36 grey image, normalised."""
    g = cv2.cvtColor(cv2.resize(img_bgr, (24, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32).ravel()
    return (g - g.mean()) / (g.std() + 1e-6)


def get_features(paths, backbone):
    """Features for the given crop paths (relative to data/), using and updating a cache.
    A cached entry is recomputed if its crop file changed since (e.g. re-extracted
    with a different crop box)."""
    C.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = C.FEATURES_DIR / f"{backbone}_{INPUT_HW[0]}x{INPUT_HW[1]}.npz"
    cache, mtimes = {}, {}
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        cache = dict(zip(z["paths"].tolist(), z["feats"]))
        if "mtimes" in z:
            mtimes = dict(zip(z["paths"].tolist(), z["mtimes"].tolist()))
    current = {p: (C.DATA_DIR / p).stat().st_mtime for p in dict.fromkeys(paths)}
    stale = [p for p in current if p in cache and mtimes.get(p) != current[p]]
    for p in stale:
        del cache[p]
    if stale:
        print(f"{len(stale)} crops changed since their features were cached; recomputing")
    missing = [p for p in current if p not in cache]
    if missing:
        model, torch = load_backbone(backbone)
        from tqdm import tqdm
        for s in tqdm(range(0, len(missing), BATCH_SIZE), desc=f"features ({backbone})"):
            batch = missing[s:s + BATCH_SIZE]
            imgs = [cv2.imread(str(C.DATA_DIR / p)) for p in batch]
            if model is None:
                feats = np.stack([pixel_features(im) for im in imgs])
            else:
                x = torch.from_numpy(np.stack([preprocess(im) for im in imgs]))
                feats = model(x).cpu().numpy()
            cache.update(zip(batch, feats))
            mtimes.update((p, current[p]) for p in batch)
        np.savez(cache_file, paths=np.array(list(cache), dtype=object),
                 feats=np.stack(list(cache.values())),
                 mtimes=np.array([mtimes.get(p, -1.0) for p in cache], dtype=np.float64))
    return np.stack([cache[p] for p in paths])


# ----------------------------------------------------------------------------- model
def fit(X, y, groups, seed=SEED):
    best_c, best_score = 1.0, -1
    n_groups = len(set(groups))
    if n_groups >= 4 and len(set(y)) == 2:
        cv = GroupKFold(n_splits=min(5, n_groups))
        for c in C_GRID:
            scores = []
            for tr, va in cv.split(X, y, groups):
                if len(set(y[tr])) < 2 or len(set(y[va])) < 2:
                    continue
                m = make_pipeline(StandardScaler(), LogisticRegression(
                    C=c, class_weight="balanced", max_iter=5000, random_state=seed))
                m.fit(X[tr], y[tr])
                scores.append(balanced_accuracy_score(y[va], m.predict(X[va])))
            if scores and np.mean(scores) > best_score:
                best_c, best_score = c, float(np.mean(scores))
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=best_c, class_weight="balanced", max_iter=5000, random_state=seed))
    model.fit(X, y)
    return model, best_c, best_score


# ----------------------------------------------------------------------------- evaluation
def smooth(p):
    """Median filter that ignores NaN (frames without a crop); NaNs are filled from neighbours."""
    if SMOOTH_WINDOW <= 1:
        out = p.copy()
    else:
        h = SMOOTH_WINDOW // 2
        padded = np.pad(p, h, constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(padded, SMOOTH_WINDOW)
        with np.errstate(all="ignore"):
            out = np.nanmedian(windows, axis=1) if np.isfinite(p).any() else p.copy()
    s = pd.Series(out).ffill().bfill()
    return s.fillna(0.0).to_numpy()


def runs(mask):
    """[(start, end_exclusive)] of True runs."""
    out, start = [], None
    for k, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = k
        elif not v and start is not None:
            out.append((start, k))
            start = None
    return out


def frame_metrics(y, p):
    pred = (p >= THRESHOLD).astype(int)
    m = {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_pressed": float(precision_score(y, pred, zero_division=0)),
        "recall_pressed": float(recall_score(y, pred, zero_division=0)),
        "f1_pressed": float(f1_score(y, pred, zero_division=0)),
        "confusion[[tn,fp],[fn,tp]]": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    if len(set(y)) == 2:
        m["roc_auc"] = float(roc_auc_score(y, p))
    return m


def event_metrics(seq):
    """seq: one session's test frames in order, with columns label, p_smooth."""
    gt = runs((seq["label"] == "pressed").to_numpy())
    pred = [r for r in runs((seq["p_smooth"] >= THRESHOLD).to_numpy())
            if r[1] - r[0] >= MIN_EVENT_FRAMES]
    used, onset = set(), []
    for gs, ge in gt:
        for k, (ps, pe) in enumerate(pred):
            if k not in used and ps < ge and pe > gs:
                used.add(k)
                onset.append(ps - gs)
                break
    return len(gt), len(pred), len(used), onset


def plot_timeline(seq, session, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = seq["frame_idx"].to_numpy()
    fig, ax = plt.subplots(figsize=(14, 3.2))
    colours = {"pressed": "tab:green", "not_pressed": "tab:blue", "ignore": "gold"}
    for lab, col in colours.items():
        for s, e in runs((seq["label"] == lab).to_numpy()):
            ax.axvspan(x[s], x[e - 1] + 1, ymin=0.92, ymax=1.0, color=col, lw=0)
    ax.plot(x, seq["p"], ".", ms=2, color="0.6", label="p(pressed) per frame")
    ax.plot(x, seq["p_smooth"], color="tab:red", lw=1.2, label=f"smoothed ({SMOOTH_WINDOW})")
    ax.axhline(THRESHOLD, color="k", lw=0.5, ls="--")
    ax.set_ylim(-0.02, 1.1)
    ax.set_xlabel("frame index")
    ax.set_title(f"{session}: top band = labels (green pressed, blue not pressed, gold ignore)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate(model, df_test, X_by_path):
    df = df_test.copy()
    has = df["crop"] != ""
    df["p"] = np.nan
    if has.any():
        df.loc[has, "p"] = model.predict_proba(X_by_path(df.loc[has, "crop"].tolist()))[:, 1]
    df["p_smooth"] = np.nan
    for s, idx in df.groupby("session").groups.items():
        seq = df.loc[idx].sort_values("frame_idx")
        df.loc[seq.index, "p_smooth"] = smooth(seq["p"].to_numpy())
    use = df[(df["usable"] == 1)]
    metrics = {"frame_raw": frame_metrics(use["target"].to_numpy(), use["p"].to_numpy()),
               "frame_smoothed": frame_metrics(use["target"].to_numpy(), use["p_smooth"].to_numpy())}
    def summarise(g, p, m, o):
        return {"labelled_press_events": g, "predicted_press_events": p, "matched": m,
                "event_recall": m / g if g else None,
                "event_precision": m / p if p else None,
                "median_onset_error_frames": float(np.median(o)) if o else None}

    n_gt = n_pred = n_match = 0
    onsets = []
    metrics["per_session"] = {}
    for s, idx in df.groupby("session").groups.items():
        g, p, m, o = event_metrics(df.loc[idx].sort_values("frame_idx"))
        n_gt, n_pred, n_match = n_gt + g, n_pred + p, n_match + m
        onsets += o
        u = use[use["session"] == s]
        metrics["per_session"][s] = {
            "frame_smoothed": (frame_metrics(u["target"].to_numpy(), u["p_smooth"].to_numpy())
                               if len(u) else None),
            "events": summarise(g, p, m, o),
        }
    metrics["events"] = summarise(n_gt, n_pred, n_match, onsets)
    return metrics, df


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--curve", action="store_true", help="also compute a learning curve")
    ap.add_argument("--train-sessions", nargs="+", metavar="SESSION",
                    help="train only on these sessions' train frames (default: all)")
    ap.add_argument("--test-sessions", nargs="+", metavar="SESSION",
                    help="evaluate only on these sessions' test frames (default: all)")
    ap.add_argument("--run-name", help="results folder name (default: the backbone name)")
    args = ap.parse_args()

    ds_path = C.DATASET_DIR / "dataset.csv"
    if not ds_path.exists():
        raise SystemExit(f"{ds_path} not found. Run build_dataset.py first.")
    df = pd.read_csv(ds_path, keep_default_na=False)
    train = df[(df["split"] == "train") & (df["train_pick"] == 1)]
    test = df[df["split"] == "test"]
    for flag, sel, part in (("--train-sessions", args.train_sessions, "train"),
                            ("--test-sessions", args.test_sessions, "test")):
        if sel:
            unknown = set(sel) - set(df["session"])
            if unknown:
                raise SystemExit(f"{flag}: not in dataset.csv: {sorted(unknown)}. "
                                 f"Known: {sorted(set(df['session']))}")
            if part == "train":
                train = train[train["session"].isin(sel)]
            else:
                test = test[test["session"].isin(sel)]
    if train.empty or test.empty:
        raise SystemExit("Need both train and test frames - check the split settings in build_dataset.py")

    needed = list(dict.fromkeys(train["crop"].tolist() + test.loc[test["crop"] != "", "crop"].tolist()))
    feats = dict(zip(needed, get_features(needed, args.backbone)))
    X_by_path = lambda paths: np.stack([feats[p] for p in paths])

    X, y, groups = X_by_path(train["crop"].tolist()), train["target"].to_numpy(), train["event_id"].to_numpy()
    model, best_c, cv_score = fit(X, y, groups)
    print(f"Trained on {len(y)} frames from {len(set(groups))} events "
          f"(pressed {int(y.sum())}, not pressed {int((1 - y).sum())}); C={best_c}"
          + (f", grouped-CV balanced acc {cv_score:.3f}" if cv_score >= 0 else ""))

    metrics, pred_df = evaluate(model, test, X_by_path)
    metrics["train"] = {"frames": int(len(y)), "events": int(len(set(groups))), "C": best_c,
                        "cv_balanced_accuracy": cv_score if cv_score >= 0 else None}
    metrics["settings"] = {"backbone": args.backbone, "threshold": THRESHOLD,
                           "smooth_window": SMOOTH_WINDOW, "min_event_frames": MIN_EVENT_FRAMES,
                           "train_sessions": sorted(set(train["session"])),
                           "test_sessions": sorted(set(test["session"]))}

    out = C.RESULTS_DIR / (args.run_name or args.backbone)
    out.mkdir(parents=True, exist_ok=True)
    for s, idx in pred_df.groupby("session").groups.items():
        plot_timeline(pred_df.loc[idx].sort_values("frame_idx"), s, out / f"timeline_{s}.png")
    pred_df[["session", "frame_idx", "label", "target", "usable", "event_id", "p", "p_smooth"]] \
        .to_csv(out / "predictions.csv", index=False)
    joblib.dump({"model": model, "backbone": args.backbone, "input_hw": INPUT_HW,
                 "threshold": THRESHOLD}, out / "model.joblib")

    fr, fs, ev = metrics["frame_raw"], metrics["frame_smoothed"], metrics["events"]
    print(f"\nTest frames (usable): {fr['n']}")
    print(f"  raw      : bal.acc {fr['balanced_accuracy']:.3f}  F1 {fr['f1_pressed']:.3f}  "
          f"P {fr['precision_pressed']:.3f}  R {fr['recall_pressed']:.3f}")
    print(f"  smoothed : bal.acc {fs['balanced_accuracy']:.3f}  F1 {fs['f1_pressed']:.3f}  "
          f"P {fs['precision_pressed']:.3f}  R {fs['recall_pressed']:.3f}")
    print(f"Events: {ev['matched']}/{ev['labelled_press_events']} presses found, "
          f"{ev['predicted_press_events']} predicted"
          + (f", median onset error {ev['median_onset_error_frames']:+.0f} frames"
             if ev["median_onset_error_frames"] is not None else ""))
    if len(metrics["per_session"]) > 1:
        print("Per test session (smoothed frames | events):")
        for s, m in metrics["per_session"].items():
            f_, e = m["frame_smoothed"], m["events"]
            fr_txt = (f"n {f_['n']:>5}  bal.acc {f_['balanced_accuracy']:.3f}  F1 {f_['f1_pressed']:.3f}"
                      if f_ else "no usable frames")
            print(f"  {s:<28} {fr_txt}  | {e['matched']}/{e['labelled_press_events']} found, "
                  f"{e['predicted_press_events']} predicted")

    if args.curve:
        rng = np.random.default_rng(SEED)
        ev_by_cls = defaultdict(list)
        for e, t in train.groupby("event_id")["target"].first().items():
            ev_by_cls[t].append(e)
        rows = []
        for n in CURVE_EVENTS:
            if n > min(len(v) for v in ev_by_cls.values()):
                break
            for rep in range(CURVE_REPEATS):
                chosen = set(np.concatenate([rng.choice(v, n, replace=False) for v in ev_by_cls.values()]))
                sub = train[train["event_id"].isin(chosen)]
                m, _, _ = fit(X_by_path(sub["crop"].tolist()), sub["target"].to_numpy(),
                              sub["event_id"].to_numpy(), seed=SEED + rep)
                met, _ = evaluate(m, test, X_by_path)
                rows.append({"events_per_class": n, "repeat": rep,
                             "bal_acc_smoothed": met["frame_smoothed"]["balanced_accuracy"],
                             "f1_smoothed": met["frame_smoothed"]["f1_pressed"],
                             "event_recall": met["events"]["event_recall"],
                             "event_precision": met["events"]["event_precision"]})
        curve = pd.DataFrame(rows)
        curve.to_csv(out / "learning_curve.csv", index=False)
        if not curve.empty:
            print("\nLearning curve (mean over repeats):")
            print(curve.groupby("events_per_class").mean(numeric_only=True)
                  .drop(columns="repeat").round(3).to_string())
        metrics["learning_curve"] = rows

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()