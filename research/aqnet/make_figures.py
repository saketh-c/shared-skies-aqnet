"""CLI orchestrator for the AQNet publication figures.

Discovers every fig_*.py module in this directory (except fig_style, the
shared style contract), collects their figure functions (any module-level
callable named F<number>_<slug> taking (mode, preview)), runs the requested
subset, and prints a summary table of what was rendered or skipped and why.

Usage (from the repo root):
    python research/aqnet/make_figures.py                # everything
    python research/aqnet/make_figures.py --list
    python research/aqnet/make_figures.py --fig F17 F19_surfaces
    python research/aqnet/make_figures.py --mode both --final

Preview vs final: quick-mode artifacts (a smoke run) must never produce
un-watermarked figures. When the artifacts carry a quick-mode marker
(metrics_loso.json "quick": true, or a quick marker in SUMMARY.md) the
default is --preview (watermarked, written to figures/preview_quick/).
When no marker can be found, an explicit --preview or --final is required.

Exit status: nonzero only when a figure (or module import) raised an
unexpected exception; graceful skips (a figure returning None after
printing a "[skip] ..." reason) exit zero.
"""
import argparse
import contextlib
import glob
import importlib
import inspect
import io
import json
import os
import re
import sys
import traceback

# ── aqnet sys.path bootstrap (aqnet + deep-learning track), Colab-safe ──────

_AQNET_DIR = os.path.dirname(os.path.abspath(__file__))
_DL_DIR = os.path.join(os.path.dirname(_AQNET_DIR), "deeplearning")
for _p in (_DL_DIR, _AQNET_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")

import config

_FIG_FUNC_RE = re.compile(r"^F\d+")


# ── Discovery ───────────────────────────────────────────────────────────────

def discover_modules():
    """Import every fig_*.py beside this script (fig_style excluded).

    Returns (modules, import_errors): {name: module} and
    [(name, traceback_str)].
    """
    modules, errors = {}, []
    for path in sorted(glob.glob(os.path.join(_AQNET_DIR, "fig_*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name == "fig_style":
            continue
        try:
            modules[name] = importlib.import_module(name)
        except Exception:
            errors.append((name, traceback.format_exc()))
    return modules, errors


def _canonical_name(fn):
    """Best figure name for a bare callable from a FIGURES tuple/list.

    Prefers the F<number>_<slug> save-name assigned inside the function body
    (e.g. ``name = "F05_study_domain"``), falling back to __name__.
    """
    try:
        match = re.search(r"""name\s*=\s*["'](F\d+\w*)["']""",
                          inspect.getsource(fn))
        if match:
            return match.group(1)
    except (OSError, TypeError):
        pass
    return getattr(fn, "__name__", str(fn))


def discover_figures(modules):
    """Ordered {figure_name: (module_name, callable)} across all modules.

    Two conventions are accepted per module:
      * module-level functions named F<number>_<slug>;
      * a module-level FIGURES registry — a {name: callable} dict, or a
        tuple/list of callables (named by their __name__).
    Duplicate names (or the same callable listed twice) keep the first hit.
    """
    registry = {}
    seen_fns = set()

    def add(name, mod_name, fn):
        if id(fn) in seen_fns:
            return
        if name in registry:
            if registry[name][1] is not fn:
                print(f"warning: duplicate figure {name} in {mod_name} "
                      f"(keeping the one from {registry[name][0]})")
            return
        registry[name] = (mod_name, fn)
        seen_fns.add(id(fn))

    for mod_name, mod in modules.items():
        for attr, obj in sorted(vars(mod).items()):
            if (_FIG_FUNC_RE.match(attr) and callable(obj)
                    and getattr(obj, "__module__", None) == mod_name):
                add(attr, mod_name, obj)
        figures = getattr(mod, "FIGURES", None)
        if isinstance(figures, dict):
            for name, fn in figures.items():
                if callable(fn):
                    add(str(name), mod_name, fn)
        elif isinstance(figures, (list, tuple)):
            for fn in figures:
                if callable(fn):
                    add(_canonical_name(fn), mod_name, fn)

    def sort_key(name):
        digits = re.match(r"F(\d+)", name)
        return (int(digits.group(1)) if digits else 10 ** 6, name)

    return {k: registry[k] for k in sorted(registry, key=sort_key)}


def resolve_requested(registry, requested):
    """Map user tokens (F17 / f17_importance / F17_importance) to names."""
    resolved, unknown = [], []
    lower = {name.lower(): name for name in registry}
    prefix = {}
    for name in registry:
        prefix.setdefault(name.split("_")[0].lower(), []).append(name)
    for token in requested:
        t = token.lower()
        if t in lower:
            resolved.append(lower[t])
        elif t in prefix:
            resolved.extend(prefix[t])
        else:
            unknown.append(token)
    seen, ordered = set(), []
    for name in resolved:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered, unknown


# ── Quick-mode detection ────────────────────────────────────────────────────

def detect_quick_mode():
    """True/False when a marker states quick-ness, None when undeterminable."""
    loso = config.artifact("metrics_loso.json")
    if os.path.exists(loso):
        try:
            with open(loso, encoding="utf-8") as f:
                marker = json.load(f).get("quick")
            if marker is not None:
                return bool(marker)
        except (OSError, ValueError):
            pass
    summary = config.artifact("SUMMARY.md")
    if os.path.exists(summary):
        try:
            with open(summary, encoding="utf-8", errors="replace") as f:
                text = f.read()
            if re.search(r"`?quick`?\s*\|?\s*(=|:)?\s*True", text):
                return True
        except OSError:
            pass
    return None


# ── Execution ───────────────────────────────────────────────────────────────

def _call_figure(fn, mode, preview):
    """Call fn with only the keyword args its signature accepts."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                     for p in params.values())
    kwargs = {}
    if "mode" in params or has_var_kw:
        kwargs["mode"] = mode
    if "preview" in params or has_var_kw:
        kwargs["preview"] = preview
    return fn(**kwargs)


def _finalize_result(out, name, mode, preview):
    """Normalize a figure function's return value into a list of paths.

    Functions may return the written paths (they saved via fig_style
    themselves), a bare matplotlib Figure (the orchestrator saves it under
    the registry name, honoring the preview flag), or None (graceful skip).
    """
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    import fig_style
    if isinstance(out, Figure):
        paths = fig_style.save_fig(out, name, mode, preview=preview)
        plt.close(out)
        return paths
    if isinstance(out, (str, os.PathLike)):
        return [str(out)]
    if out:
        return [str(p) for p in out]
    return None


def run_figures(registry, names, modes, preview):
    """Run each figure in each mode; returns result rows for the table."""
    rows = []
    for name in names:
        mod_name, fn = registry[name]
        for mode in modes:
            print(f"\n=== {name} [{mode}{', preview' if preview else ''}] "
                  f"({mod_name}) ===")
            buf = io.StringIO()
            status, detail = "rendered", ""
            try:
                with contextlib.redirect_stdout(buf):
                    out = _call_figure(fn, mode, preview)
                    out = _finalize_result(out, name, mode, preview)
            except Exception as exc:
                captured = buf.getvalue()
                if captured:
                    print(captured, end="")
                traceback.print_exc()
                rows.append({"figure": name, "mode": mode, "status": "ERROR",
                             "detail": f"{type(exc).__name__}: {exc}",
                             "paths": []})
                continue
            captured = buf.getvalue()
            if captured:
                print(captured, end="")
            if not out:
                status = "skipped"
                skip_lines = [ln.strip() for ln in captured.splitlines()
                              if "skip" in ln.lower()]
                detail = skip_lines[-1] if skip_lines else "returned no paths"
            else:
                detail = f"{len(out)} file(s)"
                for p in out:
                    print(f"  wrote {p}")
            rows.append({"figure": name, "mode": mode, "status": status,
                         "detail": detail, "paths": out or []})
    return rows


def print_table(rows):
    headers = ("Figure", "Mode", "Status", "Details")
    data = [(r["figure"], r["mode"], r["status"], r["detail"]) for r in rows]
    widths = [max(len(h), *(len(d[i]) for d in data)) if data else len(h)
              for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print("\n" + line)
    print("-" * len(line))
    for d in data:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(d)))
    n_ok = sum(r["status"] == "rendered" for r in rows)
    n_skip = sum(r["status"] == "skipped" for r in rows)
    n_err = sum(r["status"] == "ERROR" for r in rows)
    print(f"\n{n_ok} rendered, {n_skip} skipped, {n_err} errored.")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Render AQNet publication figures.",
        epilog="Figure names may be given in full (F17_importance) or by "
               "number (F17). Without --fig, every discovered figure runs.")
    ap.add_argument("--fig", nargs="+", default=None, metavar="FIG",
                    help="subset of figures to render")
    ap.add_argument("--mode", choices=("paper", "poster", "both"),
                    default="paper", help="output styling (default: paper)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--preview", action="store_true",
                     help="watermarked previews in figures/preview_quick/")
    grp.add_argument("--final", action="store_true",
                     help="publication outputs in figures/ (no watermark)")
    ap.add_argument("--list", action="store_true",
                    help="list discovered figures and exit")
    args = ap.parse_args(argv)

    modules, import_errors = discover_modules()
    registry = discover_figures(modules)
    for name, tb in import_errors:
        print(f"ERROR importing {name}.py:\n{tb}")

    if args.list:
        print("Discovered figures:")
        for name, (mod_name, fn) in registry.items():
            doc = (inspect.getdoc(fn) or "").splitlines()
            first = doc[0] if doc else ""
            print(f"  {name:<24s} [{mod_name}] {first}")
        if not registry:
            print("  (none)")
        return 1 if import_errors else 0

    # Preview/final resolution: explicit flag > quick-mode marker > refuse.
    if args.preview:
        preview = True
    elif args.final:
        preview = False
    else:
        quick = detect_quick_mode()
        if quick:
            preview = True
            print("Quick-mode artifacts detected (metrics_loso.json/"
                  "SUMMARY.md marker) -> defaulting to --preview "
                  "(watermarked).")
        else:
            ap.error("could not confirm a quick-mode marker in the "
                     "artifacts; pass --preview or --final explicitly")
            return 2  # unreachable (ap.error exits), kept for clarity

    if args.fig:
        names, unknown = resolve_requested(registry, args.fig)
        if unknown:
            ap.error(f"unknown figure(s): {', '.join(unknown)} "
                     f"(use --list to see what is available)")
        if not names:
            ap.error("no figures matched the --fig selection")
    else:
        names = list(registry)
    if not names:
        print("No figure functions discovered.")
        return 1 if import_errors else 0

    modes = ["paper", "poster"] if args.mode == "both" else [args.mode]
    rows = run_figures(registry, names, modes, preview)
    print_table(rows)

    failed = bool(import_errors) or any(r["status"] == "ERROR" for r in rows)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
