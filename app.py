import os
import json
import time
from datetime import datetime

import gradio as gr
import requests

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "").rstrip("/")
RESULTS_PATH = "bulk_results_export.json"


# 1.1 FastAPI Helpers
def api_health():
    if not FASTAPI_BASE_URL:
        return {"status": "error", "message": "FASTAPI_BASE_URL is not set."}

    try:
        r = requests.get(f"{FASTAPI_BASE_URL}/health", timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def api_get_taxonomy():
    if not FASTAPI_BASE_URL:
        print("FASTAPI_BASE_URL is not set.")
        return [], [], [], [], [], []

    try:
        r = requests.get(f"{FASTAPI_BASE_URL}/taxonomy", timeout=20)
        r.raise_for_status()
        data = r.json()

        return (
            data.get("categories", []),
            data.get("subcategories", []),
            data.get("colours", []),
            data.get("patterns", []),
            data.get("fits", []),
            data.get("features", []),
        )
    except Exception as e:
        print(f"Failed to fetch taxonomy from FastAPI: {e}")
        return [], [], [], [], [], []


def api_predict_image(image_path):
    if not FASTAPI_BASE_URL:
        return {"_warning": "FASTAPI_BASE_URL is not set."}

    if not image_path or not os.path.exists(image_path):
        return {"_warning": "Image file not found."}

    try:
        with open(image_path, "rb") as f:
            files = {
                "file": (os.path.basename(image_path), f, "application/octet-stream")
            }
            r = requests.post(f"{FASTAPI_BASE_URL}/predict", files=files, timeout=180)

        if r.status_code != 200:
            return {"_warning": f"FastAPI predict failed: {r.status_code} - {r.text}"}

        data = r.json()
        pred = (data or {}).get("prediction", {}) or {}

        return {
            "category": pred.get("category"),
            "subcategory": pred.get("subcategory"),
            "colour": pred.get("colour"),
            "pattern": pred.get("pattern"),
            "fit": pred.get("fit"),
            "features": pred.get("features", []),
            "_warning": pred.get("warning"),
        }
    except Exception as e:
        return {"_warning": f"FastAPI predict failed: {str(e)}"}


def build_local_export_file(state):
    if not state or not state.get("items"):
        return None, "<div class='muted'>Nothing to export.</div>"

    payload_items = []

    for it in state["items"]:
        final = it.get("final") or it.get("pred")
        if not final:
            continue

        payload_items.append(
            {
                "timestamp": it.get("timestamp"),
                "image_name": it.get("image_name"),
                "image_path": it.get("image_path"),
                "edited": bool(it.get("edited")),
                "prediction": {
                    "category": final.get("category"),
                    "subcategory": final.get("subcategory"),
                    "colour": final.get("colour"),
                    "pattern": final.get("pattern"),
                    "fit": final.get("fit"),
                    "features": final.get("features", []),
                    "warning": final.get("_warning") or it.get("warning"),
                },
            }
        )

    if not payload_items:
        return None, "<div class='muted'>Nothing valid to export.</div>"

    export_payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "backend_base_url": FASTAPI_BASE_URL,
        "items": payload_items,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, ensure_ascii=False, indent=2)

    return RESULTS_PATH, "<div class='ok'>Reviewed JSON exported locally.</div>"


# 1.2 Taxonomy Load
category_labels, subcategory_labels, colour_labels, pattern_labels, fit_labels, feature_labels = api_get_taxonomy()


# 2.1 Bulk State Helpers
def _new_item(image_path, image_name):
    return {
        "image_path": image_path,
        "image_name": image_name,
        "pred": None,
        "final": None,
        "edited": False,
        "warning": None,
        "timestamp": None,
    }


def _safe_dd_value(v, choices):
    if v is None:
        return None
    v = str(v).strip()
    return v if v in (choices or []) else None


def _safe_cb_values(vals, choices):
    if vals is None:
        return []
    if isinstance(vals, str):
        vals = [vals]
    out = []
    for x in vals:
        if isinstance(x, str) and x in (choices or []):
            out.append(x)
    return list(dict.fromkeys(out))


def _gallery_from_state(state):
    items = (state or {}).get("items", []) or []
    return [(it["image_path"], it.get("image_name") or "image") for it in items]


def _index_choices(state):
    items = (state or {}).get("items", []) or []
    return [f"{i + 1}. {it.get('image_name', 'image')}" for i, it in enumerate(items)]


def _get_selected_item(state, sel):
    items = (state or {}).get("items", []) or []
    if not items:
        return None, None

    if not sel:
        return 0, items[0]

    try:
        idx = int(sel.split(".")[0].strip()) - 1
        idx = max(0, min(len(items) - 1, idx))
        return idx, items[idx]
    except Exception:
        return 0, items[0]


def _summary_text(item):
    if not item:
        return "<div class='muted'>No item selected.</div>"

    warn = item.get("warning")
    final = item.get("final") or item.get("pred")

    if warn and not final:
        return f"<div class='warn'>{warn}</div>"

    if not final:
        return "<div class='muted'>Not processed yet.</div>"

    warn_html = f"<div class='warn' style='margin-bottom:10px'>{warn}</div>" if warn else ""

    return warn_html + f"""
    <div class="kv-wrap">
      <div class="kv"><div class="k">Image Name</div><div class="v">{item.get('image_name','')}</div></div>
      <div class="kv"><div class="k">Category</div><div class="v">{final.get('category','')}</div></div>
      <div class="kv"><div class="k">Subcategory</div><div class="v">{final.get('subcategory','')}</div></div>
      <div class="kv"><div class="k">Colour</div><div class="v">{final.get('colour','')}</div></div>
      <div class="kv"><div class="k">Pattern</div><div class="v">{final.get('pattern','')}</div></div>
      <div class="kv"><div class="k">Fit</div><div class="v">{final.get('fit','')}</div></div>
    </div>
    """


def _features_html(item):
    if not item:
        return "<div class='muted'>No item selected.</div>"

    final = item.get("final") or item.get("pred")
    if not final:
        return "<div class='muted'>Not processed yet.</div>"

    feats = final.get("features") or []
    if not feats:
        return "<div class='muted'>No features.</div>"

    feats = [x for x in feats if isinstance(x, str)]
    feats = list(dict.fromkeys(feats))
    chips = "".join([f"<span class='chip'>{x}</span>" for x in feats])
    return f"<div class='chip-wrap'>{chips}</div>"


# 2.2 Overlay Control
def _overlay(show, text="Loading...", percent=0):
    if show:
        return gr.update(
            visible=True,
            value=f"""
            <div class='overlay-inner'>
                <div class='spinner'></div>
                <div class='overlay-text'>{text}</div>
                <div class='progress-wrap'>
                    <div class='progress-bar'>
                        <div class='progress-fill' style='width:{percent}%;'></div>
                    </div>
                    <div class='progress-percent'>{percent}%</div>
                </div>
            </div>
            """
        )
    return gr.update(visible=False, value="")


# 3.1 Core Actions
def on_upload(files):
    st = {
        "items": [],
        "created_at": datetime.utcnow().isoformat() + "Z",
        "next_idx": 0,
    }

    if not files:
        return (
            st,
            [],
            gr.update(choices=[], value=None),
            None,
            "<div class='muted'>Upload images to begin.</div>",
            "<div class='muted'>No features.</div>",
            "<div class='muted'></div>",
            gr.update(visible=False),
            gr.update(visible=False),
            _overlay(False),
            gr.update(visible=False),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=[]),
        )

    for f in files:
        path = getattr(f, "name", None) or str(f)
        name = os.path.basename(path)
        if path:
            st["items"].append(_new_item(path, name))

    gallery = _gallery_from_state(st)
    choices = _index_choices(st)

    first = st["items"][0] if st["items"] else None
    preview_path = first["image_path"] if first else None
    first_final = (first.get("final") or first.get("pred") or {}) if first else {}

    return (
        st,
        gallery,
        gr.update(choices=choices, value=(choices[0] if choices else None)),
        preview_path,
        _summary_text(first),
        _features_html(first),
        "<div class='muted'>Ready. Click “Process Next” or “Process All”.</div>",
        gr.update(visible=True),
        gr.update(visible=True),
        _overlay(False),
        gr.update(visible=True),
        gr.update(value=_safe_dd_value(first_final.get("category"), category_labels)),
        gr.update(value=_safe_dd_value(first_final.get("subcategory"), subcategory_labels)),
        gr.update(value=_safe_dd_value(first_final.get("colour"), colour_labels)),
        gr.update(value=_safe_dd_value(first_final.get("pattern"), pattern_labels)),
        gr.update(value=_safe_dd_value(first_final.get("fit"), fit_labels)),
        gr.update(value=_safe_cb_values(first_final.get("features", []), feature_labels)),
    )


def process_next(state, sel):
    if not state or not state.get("items"):
        return (
            state,
            _gallery_from_state(state),
            gr.update(choices=_index_choices(state), value=None),
            None,
            "<div class='muted'>Upload images first.</div>",
            "<div class='muted'>No features.</div>",
            "<div class='muted'></div>",
            _overlay(False),
            gr.update(visible=False),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=[]),
        )

    items = state["items"]
    idx = int(state.get("next_idx", 0))
    total = len(items)

    if idx >= total:
        _, it_sel = _get_selected_item(state, sel)
        preview_path = it_sel["image_path"] if it_sel else None
        final = (it_sel.get("final") or it_sel.get("pred") or {}) if it_sel else {}

        return (
            state,
            _gallery_from_state(state),
            gr.update(choices=_index_choices(state), value=sel),
            preview_path,
            _summary_text(it_sel),
            _features_html(it_sel),
            "<div class='ok'>All images processed.</div>",
            _overlay(False),
            gr.update(visible=True),
            gr.update(value=_safe_dd_value(final.get("category"), category_labels)),
            gr.update(value=_safe_dd_value(final.get("subcategory"), subcategory_labels)),
            gr.update(value=_safe_dd_value(final.get("colour"), colour_labels)),
            gr.update(value=_safe_dd_value(final.get("pattern"), pattern_labels)),
            gr.update(value=_safe_dd_value(final.get("fit"), fit_labels)),
            gr.update(value=_safe_cb_values(final.get("features", []), feature_labels)),
        )

    percent_before = int((idx / total) * 100)

    yield (
        state,
        _gallery_from_state(state),
        gr.update(choices=_index_choices(state), value=sel),
        None,
        "<div class='muted'>Loading...</div>",
        "<div class='muted'>Loading...</div>",
        f"<div class='ok'>Processing {idx + 1}/{total}...</div>",
        _overlay(True, f"Loading... {idx + 1}/{total}", percent_before),
        gr.update(visible=True),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=[]),
    )

    it = items[idx]
    pred = api_predict_image(it["image_path"])
    it["timestamp"] = datetime.utcnow().isoformat() + "Z"

    if pred.get("_warning"):
        it["warning"] = pred.get("_warning")
        it["pred"] = None
        it["final"] = None
        it["edited"] = False
    else:
        it["warning"] = None
        it["pred"] = pred
        it["final"] = pred
        it["edited"] = False

    items[idx] = it
    state["items"] = items
    state["next_idx"] = idx + 1

    _, it_sel = _get_selected_item(state, sel)
    preview_path = it_sel["image_path"] if it_sel else None
    final = (it_sel.get("final") or it_sel.get("pred") or {}) if it_sel else {}

    yield (
        state,
        _gallery_from_state(state),
        gr.update(choices=_index_choices(state), value=sel),
        preview_path,
        _summary_text(it_sel),
        _features_html(it_sel),
        f"<div class='ok'>Processed {idx + 1}/{total}.</div>",
        _overlay(False),
        gr.update(visible=True),
        gr.update(value=_safe_dd_value(final.get("category"), category_labels)),
        gr.update(value=_safe_dd_value(final.get("subcategory"), subcategory_labels)),
        gr.update(value=_safe_dd_value(final.get("colour"), colour_labels)),
        gr.update(value=_safe_dd_value(final.get("pattern"), pattern_labels)),
        gr.update(value=_safe_dd_value(final.get("fit"), fit_labels)),
        gr.update(value=_safe_cb_values(final.get("features", []), feature_labels)),
    )


def process_all(state, sel):
    if not state or not state.get("items"):
        yield (
            state,
            _gallery_from_state(state),
            gr.update(choices=_index_choices(state), value=None),
            None,
            "<div class='muted'>Upload images first.</div>",
            "<div class='muted'>No features.</div>",
            "<div class='warn'>No images.</div>",
            _overlay(False),
            gr.update(visible=False),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=[]),
        )
        return

    items = state["items"]
    n = len(items)

    yield (
        state,
        _gallery_from_state(state),
        gr.update(choices=_index_choices(state), value=sel),
        None,
        "<div class='muted'>Loading...</div>",
        "<div class='muted'>Loading...</div>",
        "<div class='ok'>Processing started...</div>",
        _overlay(True, f"Loading... 0/{n}", 0),
        gr.update(visible=True),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=[]),
    )

    processed_count = 0

    for i in range(n):
        it = items[i]

        if it.get("final") or it.get("pred"):
            processed_count += 1
            percent = int((processed_count / n) * 100)
            yield (
                state,
                _gallery_from_state(state),
                gr.update(choices=_index_choices(state), value=sel),
                None,
                "<div class='muted'>Loading...</div>",
                "<div class='muted'>Loading...</div>",
                f"<div class='ok'>Loading... {processed_count}/{n}</div>",
                _overlay(True, f"Loading... {processed_count}/{n}", percent),
                gr.update(visible=True),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=[]),
            )
            continue

        pred = api_predict_image(it["image_path"])
        it["timestamp"] = datetime.utcnow().isoformat() + "Z"

        if pred.get("_warning"):
            it["warning"] = pred.get("_warning")
            it["pred"] = None
            it["final"] = None
            it["edited"] = False
        else:
            it["warning"] = None
            it["pred"] = pred
            it["final"] = pred
            it["edited"] = False

        items[i] = it
        state["items"] = items
        state["next_idx"] = max(int(state.get("next_idx", 0)), i + 1)

        processed_count += 1
        percent = int((processed_count / n) * 100)

        yield (
            state,
            _gallery_from_state(state),
            gr.update(choices=_index_choices(state), value=sel),
            None,
            "<div class='muted'>Loading...</div>",
            "<div class='muted'>Loading...</div>",
            f"<div class='ok'>Loading... {processed_count}/{n}</div>",
            _overlay(True, f"Loading... {processed_count}/{n}", percent),
            gr.update(visible=True),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=[]),
        )

        time.sleep(0.03)

    choices = _index_choices(state)
    if choices and sel not in choices:
        sel = choices[0]

    _, it_sel = _get_selected_item(state, sel)
    preview_path = it_sel["image_path"] if it_sel else None
    final = (it_sel.get("final") or it_sel.get("pred") or {}) if it_sel else {}

    yield (
        state,
        _gallery_from_state(state),
        gr.update(choices=_index_choices(state), value=sel),
        preview_path,
        _summary_text(it_sel),
        _features_html(it_sel),
        "<div class='ok'>All images processed.</div>",
        _overlay(False),
        gr.update(visible=True),
        gr.update(value=_safe_dd_value(final.get("category"), category_labels)),
        gr.update(value=_safe_dd_value(final.get("subcategory"), subcategory_labels)),
        gr.update(value=_safe_dd_value(final.get("colour"), colour_labels)),
        gr.update(value=_safe_dd_value(final.get("pattern"), pattern_labels)),
        gr.update(value=_safe_dd_value(final.get("fit"), fit_labels)),
        gr.update(value=_safe_cb_values(final.get("features", []), feature_labels)),
    )


def on_select(state, sel):
    _, it = _get_selected_item(state, sel)
    if it is None:
        return (
            None,
            "<div class='muted'>No item selected.</div>",
            "<div class='muted'>No features.</div>",
            gr.update(visible=False),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=[]),
        )

    final = it.get("final") or it.get("pred") or {}

    return (
        it["image_path"],
        _summary_text(it),
        _features_html(it),
        gr.update(visible=True),
        gr.update(value=_safe_dd_value(final.get("category"), category_labels)),
        gr.update(value=_safe_dd_value(final.get("subcategory"), subcategory_labels)),
        gr.update(value=_safe_dd_value(final.get("colour"), colour_labels)),
        gr.update(value=_safe_dd_value(final.get("pattern"), pattern_labels)),
        gr.update(value=_safe_dd_value(final.get("fit"), fit_labels)),
        gr.update(value=_safe_cb_values(final.get("features", []), feature_labels)),
    )


def save_edit(state, sel, cat, subcat, colour, pattern, fit, feats):
    idx, it = _get_selected_item(state, sel)
    if it is None:
        return state, "<div class='muted'>Select an image first.</div>", "<div class='muted'>No features.</div>"

    final = {
        "category": cat,
        "subcategory": subcat,
        "colour": colour,
        "pattern": pattern,
        "fit": fit,
        "features": feats or [],
    }

    it["final"] = final
    it["edited"] = True
    it["warning"] = None
    it["timestamp"] = datetime.utcnow().isoformat() + "Z"

    state["items"][idx] = it

    return (
        state,
        "<div class='ok'>Saved edits locally.</div>" + _summary_text(it),
        _features_html(it),
    )


def export_json(state):
    return build_local_export_file(state)


# 4.1 CSS
css = """
:root{
  --bg:#0b1220;
  --card:rgba(255,255,255,0.06);
  --card2:rgba(255,255,255,0.08);
  --stroke:rgba(255,255,255,0.12);
  --text:rgba(255,255,255,0.92);
  --muted:rgba(255,255,255,0.65);
  --shadow:0 10px 30px rgba(0,0,0,0.35);
}

body, .gradio-container{
  background: radial-gradient(1200px 800px at 15% 0%, rgba(124,58,237,0.25), transparent 55%),
              radial-gradient(1200px 800px at 85% 10%, rgba(6,182,212,0.22), transparent 60%),
              radial-gradient(900px 600px at 60% 90%, rgba(34,197,94,0.18), transparent 60%),
              var(--bg) !important;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important;
}

.gr-markdown h1, .gr-markdown h1 *{
  color:#fff !important;
  -webkit-text-fill-color:#fff !important;
  font-weight:900 !important;
}

.section-title{
  color:#fff !important;
  font-weight:900 !important;
  font-size:1.05rem;
  margin: 6px 0 10px 0;
}

.desc-white, .desc-white *{
  color:#fff !important;
  -webkit-text-fill-color:#fff !important;
}

.panel{
  background: var(--card) !important;
  border: 1px solid var(--stroke) !important;
  border-radius: 18px !important;
  padding: 14px 16px !important;
  box-shadow: var(--shadow);
}

.muted{ color: var(--muted); }

.ok{
  color: rgba(255,255,255,0.95);
  background: rgba(34,197,94,0.16);
  border: 1px solid rgba(34,197,94,0.35);
  padding: 10px 12px;
  border-radius: 14px;
}

.warn{
  color: rgba(255,255,255,0.95);
  background: rgba(245,158,11,0.14);
  border: 1px solid rgba(245,158,11,0.35);
  padding: 10px 12px;
  border-radius: 14px;
}

.kv-wrap{ display:flex; flex-direction:column; gap:10px; }
.kv{
  display:flex; justify-content:space-between; align-items:center; gap:10px;
  padding:10px 12px; border-radius:14px; background: var(--card2);
  border: 1px solid var(--stroke);
}
.k{ color: var(--muted); font-weight: 800; }
.v{ color: var(--text); font-weight: 900; }

.chip-wrap{ display:flex; flex-wrap:wrap; gap:10px; }
.chip{
  display:inline-flex;
  padding:10px 12px;
  border-radius:999px;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.14);
  color:#fff;
  font-weight: 900;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container option{
  color:#000 !important;
  -webkit-text-fill-color:#000 !important;
  background:#fff !important;
}

#overlay{
  position: fixed;
  inset: 0;
  z-index: 999999 !important;
  background: rgba(8, 14, 26, 0.82);
  backdrop-filter: blur(6px);
  display: flex;
  align-items: center;
  justify-content: center;
}

.overlay-inner{
  text-align: center;
  padding: 24px 26px;
  border-radius: 18px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.14);
  box-shadow: 0 20px 60px rgba(0,0,0,0.45);
  min-width: 360px;
  max-width: 90vw;
}

.overlay-text{
  color: #fff;
  font-weight: 900;
  font-size: 20px;
  margin-top: 14px;
}

.spinner{
  width: 44px;
  height: 44px;
  border-radius: 999px;
  border: 4px solid rgba(255,255,255,0.22);
  border-top-color: rgba(255,255,255,0.9);
  margin: 0 auto;
  animation: spin 1s linear infinite;
}

.progress-wrap{
  margin-top: 18px;
  width: 320px;
  max-width: 80vw;
}

.progress-bar{
  width: 100%;
  height: 14px;
  border-radius: 999px;
  background: rgba(255,255,255,0.14);
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.12);
}

.progress-fill{
  height: 100%;
  width: 0%;
  border-radius: 999px;
  background: linear-gradient(90deg, #22c55e, #06b6d4, #7c3aed);
  transition: width 0.25s ease;
}

.progress-percent{
  margin-top: 10px;
  color: #fff;
  font-weight: 900;
  font-size: 16px;
  text-align: center;
}

@keyframes spin{
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
"""


# 5.1 UI
gr.close_all()

health_info = api_health()
health_text = (
    f"Backend status: {health_info.get('status', 'unknown')} | "
    f"Model: {health_info.get('model', 'N/A')} | "
    f"Strapi enabled: {health_info.get('strapi_enabled', False)}"
    if isinstance(health_info, dict)
    else "Backend status: unknown"
)

with gr.Blocks(title="WearNext Bulk (FastAPI)") as demo:
    gr.HTML(f"<style>{css}</style>")

    gr.Markdown("# WearNext Bulk (FastAPI)", elem_classes=["desc-white"])
    gr.Markdown(
        f"{health_text}. Upload multiple images, process them, review/edit the labels, and export the reviewed JSON locally.",
        elem_classes=["desc-white"],
    )

    state = gr.State({"items": [], "created_at": None, "next_idx": 0})

    overlay_wrap = gr.HTML("", visible=False, elem_id="overlay")

    upload_files = gr.Files(label="Upload Images (multiple)", file_types=["image"])

    with gr.Row():
        btn_load = gr.Button("Load Images")
        btn_process = gr.Button("Process Next", visible=False)
        btn_process_all = gr.Button("Process All", visible=False)

    gallery = gr.Gallery(label="Images", columns=3, height=240)
    sel = gr.Dropdown(label="Select Image", choices=[], value=None)

    with gr.Row():
        preview = gr.Image(label="Preview", type="filepath")
        with gr.Column():
            gr.HTML("<div class='section-title'>Selected Item</div>")
            summary = gr.HTML(elem_classes=["panel"])
            gr.HTML("<div class='section-title'>Features</div>")
            feats_html = gr.HTML(elem_classes=["panel"])
            status = gr.HTML()

    with gr.Accordion("Edit Selected", open=False):
        cat_dd = gr.Dropdown(choices=category_labels, label="Category")
        subcat_dd = gr.Dropdown(choices=subcategory_labels, label="Subcategory")
        col_dd = gr.Dropdown(choices=colour_labels, label="Colour")
        pat_dd = gr.Dropdown(choices=pattern_labels, label="Pattern")
        fit_dd = gr.Dropdown(choices=fit_labels, label="Fit")
        feats_cb = gr.CheckboxGroup(choices=feature_labels, label="Features")
        btn_save = gr.Button("Save Edit", visible=False)

    with gr.Row():
        btn_export = gr.Button("Export JSON")
        exported_file = gr.File(label="Exported JSON File")

    btn_load.click(
        fn=on_upload,
        inputs=[upload_files],
        outputs=[
            state,
            gallery,
            sel,
            preview,
            summary,
            feats_html,
            status,
            btn_process,
            btn_process_all,
            overlay_wrap,
            btn_save,
            cat_dd,
            subcat_dd,
            col_dd,
            pat_dd,
            fit_dd,
            feats_cb,
        ],
        queue=True,
    )

    btn_process.click(
        fn=process_next,
        inputs=[state, sel],
        outputs=[
            state,
            gallery,
            sel,
            preview,
            summary,
            feats_html,
            status,
            overlay_wrap,
            btn_save,
            cat_dd,
            subcat_dd,
            col_dd,
            pat_dd,
            fit_dd,
            feats_cb,
        ],
        queue=True,
    )

    btn_process_all.click(
        fn=process_all,
        inputs=[state, sel],
        outputs=[
            state,
            gallery,
            sel,
            preview,
            summary,
            feats_html,
            status,
            overlay_wrap,
            btn_save,
            cat_dd,
            subcat_dd,
            col_dd,
            pat_dd,
            fit_dd,
            feats_cb,
        ],
        queue=True,
    )

    sel.change(
        fn=on_select,
        inputs=[state, sel],
        outputs=[preview, summary, feats_html, btn_save, cat_dd, subcat_dd, col_dd, pat_dd, fit_dd, feats_cb],
        queue=True,
    )

    btn_save.click(
        fn=save_edit,
        inputs=[state, sel, cat_dd, subcat_dd, col_dd, pat_dd, fit_dd, feats_cb],
        outputs=[state, summary, feats_html],
        queue=True,
    )

    btn_export.click(
        fn=export_json,
        inputs=[state],
        outputs=[exported_file, status],
        queue=True,
    )

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
