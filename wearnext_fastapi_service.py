from __future__ import annotations

# 1.1 Imports
import json
import mimetypes
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import google.generativeai as genai
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image


# 1.2 App Config
APP_TITLE = "WearNext LLM Service"
APP_VERSION = "1.0.0"
BASE_DIR = Path(os.getenv("WEARNEXT_BASE_DIR", ".")).resolve()
EXPORT_DIR = Path(os.getenv("WEARNEXT_EXPORT_DIR", "exports")).resolve()
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES_JSON = Path(os.getenv("CATEGORIES_JSON", BASE_DIR / "categories.json"))
COLOURS_JSON = Path(os.getenv("COLOURS_JSON", BASE_DIR / "colours.json"))
PATTERN_JSON = Path(os.getenv("PATTERN_JSON", BASE_DIR / "pattern.json"))
FIT_JSON = Path(os.getenv("FIT_JSON", BASE_DIR / "fit.json"))
FEATURES_JSON = Path(os.getenv("FEATURES_JSON", BASE_DIR / "WN_Features1.JSON"))
SUBCATS_JSON = Path(os.getenv("SUBCATS_JSON", BASE_DIR / "WN_SubCategories.JSON"))

STRAPI_BASE_URL = os.getenv("STRAPI_BASE_URL", "https://api.wearnext.com.au").rstrip("/")
STRAPI_TOKEN = os.getenv("STRAPI_TOKEN", "").strip()
STRAPI_DEFAULT_INVENTORY_ENDPOINT = os.getenv(
    "STRAPI_DEFAULT_INVENTORY_ENDPOINT",
    f"{STRAPI_BASE_URL}/api/default-inventories",
).rstrip("/")
STRAPI_UPLOAD_ENDPOINT = os.getenv("STRAPI_UPLOAD_ENDPOINT", f"{STRAPI_BASE_URL}/api/upload").rstrip("/")
STRAPI_CATEGORIES_ENDPOINT = os.getenv("STRAPI_CATEGORIES_ENDPOINT", f"{STRAPI_BASE_URL}/api/categories").rstrip("/")
STRAPI_SUBCATEGORIES_ENDPOINT = os.getenv("STRAPI_SUBCATEGORIES_ENDPOINT", f"{STRAPI_BASE_URL}/api/sub-categories").rstrip("/")
STRAPI_COLOURS_ENDPOINT = os.getenv("STRAPI_COLOURS_ENDPOINT", f"{STRAPI_BASE_URL}/api/colours").rstrip("/")
STRAPI_PATTERNS_ENDPOINT = os.getenv("STRAPI_PATTERNS_ENDPOINT", f"{STRAPI_BASE_URL}/api/patterns").rstrip("/")
STRAPI_FEATURES_ENDPOINT = os.getenv("STRAPI_FEATURES_ENDPOINT", f"{STRAPI_BASE_URL}/api/features").rstrip("/")
STRAPI_HEADERS = {"Authorization": f"Bearer {STRAPI_TOKEN}"} if STRAPI_TOKEN else {}


# 1.3 FastAPI App
app = FastAPI(title=APP_TITLE, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 2.1 Pydantic Models
class PredictionResult(BaseModel):
    category: str | None = None
    subcategory: str | None = None
    colour: str | None = None
    pattern: str | None = None
    fit: str | None = None
    features: list[str] = Field(default_factory=list)
    warning: str | None = None


class PredictedItem(BaseModel):
    image_name: str
    image_path: str | None = None
    edited: bool = False
    prediction: PredictionResult


class ExportRequest(BaseModel):
    items: list[PredictedItem]
    push_to_strapi: bool = True
    export_filename: str | None = None


class ExportResponse(BaseModel):
    export_path: str
    pushed: int
    failed: int
    failures: list[str] = Field(default_factory=list)


# 2.2 Taxonomy Helpers

def _safe_load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None



def _extract_labels(obj: Any, key: str) -> list[str]:
    labels: list[str] = []
    if isinstance(obj, dict) and isinstance(obj.get(key), list):
        for item in obj[key]:
            if isinstance(item, dict) and isinstance(item.get("label"), str) and item["label"].strip():
                labels.append(item["label"].strip())
            elif isinstance(item, str) and item.strip():
                labels.append(item.strip())
    return labels



def load_all_labels() -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    categories = _extract_labels(_safe_load_json(CATEGORIES_JSON), "categories")
    colours = _extract_labels(_safe_load_json(COLOURS_JSON), "colours")
    patterns = _extract_labels(_safe_load_json(PATTERN_JSON), "patterns")
    fits = _extract_labels(_safe_load_json(FIT_JSON), "fit")
    features = _extract_labels(_safe_load_json(FEATURES_JSON), "features")
    subcats = _extract_labels(_safe_load_json(SUBCATS_JSON), "labels")
    return categories, subcats, colours, patterns, fits, features


category_labels, subcategory_labels, colour_labels, pattern_labels, fit_labels, feature_labels = load_all_labels()


# 2.3 Gemini Helpers

def _get_api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")



def pick_gemini_model() -> str | None:
    api_key = _get_api_key()
    if not api_key:
        return None

    genai.configure(api_key=api_key)
    usable: list[str] = []

    for m in genai.list_models():
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            usable.append(m.name)

    if not usable:
        return None

    for token in ["2.5-flash", "flash", "2.0-flash", "1.5-flash", "pro"]:
        for name in usable:
            if token in name.lower():
                return name

    return usable[0]


GEMINI_MODEL_NAME = pick_gemini_model()



def _cleanup_model_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned



def build_prompt_primary() -> str:
    return f"""
You are a High-Precision Fashion Classifier.

Task:
- Identify ONLY the PRIMARY clothing item in the image.
- Return ONLY valid JSON. No markdown. No extra text.

STRICT RULES:
- category MUST be one of: {category_labels}
- subcategory MUST be one of: {subcategory_labels}
- colour MUST be one of: {colour_labels}
- pattern MUST be one of: {pattern_labels}
- fit MUST be one of: {fit_labels}
- features MUST be a list where each item is one of: {feature_labels}

Return JSON EXACTLY like this:
{{
  "category": "",
  "subcategory": "",
  "colour": "",
  "pattern": "",
  "fit": "",
  "features": []
}}
""".strip()



def gemini_classify_primary(image_path: str) -> dict[str, Any]:
    api_key = _get_api_key()
    if not api_key:
        return {"_warning": "GOOGLE_API_KEY / GEMINI_API_KEY not set."}

    if not GEMINI_MODEL_NAME:
        return {"_warning": "No usable Gemini model found for this API key."}

    genai.configure(api_key=api_key)

    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((768, 768))
    except Exception as e:
        return {"_warning": f"Could not open image: {e}"}

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    try:
        resp = model.generate_content([build_prompt_primary(), img])
        text = _cleanup_model_text(getattr(resp, "text", ""))
        if not text:
            return {"_warning": "Empty model response."}

        data = json.loads(text)
        feats = data.get("features", [])
        if not isinstance(feats, list):
            feats = []
        feats = [x for x in feats if isinstance(x, str)]

        return {
            "category": data.get("category"),
            "subcategory": data.get("subcategory"),
            "colour": data.get("colour"),
            "pattern": data.get("pattern"),
            "fit": data.get("fit"),
            "features": list(dict.fromkeys(feats)),
        }
    except Exception as e:
        return {"_warning": f"Gemini call failed: {type(e).__name__}: {e}"}


# 2.4 File Helpers

def _save_upload_to_temp(upload: UploadFile) -> tuple[str, str]:
    suffix = Path(upload.filename or "image.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = upload.file.read()
        tmp.write(content)
        return tmp.name, Path(upload.filename or tmp.name).name



def _record_from_pred(image_path: str | None, image_name: str, pred: dict[str, Any], source: str = "llm") -> dict[str, Any]:
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "image_path": image_path,
        "image_name": image_name,
        "source": source,
        "prediction": {
            "category": pred.get("category") if pred else None,
            "subcategory": pred.get("subcategory") if pred else None,
            "colour": pred.get("colour") if pred else None,
            "pattern": pred.get("pattern") if pred else None,
            "fit": pred.get("fit") if pred else None,
            "features": pred.get("features", []) if pred else [],
        },
    }


# 3.1 Strapi Helpers

def _normalize_label(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip().lower().replace("-", " ").replace("_", " ")



def _strapi_enabled() -> bool:
    return bool(STRAPI_TOKEN) and STRAPI_TOKEN.lower() not in ("none", "null", "undefined")



def _strapi_get_first_id(endpoint: str, name_value: str) -> int | None:
    if not name_value:
        return None
    params = {
        "filters[name][$eq]": name_value,
        "pagination[pageSize]": 1,
        "fields[0]": "name",
    }
    r = requests.get(endpoint, headers=STRAPI_HEADERS, params=params, timeout=15)
    if r.status_code != 200:
        return None
    data = (r.json() or {}).get("data", []) or []
    if not data:
        return None
    return data[0].get("id")



def _strapi_create_and_get_id(endpoint: str, name_value: str) -> int | None:
    if not name_value:
        return None
    payload = {"data": {"name": name_value}}
    r = requests.post(endpoint, headers={**STRAPI_HEADERS, "Content-Type": "application/json"}, json=payload, timeout=15)
    if r.status_code not in (200, 201):
        return None
    obj = (r.json() or {}).get("data") or {}
    return obj.get("id")



def _strapi_resolve_id(endpoint: str, name_value: str) -> int | None:
    return _strapi_get_first_id(endpoint, name_value) or _strapi_create_and_get_id(endpoint, name_value)



def _strapi_get_all(endpoint: str, pagination_size: int = 200) -> list[dict[str, Any]]:
    try:
        r = requests.get(
            endpoint,
            headers=STRAPI_HEADERS,
            params={"pagination[page]": 1, "pagination[pageSize]": pagination_size},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("data", []) if isinstance(data, dict) else []
    except Exception:
        return []



def _build_label_to_id_map(endpoint: str) -> dict[str, int]:
    items = _strapi_get_all(endpoint)
    out: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        attrs = item.get("attributes", {}) or {}
        label = attrs.get("label") or attrs.get("name") or attrs.get("title")
        if isinstance(label, str) and label.strip() and item_id is not None:
            out[label.strip()] = item_id
    return out



def _load_strapi_maps() -> dict[str, dict[str, int]]:
    return {
        "categories": _build_label_to_id_map(STRAPI_CATEGORIES_ENDPOINT),
        "subcategories": _build_label_to_id_map(STRAPI_SUBCATEGORIES_ENDPOINT),
        "colours": _build_label_to_id_map(STRAPI_COLOURS_ENDPOINT),
        "patterns": _build_label_to_id_map(STRAPI_PATTERNS_ENDPOINT),
        "features": _build_label_to_id_map(STRAPI_FEATURES_ENDPOINT),
    }


STRAPI_MAPS = _load_strapi_maps()



def _map_label_flexible(label: str | None, mapping_dict: dict[str, int]) -> int | None:
    if not label or not isinstance(label, str):
        return None
    target = _normalize_label(label)
    for k, v in mapping_dict.items():
        if _normalize_label(k) == target:
            return v
    return None



def _strapi_upload_image(image_path: str) -> tuple[int | None, str | None]:
    if not image_path or not os.path.exists(image_path):
        return None, None
    mime_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
    with open(image_path, "rb") as f:
        files = {"files": (os.path.basename(image_path), f, mime_type)}
        r = requests.post(STRAPI_UPLOAD_ENDPOINT, headers=STRAPI_HEADERS, files=files, timeout=60)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Upload failed: {r.status_code} - {r.text}")
    uploaded = r.json()
    if isinstance(uploaded, list) and uploaded:
        file_obj = uploaded[0]
        return file_obj.get("id"), file_obj.get("url")
    return None, None



def _strapi_push_default_inventory(record: dict[str, Any]) -> dict[str, Any]:
    if not _strapi_enabled():
        raise RuntimeError("STRAPI_TOKEN not set")

    pred = (record or {}).get("prediction", {}) or {}
    image_id, image_url = _strapi_upload_image(record.get("image_path"))

    cat = pred.get("category")
    subcat = pred.get("subcategory")
    colour = pred.get("colour")
    pattern = pred.get("pattern")
    feats = pred.get("features") or []

    cat_id = _map_label_flexible(cat, STRAPI_MAPS["categories"]) or (_strapi_resolve_id(STRAPI_CATEGORIES_ENDPOINT, cat) if cat else None)
    subcat_id = _map_label_flexible(subcat, STRAPI_MAPS["subcategories"]) or (_strapi_resolve_id(STRAPI_SUBCATEGORIES_ENDPOINT, subcat) if subcat else None)
    colour_id = _map_label_flexible(colour, STRAPI_MAPS["colours"]) or (_strapi_resolve_id(STRAPI_COLOURS_ENDPOINT, colour) if colour else None)
    pattern_id = _map_label_flexible(pattern, STRAPI_MAPS["patterns"]) or (_strapi_resolve_id(STRAPI_PATTERNS_ENDPOINT, pattern) if pattern else None)

    feat_ids: list[int] = []
    for feat in feats:
        fid = _map_label_flexible(feat, STRAPI_MAPS["features"]) or (_strapi_resolve_id(STRAPI_FEATURES_ENDPOINT, feat) if feat else None)
        if fid is not None:
            feat_ids.append(fid)
    feat_ids = list(dict.fromkeys(feat_ids))

    data: dict[str, Any] = {"name": record.get("image_name") or f"outfit-{record.get('timestamp', '')}"}
    if image_id is not None:
        data["image"] = image_id
    if image_url:
        data["image_url"] = image_url
    if cat_id is not None:
        data["categories"] = [cat_id]
    if subcat_id is not None:
        data["sub_categories"] = [subcat_id]
    if colour_id is not None:
        data["colours"] = [colour_id]
    if pattern_id is not None:
        data["patterns"] = [pattern_id]
    if feat_ids:
        data["features"] = feat_ids
    if pred.get("fit"):
        data["fit_text"] = pred.get("fit")

    payload = {"data": data}
    r = requests.post(
        STRAPI_DEFAULT_INVENTORY_ENDPOINT,
        headers={**STRAPI_HEADERS, "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Create entry failed: {r.status_code} - {r.text}")
    return r.json()


# 3.2 Health and Metadata Endpoints
@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": GEMINI_MODEL_NAME,
        "strapi_enabled": _strapi_enabled(),
        "taxonomy_loaded": {
            "categories": len(category_labels),
            "subcategories": len(subcategory_labels),
            "colours": len(colour_labels),
            "patterns": len(pattern_labels),
            "fits": len(fit_labels),
            "features": len(feature_labels),
        },
    }


@app.get("/taxonomy")
def taxonomy() -> dict[str, list[str]]:
    return {
        "categories": category_labels,
        "subcategories": subcategory_labels,
        "colours": colour_labels,
        "patterns": pattern_labels,
        "fits": fit_labels,
        "features": feature_labels,
    }


# 4.1 Prediction Endpoints
@app.post("/predict")
def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    image_path, image_name = _save_upload_to_temp(file)
    try:
        pred = gemini_classify_primary(image_path)
        record = _record_from_pred(image_path, image_name, pred, source="llm")
        return record
    finally:
        try:
            os.remove(image_path)
        except OSError:
            pass


@app.post("/predict/bulk")
def predict_bulk(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    items: list[dict[str, Any]] = []
    for file in files:
        image_path, image_name = _save_upload_to_temp(file)
        try:
            pred = gemini_classify_primary(image_path)
            items.append(_record_from_pred(image_path, image_name, pred, source="llm"))
        finally:
            try:
                os.remove(image_path)
            except OSError:
                pass

    return {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model": GEMINI_MODEL_NAME,
        "items": items,
    }


# 4.2 Export Endpoint
@app.post("/export", response_model=ExportResponse)
def export_results(payload: ExportRequest) -> ExportResponse:
    if not payload.items:
        raise HTTPException(status_code=400, detail="No items supplied for export.")

    export_name = payload.export_filename or f"wearnext_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    export_path = EXPORT_DIR / export_name

    out = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model": GEMINI_MODEL_NAME,
        "items": [],
    }

    pushed = 0
    failed = 0
    failures: list[str] = []

    for item in payload.items:
        out["items"].append(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "image_name": item.image_name,
                "image_path": item.image_path,
                "prediction": item.prediction.model_dump(),
            }
        )

    with export_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if payload.push_to_strapi:
        for item in payload.items:
            if not item.image_path:
                failed += 1
                failures.append(f"{item.image_name}: missing image_path for Strapi upload")
                continue
            record = _record_from_pred(
                image_path=item.image_path,
                image_name=item.image_name,
                pred=item.prediction.model_dump(),
                source="manual_edit" if item.edited else "llm",
            )
            try:
                _strapi_push_default_inventory(record)
                pushed += 1
            except Exception as e:
                failed += 1
                failures.append(f"{item.image_name}: {e}")

    return ExportResponse(
        export_path=str(export_path),
        pushed=pushed,
        failed=failed,
        failures=failures,
    )


# 4.3 Optional Single-Step Endpoint
@app.post("/predict-and-export")
def predict_and_export(
    file: UploadFile = File(...),
    push_to_strapi: bool = Form(True),
) -> dict[str, Any]:
    image_path, image_name = _save_upload_to_temp(file)
    try:
        pred = gemini_classify_primary(image_path)
        item = PredictedItem(
            image_name=image_name,
            image_path=image_path,
            edited=False,
            prediction=PredictionResult(
                category=pred.get("category"),
                subcategory=pred.get("subcategory"),
                colour=pred.get("colour"),
                pattern=pred.get("pattern"),
                fit=pred.get("fit"),
                features=pred.get("features", []),
                warning=pred.get("_warning"),
            ),
        )
        export_resp = export_results(ExportRequest(items=[item], push_to_strapi=push_to_strapi))
        return {
            "prediction": _record_from_pred(image_path, image_name, pred, source="llm"),
            "export": export_resp.model_dump(),
        }
    finally:
        pass


# 5.1 Startup Validation
@app.on_event("startup")
def startup_event() -> None:
    if not GEMINI_MODEL_NAME:
        print("Warning: No Gemini model available. Prediction endpoints will fail until API key is set.")
    missing = [
        str(path)
        for path in [CATEGORIES_JSON, COLOURS_JSON, PATTERN_JSON, FIT_JSON, FEATURES_JSON, SUBCATS_JSON]
        if not path.exists()
    ]
    if missing:
        print(f"Warning: Missing taxonomy files: {missing}")


# 5.2 Local Dev Entrypoint
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("wearnext_fastapi_service:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
