"""DeepLens - a Streamlit interface for image and video deepfake detection."""

from __future__ import annotations

import importlib
import io
import time
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parent
IMAGE_WEIGHTS = ROOT / "model" / "efficientnet_b0_baseline.pth"
VIDEO_WEIGHTS = ROOT / "model" / "deepguard_weights.pth"


st.set_page_config(
    page_title="DeepLens | Deepfake Detection",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#f3f5ff; --muted:#9ba4c3; --panel:#151a2e; --line:#28304d; --cyan:#5ee7d2; --violet:#9b8cff; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: radial-gradient(circle at 75% 0%, #202653 0, #0b0e1a 42%, #080a12 100%); color:var(--ink); }
    .block-container { max-width: 1240px; padding: 2rem 3rem 4rem; }
    [data-testid="stSidebar"] { background:#0d1020; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] .block-container { padding:2rem 1.3rem; }
    .brand { display:flex; align-items:center; gap:11px; margin-bottom:2.4rem; }
    .brand-mark { width:37px; height:37px; display:grid; place-items:center; border-radius:12px; color:#07121d; font-weight:700; font-size:22px; background:linear-gradient(135deg,var(--cyan),var(--violet)); box-shadow:0 0 25px #5ee7d255; }
    .brand-name { font-family:'Space Grotesk'; font-weight:700; font-size:20px; letter-spacing:-.5px; }
    .eyebrow { color:var(--cyan); text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:700; margin-bottom:10px; }
    h1, h2, h3 { font-family:'Space Grotesk', sans-serif !important; color:var(--ink) !important; letter-spacing:-1px; }
    h1 { font-size:clamp(2.2rem,4vw,4.2rem) !important; line-height:1.04 !important; margin:0 !important; }
    .hero-copy { color:var(--muted); font-size:17px; line-height:1.65; max-width:650px; margin-top:18px; }
    .panel { background:linear-gradient(145deg,#171c34dd,#111528dd); border:1px solid var(--line); border-radius:20px; padding:24px; }
    .metric { background:#12172a; border:1px solid var(--line); border-radius:15px; padding:18px; }
    .metric-label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:1px; }
    .metric-value { font-family:'Space Grotesk'; font-size:25px; font-weight:700; margin-top:6px; }
    .status { display:inline-flex; align-items:center; gap:8px; border-radius:99px; padding:7px 12px; font-size:12px; font-weight:700; }
    .status-dot { width:7px; height:7px; border-radius:50%; background:currentColor; }
    .safe { color:#61e5b3; background:#163b36; } .danger { color:#ff8f9d; background:#492431; } .warn { color:#ffd27a; background:#43351c; }
    .result-title { font-family:'Space Grotesk'; font-size:32px; font-weight:700; margin:10px 0 2px; }
    .stButton > button { border-radius:11px; border:1px solid #3c4770; background:linear-gradient(135deg,#8276ee,#5b4ed1); color:white; font-weight:700; padding:.7rem 1rem; }
    .stButton > button:hover { border-color:var(--cyan); color:white; }
    .stFileUploader { border:1px dashed #49547d; border-radius:15px; background:#10152a; }
    div[data-testid="stMetric"] { background:#12172a; border:1px solid var(--line); border-radius:15px; padding:12px; }
    .footer { color:#6f7898; text-align:center; font-size:12px; margin-top:45px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _status(text: str, kind: str) -> None:
    st.markdown(
        f'<span class="status {kind}"><span class="status-dot"></span>{text}</span>',
        unsafe_allow_html=True,
    )


def _load_torch():
    import torch
    from PIL import Image
    from torchvision import transforms
    from torchvision.models import efficientnet_b0
    from torch import nn

    return torch, Image, transforms, efficientnet_b0, nn


@st.cache_resource(show_spinner=False)
def load_image_model():
    torch, _, _, efficientnet_b0, nn = _load_torch()
    if not IMAGE_WEIGHTS.exists():
        raise FileNotFoundError(f"Image checkpoint not found: {IMAGE_WEIGHTS}")
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    state = torch.load(IMAGE_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def image_prediction(uploaded_file: Any) -> dict[str, float | str]:
    torch, Image, transforms, _, _ = _load_torch()
    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    with torch.inference_mode():
        probabilities = torch.softmax(load_image_model()(transform(image).unsqueeze(0)), dim=1)[0]
    fake_probability = float(probabilities[0])
    return {
        "label": "Likely manipulated" if fake_probability >= 0.5 else "Likely authentic",
        "fake_probability": fake_probability,
        "confidence": max(fake_probability, 1 - fake_probability),
    }


def load_video_predictor() -> Any:
    """Load the predictor used by the video notebook when DeepGuard is installed."""
    candidates = [
        ("inference.video_predictor", "VideoPredictor"),
        ("deepguard.inference.video_predictor", "VideoPredictor"),
    ]
    for module_name, class_name in candidates:
        try:
            predictor_class = getattr(importlib.import_module(module_name), class_name)
            predictor = predictor_class(
                margin_ratio=0.2, conf_thres=0.5, min_face_ratio=0.01,
                model_name="ms_eff_gcvit_b0", dataset="ff++",
            )
            if hasattr(predictor, "model") and VIDEO_WEIGHTS.exists():
                torch, *_ = _load_torch()
                state = torch.load(VIDEO_WEIGHTS, map_location="cpu", weights_only=True)
                predictor.model.load_state_dict(state)
                predictor.model.eval()
            return predictor
        except (ImportError, AttributeError, TypeError, RuntimeError, FileNotFoundError):
            continue
    raise RuntimeError(
        "DeepGuard's VideoPredictor is not installed. Install the project package that "
        "provides `inference.video_predictor` to enable video analysis."
    )


def video_prediction(uploaded_file: Any, predictor: Any) -> dict[str, float | str]:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp_path = ROOT / f".deeplens_upload{suffix}"
    temp_path.write_bytes(uploaded_file.getvalue())
    try:
        result = predictor.predict_video(
            video_path=str(temp_path), num_frames=20, agg_mode="conf", tta_hflip=0.0
        )
    finally:
        temp_path.unlink(missing_ok=True)
    score = float(result.get("score", result) if isinstance(result, dict) else result)
    return {
        "label": "Likely manipulated" if score >= 0.5 else "Likely authentic",
        "fake_probability": score,
        "confidence": max(score, 1 - score),
    }


def render_result(result: dict[str, float | str], media_type: str) -> None:
    is_fake = float(result["fake_probability"]) >= 0.5
    kind = "danger" if is_fake else "safe"
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    _status("MANIPULATION SIGNAL DETECTED" if is_fake else "NO STRONG MANIPULATION SIGNAL", kind)
    st.markdown(f'<div class="result-title">{result["label"]}</div>', unsafe_allow_html=True)
    st.caption(f"{media_type} scan complete · model confidence {float(result['confidence']):.1%}")
    st.progress(float(result["fake_probability"]), text=f"Manipulation probability · {float(result['fake_probability']):.1%}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Manipulated", f"{float(result['fake_probability']):.1%}")
    with col2:
        st.metric("Authentic", f"{1 - float(result['fake_probability']):.1%}")
    st.markdown("</div>", unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">◈</div><div class="brand-name">DeepLens</div></div>', unsafe_allow_html=True)
    page = st.radio("Workspace", ["Analyze media", "About the models"], label_visibility="collapsed")
    st.divider()
    _status("SYSTEM ONLINE", "safe")
    st.caption("Local inference · your files stay on this machine")

if page == "Analyze media":
    st.markdown('<div class="eyebrow">Media forensics · v1.0</div>', unsafe_allow_html=True)
    st.markdown("<h1>See beyond<br>the pixels.</h1>", unsafe_allow_html=True)
    st.markdown('<p class="hero-copy">Upload an image or video and let two purpose-built neural networks inspect it for synthetic traces, face inconsistencies, and manipulation artifacts.</p>', unsafe_allow_html=True)
    st.write("")
    mode = st.selectbox("Analysis mode", ["Image analysis", "Video analysis"], label_visibility="collapsed")
    if mode == "Image analysis":
        uploaded = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png", "webp"], help="Supported formats: JPG, PNG, WEBP")
        model_name = "EfficientNet-B0"
    else:
        uploaded = st.file_uploader("Drop a video here", type=["mp4", "mov", "avi", "mkv"], help="Supported formats: MP4, MOV, AVI, MKV")
        model_name = "DeepGuard · MS-Eff-GCViT-B0"
    if uploaded:
        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            if mode == "Image analysis":
                st.image(uploaded, use_container_width=True)
            else:
                st.video(uploaded)
        with right:
            st.markdown(f'<div class="panel"><div class="eyebrow">Ready to scan</div><h3>{uploaded.name}</h3><p style="color:#9ba4c3">{uploaded.size / 1024:.0f} KB · {model_name}</p></div>', unsafe_allow_html=True)
            if st.button("Run forensic scan  →", use_container_width=True):
                started = time.perf_counter()
                with st.spinner("Inspecting media for synthetic traces..."):
                    try:
                        if mode == "Image analysis":
                            result = image_prediction(uploaded)
                        else:
                            result = video_prediction(uploaded, load_video_predictor())
                        st.session_state["last_result"] = (result, mode.split()[0])
                        st.session_state["elapsed"] = time.perf_counter() - started
                    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
                        st.error(str(exc))
    if "last_result" in st.session_state:
        st.write("")
        render_result(*st.session_state["last_result"])
        st.caption(f"Completed in {st.session_state.get('elapsed', 0):.2f}s · Results are indicative, not a substitute for source verification.")
    else:
        st.markdown('<div class="footer">Choose a media file above to begin your analysis.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="eyebrow">Behind the detection</div>', unsafe_allow_html=True)
    st.markdown("<h1>Two models.<br>One clear signal.</h1>", unsafe_allow_html=True)
    st.markdown('<p class="hero-copy">This project uses transfer learning: proven pretrained vision architectures are adapted and fine-tuned for deepfake detection, combining strong visual representations with a focused forensic classification task.</p>', unsafe_allow_html=True)
    st.write("")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="panel"><div class="eyebrow">For images · transfer learning</div><h2>EfficientNet-B0</h2><p style="color:#9ba4c3;line-height:1.7">An ImageNet-pretrained EfficientNet-B0 adapted through transfer learning and fine-tuned to separate authentic and manipulated face imagery. Images are resized to 224 × 224 and normalized with ImageNet statistics.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="panel"><div class="eyebrow">For video · transfer learning</div><h2>DeepGuard</h2><p style="color:#9ba4c3;line-height:1.7">A pretrained multi-scale Efficient GCViT model adapted with transfer learning for deepfake video detection. It samples 20 frames and aggregates frame-level evidence into a single manipulation score.</p></div>', unsafe_allow_html=True)
    st.info("Video analysis requires the DeepGuard package that provides `inference.video_predictor.VideoPredictor`. The checkpoint is already present in the model folder.")