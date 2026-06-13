import cv2
import os
import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="Brain Tumor AI Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================
# CONSTANTS
# ==================================

APP_VERSION = "v2.0.0"
MODEL_ACCURACY = "88.8%"
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

CLASS_COLORS = {
    "Glioma": {"primary": "#ef4444", "secondary": "#dc2626", "glow": "rgba(239,68,68,0.4)"},
    "Meningioma": {"primary": "#f59e0b", "secondary": "#d97706", "glow": "rgba(245,158,11,0.4)"},
    "No Tumor": {"primary": "#10b981", "secondary": "#059669", "glow": "rgba(16,185,129,0.4)"},
    "Pituitary": {"primary": "#8b5cf6", "secondary": "#7c3aed", "glow": "rgba(139,92,246,0.4)"},
}

RECOMMENDATIONS = {
    "No Tumor": {
        "level": "safe",
        "icon": "✅",
        "title": "No Tumor Detected",
        "message": "The AI model did not detect any signs of a brain tumor in this MRI scan. Continue routine health monitoring as advised by your physician.",
        "action": "Routine follow-up recommended as per standard medical guidelines.",
    },
    "Glioma": {
        "level": "critical",
        "icon": "🚨",
        "title": "Critical — Possible Glioma",
        "message": "The model has identified patterns consistent with Glioma. This is a serious finding that requires immediate medical attention.",
        "action": "Consult a neurologist or neuro-oncologist immediately for confirmatory diagnosis.",
    },
    "Meningioma": {
        "level": "warning",
        "icon": "⚠️",
        "title": "Warning — Possible Meningioma",
        "message": "Patterns suggestive of Meningioma have been detected. Further clinical evaluation is strongly recommended.",
        "action": "Schedule an appointment with a neurologist for comprehensive assessment.",
    },
    "Pituitary": {
        "level": "warning",
        "icon": "⚠️",
        "title": "Warning — Possible Pituitary Tumor",
        "message": "The scan shows indicators that may correspond to a Pituitary tumor. Specialist consultation is advised.",
        "action": "Refer to an endocrinologist or neurosurgeon for detailed evaluation.",
    },
}

# ==================================
# CUSTOM CSS
# ==================================

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0;
}

.stApp {
    background: #060b18;
    color: #e2e8f0;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}

/* ── Animated Background ── */
.bg-layer {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

.bg-orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.35;
    animation: floatOrb 12s ease-in-out infinite;
}

.bg-orb-1 {
    width: 500px; height: 500px;
    background: radial-gradient(circle, #3b82f6, transparent 70%);
    top: -10%; left: -5%;
    animation-delay: 0s;
}

.bg-orb-2 {
    width: 400px; height: 400px;
    background: radial-gradient(circle, #06b6d4, transparent 70%);
    top: 40%; right: -8%;
    animation-delay: -4s;
}

.bg-orb-3 {
    width: 450px; height: 450px;
    background: radial-gradient(circle, #8b5cf6, transparent 70%);
    bottom: -5%; left: 30%;
    animation-delay: -8s;
}

@keyframes floatOrb {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(30px, -20px) scale(1.05); }
    66% { transform: translate(-20px, 15px) scale(0.95); }
}

/* ── Particles ── */
.particles {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
}

.particle {
    position: absolute;
    width: 3px; height: 3px;
    background: rgba(96, 165, 250, 0.6);
    border-radius: 50%;
    animation: rise linear infinite;
}

.particle:nth-child(1)  { left: 5%;  animation-duration: 14s; animation-delay: 0s; }
.particle:nth-child(2)  { left: 15%; animation-duration: 18s; animation-delay: -3s; }
.particle:nth-child(3)  { left: 25%; animation-duration: 12s; animation-delay: -6s; }
.particle:nth-child(4)  { left: 35%; animation-duration: 16s; animation-delay: -1s; }
.particle:nth-child(5)  { left: 45%; animation-duration: 20s; animation-delay: -8s; }
.particle:nth-child(6)  { left: 55%; animation-duration: 13s; animation-delay: -4s; }
.particle:nth-child(7)  { left: 65%; animation-duration: 17s; animation-delay: -10s; }
.particle:nth-child(8)  { left: 75%; animation-duration: 15s; animation-delay: -2s; }
.particle:nth-child(9)  { left: 85%; animation-duration: 19s; animation-delay: -7s; }
.particle:nth-child(10) { left: 95%; animation-duration: 11s; animation-delay: -5s; }

@keyframes rise {
    0%   { bottom: -5%; opacity: 0; transform: translateX(0); }
    10%  { opacity: 1; }
    90%  { opacity: 0.6; }
    100% { bottom: 105%; opacity: 0; transform: translateX(40px); }
}

/* ── Hero ── */
.hero-section {
    position: relative;
    border-radius: 28px;
    overflow: hidden;
    margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 25px 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06);
}

.hero-bg {
    position: absolute;
    inset: 0;
    background:
        linear-gradient(135deg, rgba(6,11,24,0.92) 0%, rgba(15,23,42,0.85) 50%, rgba(30,27,75,0.88) 100%),
        url('https://images.unsplash.com/photo-1559757175-08f01b683993?w=1600&q=80') center/cover no-repeat;
    z-index: 0;
}

.hero-content {
    position: relative;
    z-index: 2;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 3.5rem 3rem;
    gap: 2rem;
    flex-wrap: wrap;
}

.hero-text { flex: 1; min-width: 280px; }

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 50px;
    padding: 6px 16px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #60a5fa;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    animation: pulseBadge 3s ease-in-out infinite;
}

@keyframes pulseBadge {
    0%, 100% { box-shadow: 0 0 0 0 rgba(59,130,246,0.3); }
    50% { box-shadow: 0 0 20px 4px rgba(59,130,246,0.15); }
}

.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #ffffff 0%, #93c5fd 50%, #c4b5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 1rem 0;
}

.hero-subtitle {
    font-size: clamp(0.95rem, 2vw, 1.15rem);
    color: #94a3b8;
    line-height: 1.7;
    max-width: 520px;
    margin: 0;
}

/* ── Animated Brain SVG ── */
.brain-illustration {
    flex-shrink: 0;
    width: 180px;
    height: 180px;
    animation: brainFloat 6s ease-in-out infinite;
}

@keyframes brainFloat {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    50% { transform: translateY(-12px) rotate(2deg); }
}

.brain-ring {
    animation: ringPulse 4s ease-in-out infinite;
    transform-origin: center;
}

@keyframes ringPulse {
    0%, 100% { opacity: 0.3; r: 85; }
    50% { opacity: 0.6; }
}

.brain-synapse {
    animation: synapseFlash 2s ease-in-out infinite;
}

.brain-synapse:nth-child(2) { animation-delay: 0.4s; }
.brain-synapse:nth-child(3) { animation-delay: 0.8s; }
.brain-synapse:nth-child(4) { animation-delay: 1.2s; }
.brain-synapse:nth-child(5) { animation-delay: 1.6s; }

@keyframes synapseFlash {
    0%, 100% { opacity: 0.2; }
    50% { opacity: 1; }
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 2rem 0 1.2rem 0;
}

.section-icon {
    width: 42px; height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(139,92,246,0.2));
    border: 1px solid rgba(255,255,255,0.1);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
}

.section-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0;
}

.section-subtitle {
    font-size: 0.85rem;
    color: #64748b;
    margin: 0;
}

/* ── Metric Cards ── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.2rem;
    margin-bottom: 2rem;
}

@media (max-width: 900px) {
    .metrics-grid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 500px) {
    .metrics-grid { grid-template-columns: 1fr; }
}

.metric-card {
    position: relative;
    border-radius: 20px;
    padding: 1.5rem;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: default;
}

.metric-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 20px;
    opacity: 0;
    transition: opacity 0.35s;
}

.metric-card:hover {
    transform: translateY(-6px);
    border-color: rgba(255,255,255,0.18);
}

.metric-card:hover::before { opacity: 1; }

.metric-card-1 {
    background: linear-gradient(135deg, rgba(37,99,235,0.25), rgba(59,130,246,0.1));
    box-shadow: 0 8px 32px rgba(37,99,235,0.15);
}
.metric-card-1:hover { box-shadow: 0 16px 48px rgba(37,99,235,0.35); }
.metric-card-1::before { background: radial-gradient(circle at top right, rgba(59,130,246,0.2), transparent 60%); }

.metric-card-2 {
    background: linear-gradient(135deg, rgba(124,58,237,0.25), rgba(139,92,246,0.1));
    box-shadow: 0 8px 32px rgba(124,58,237,0.15);
}
.metric-card-2:hover { box-shadow: 0 16px 48px rgba(124,58,237,0.35); }
.metric-card-2::before { background: radial-gradient(circle at top right, rgba(139,92,246,0.2), transparent 60%); }

.metric-card-3 {
    background: linear-gradient(135deg, rgba(6,182,212,0.25), rgba(8,145,178,0.1));
    box-shadow: 0 8px 32px rgba(6,182,212,0.15);
}
.metric-card-3:hover { box-shadow: 0 16px 48px rgba(6,182,212,0.35); }
.metric-card-3::before { background: radial-gradient(circle at top right, rgba(6,182,212,0.2), transparent 60%); }

.metric-card-4 {
    background: linear-gradient(135deg, rgba(16,185,129,0.25), rgba(5,150,105,0.1));
    box-shadow: 0 8px 32px rgba(16,185,129,0.15);
}
.metric-card-4:hover { box-shadow: 0 16px 48px rgba(16,185,129,0.35); }
.metric-card-4::before { background: radial-gradient(circle at top right, rgba(16,185,129,0.2), transparent 60%); }

.metric-icon {
    font-size: 1.6rem;
    margin-bottom: 0.8rem;
    display: block;
}

.metric-value {
    font-family: 'Sora', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #f8fafc;
    line-height: 1;
    margin-bottom: 0.4rem;
}

.metric-label {
    font-size: 0.82rem;
    color: #94a3b8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Glass Card ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: border-color 0.3s, box-shadow 0.3s;
}

.glass-card:hover {
    border-color: rgba(255,255,255,0.14);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}

/* ── Upload Zone ── */
.upload-zone-header {
    text-align: center;
    margin-bottom: 1.5rem;
}

.upload-icon-wrap {
    width: 72px; height: 72px;
    margin: 0 auto 1rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(139,92,246,0.2));
    border: 1px solid rgba(59,130,246,0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    animation: uploadBounce 2.5s ease-in-out infinite;
}

@keyframes uploadBounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}

.upload-zone-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0 0 0.4rem 0;
}

.upload-zone-hint {
    font-size: 0.85rem;
    color: #64748b;
    margin: 0;
}

.upload-formats {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-top: 1rem;
    flex-wrap: wrap;
}

.format-badge {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: #94a3b8;
    font-weight: 500;
}

/* Streamlit file uploader overrides */
[data-testid="stFileUploader"] {
    background: rgba(59,130,246,0.05) !important;
    border: 2px dashed rgba(59,130,246,0.3) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: all 0.3s !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(96,165,250,0.6) !important;
    background: rgba(59,130,246,0.1) !important;
    box-shadow: 0 0 30px rgba(59,130,246,0.15) !important;
}

[data-testid="stFileUploader"] section {
    padding: 0 !important;
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

[data-testid="stFileUploader"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(37,99,235,0.4) !important;
}

/* ── Image Preview Card ── */
.preview-card {
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(0,0,0,0.3);
    transition: transform 0.3s, box-shadow 0.3s;
}

.preview-card:hover {
    transform: scale(1.01);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}

.preview-label {
    background: rgba(255,255,255,0.05);
    padding: 10px 16px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* ── Prediction Card ── */
.prediction-card {
    border-radius: 24px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.1);
    transition: transform 0.3s;
}

.prediction-card:hover { transform: translateY(-4px); }

.prediction-card::after {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
    pointer-events: none;
}

.prediction-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

.prediction-class {
    font-family: 'Sora', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 0.5rem 0;
    color: #ffffff;
}

.prediction-confidence {
    font-size: 3.5rem;
    font-weight: 800;
    font-family: 'Sora', sans-serif;
    line-height: 1;
    margin: 1rem 0;
}

.confidence-label {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.7);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1.5rem;
}

.confidence-bar-wrap {
    background: rgba(0,0,0,0.3);
    border-radius: 50px;
    height: 8px;
    overflow: hidden;
    margin: 0 auto;
    max-width: 280px;
}

.confidence-bar {
    height: 100%;
    border-radius: 50px;
    transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 12px currentColor;
}

.diagnosis-summary {
    margin-top: 1.5rem;
    padding: 1rem 1.2rem;
    background: rgba(0,0,0,0.25);
    border-radius: 14px;
    font-size: 0.9rem;
    line-height: 1.6;
    color: rgba(255,255,255,0.85);
    text-align: left;
}

/* ── Recommendation Card ── */
.rec-card {
    border-radius: 20px;
    padding: 1.8rem 2rem;
    display: flex;
    gap: 1.2rem;
    align-items: flex-start;
    border: 1px solid;
    transition: transform 0.3s;
}

.rec-card:hover { transform: translateY(-3px); }

.rec-card-safe {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.08));
    border-color: rgba(16,185,129,0.3);
    box-shadow: 0 8px 32px rgba(16,185,129,0.15);
}

.rec-card-warning {
    background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(217,119,6,0.08));
    border-color: rgba(245,158,11,0.3);
    box-shadow: 0 8px 32px rgba(245,158,11,0.15);
}

.rec-card-critical {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
    border-color: rgba(239,68,68,0.3);
    box-shadow: 0 8px 32px rgba(239,68,68,0.15);
}

.rec-icon { font-size: 2rem; flex-shrink: 0; }

.rec-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
}

.rec-message {
    font-size: 0.9rem;
    color: #cbd5e1;
    line-height: 1.6;
    margin: 0 0 0.8rem 0;
}

.rec-action {
    font-size: 0.82rem;
    font-weight: 600;
    padding: 8px 14px;
    border-radius: 10px;
    display: inline-block;
}

.rec-action-safe { background: rgba(16,185,129,0.2); color: #6ee7b7; }
.rec-action-warning { background: rgba(245,158,11,0.2); color: #fcd34d; }
.rec-action-critical { background: rgba(239,68,68,0.2); color: #fca5a5; }

/* ── XAI Cards ── */
.xai-card {
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.03);
    transition: all 0.3s;
}

.xai-card:hover {
    border-color: rgba(96,165,250,0.3);
    box-shadow: 0 12px 40px rgba(59,130,246,0.15);
    transform: translateY(-4px);
}

.xai-card-header {
    padding: 12px 18px;
    background: rgba(255,255,255,0.04);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex;
    align-items: center;
    gap: 8px;
}

.xai-card-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
}

.xai-card-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Footer ── */
.app-footer {
    margin-top: 3rem;
    padding: 2rem;
    border-radius: 20px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    text-align: center;
}

.footer-brand {
    font-family: 'Sora', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.5rem;
}

.footer-tagline {
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 1.2rem;
}

.footer-links {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}

.footer-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #60a5fa;
    text-decoration: none;
    font-size: 0.88rem;
    font-weight: 500;
    transition: color 0.2s;
}

.footer-link:hover { color: #93c5fd; }

.footer-meta {
    font-size: 0.78rem;
    color: #475569;
    margin-top: 0.8rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1929 0%, #060b18 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem !important;
}

.sidebar-brand {
    text-align: center;
    padding: 1rem 0 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}

.sidebar-logo {
    width: 64px; height: 64px;
    margin: 0 auto 0.8rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    box-shadow: 0 8px 24px rgba(37,99,235,0.4);
}

.sidebar-app-name {
    font-family: 'Sora', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0;
}

.sidebar-app-version {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 4px;
}

.sidebar-section {
    margin-bottom: 1.2rem;
}

.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin-bottom: 0.8rem;
    padding-left: 4px;
}

.sidebar-nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 12px;
    margin-bottom: 4px;
    font-size: 0.88rem;
    color: #94a3b8;
    transition: all 0.2s;
    cursor: default;
}

.sidebar-nav-item.active {
    background: rgba(59,130,246,0.15);
    color: #60a5fa;
    border: 1px solid rgba(59,130,246,0.2);
}

.sidebar-nav-item:hover {
    background: rgba(255,255,255,0.05);
    color: #e2e8f0;
}

.sidebar-info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    border-radius: 10px;
    background: rgba(255,255,255,0.03);
    margin-bottom: 6px;
    font-size: 0.82rem;
}

.sidebar-info-label { color: #64748b; }
.sidebar-info-value { color: #e2e8f0; font-weight: 600; }

.sidebar-profile {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    margin-top: 1rem;
}

.sidebar-avatar {
    width: 56px; height: 56px;
    border-radius: 50%;
    background: linear-gradient(135deg, #06b6d4, #3b82f6);
    margin: 0 auto 0.8rem;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    font-weight: 700;
    color: white;
    box-shadow: 0 4px 16px rgba(6,182,212,0.3);
}

.sidebar-dev-name {
    font-family: 'Sora', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 4px 0;
}

.sidebar-dev-role {
    font-size: 0.78rem;
    color: #64748b;
    margin: 0;
}

/* ── Progress bar override ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    border-radius: 50px !important;
}

/* ── Divider ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    margin: 2rem 0;
}

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 0.78rem;
    color: #fbbf24;
    text-align: center;
    margin-top: 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ==================================
# BACKGROUND LAYER
# ==================================

st.markdown(
    """
<div class="bg-layer">
    <div class="bg-orb bg-orb-1"></div>
    <div class="bg-orb bg-orb-2"></div>
    <div class="bg-orb bg-orb-3"></div>
</div>
<div class="particles">
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
</div>
""",
    unsafe_allow_html=True,
)

# ==================================
# LOAD MODEL
# ==================================

print("Current files:")
print(os.listdir())


@st.cache_resource
def load_model():
    print("Loading model...")
    model = tf.keras.models.load_model(
        "best_brain_tumor_model.keras",
        compile=False,
    )
    print("Model loaded successfully!")
    return model


model = load_model()


def make_gradcam_heatmap(img_array, model):
    efficientnet = model.get_layer("efficientnetb0")

    feature_extractor = tf.keras.Model(
        efficientnet.input,
        efficientnet.output,
    )

    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        features = feature_extractor(img_tensor, training=False)
        tape.watch(features)

        x = model.layers[3](features)
        x = model.layers[4](x, training=False)
        x = model.layers[5](x)
        x = model.layers[6](x, training=False)
        predictions = model.layers[7](x)

        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, features)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    features = features[0]

    heatmap = tf.reduce_sum(features * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


# ==================================
# UI HELPERS
# ==================================


def build_probability_chart(prediction):
    colors = ["#ef4444", "#f59e0b", "#10b981", "#8b5cf6"]
    probs = [float(p) * 100 for p in prediction[0]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=CLASS_NAMES,
                y=probs,
                marker=dict(
                    color=colors,
                    line=dict(color="rgba(255,255,255,0.1)", width=1),
                ),
                text=[f"{p:.1f}%" for p in probs],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=13, family="Inter"),
                hovertemplate="<b>%{x}</b><br>Probability: %{y:.2f}%<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8"),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color="#94a3b8", size=12),
            linecolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#64748b", size=11),
            ticksuffix="%",
            range=[0, max(probs) * 1.25 + 5],
            linecolor="rgba(255,255,255,0.1)",
        ),
        bargap=0.35,
        transition=dict(duration=800, easing="cubic-in-out"),
    )

    return fig


def render_prediction_card(predicted_class, confidence):
    colors = CLASS_COLORS[predicted_class]
    conf_pct = confidence * 100
    badge_map = {
        "No Tumor": ("✓ Clear", "rgba(16,185,129,0.2)", "#6ee7b7"),
        "Glioma": ("⚠ Critical", "rgba(239,68,68,0.2)", "#fca5a5"),
        "Meningioma": ("⚠ Warning", "rgba(245,158,11,0.2)", "#fcd34d"),
        "Pituitary": ("⚠ Warning", "rgba(245,158,11,0.2)", "#fcd34d"),
    }
    badge_text, badge_bg, badge_color = badge_map[predicted_class]

    summaries = {
        "Glioma": "Aggressive tumor originating from glial cells. Grad-CAM highlights regions of abnormal tissue density.",
        "Meningioma": "Typically benign tumor arising from meninges. Model attention focused on membrane-adjacent structures.",
        "No Tumor": "No significant abnormal patterns detected. Brain tissue appears within normal classification parameters.",
        "Pituitary": "Tumor detected near pituitary gland region. Model identified characteristic sellar region patterns.",
    }

    st.markdown(
        f"""
        <div class="prediction-card" style="
            background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']});
            box-shadow: 0 16px 48px {colors['glow']};
        ">
            <div class="prediction-badge" style="background:{badge_bg}; color:{badge_color};">
                {badge_text}
            </div>
            <p class="prediction-class">{predicted_class}</p>
            <p class="confidence-label">AI Confidence Score</p>
            <p class="prediction-confidence">{conf_pct:.1f}%</p>
            <div class="confidence-bar-wrap">
                <div class="confidence-bar" style="
                    width: {conf_pct}%;
                    background: linear-gradient(90deg, rgba(255,255,255,0.8), #ffffff);
                "></div>
            </div>
            <div class="diagnosis-summary">
                <strong>AI Diagnosis Summary</strong><br>
                {summaries[predicted_class]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation(predicted_class):
    rec = RECOMMENDATIONS[predicted_class]
    level_class = f"rec-card-{rec['level']}"
    action_class = f"rec-action-{rec['level']}"

    st.markdown(
        f"""
        <div class="rec-card {level_class}">
            <div class="rec-icon">{rec['icon']}</div>
            <div>
                <p class="rec-title">{rec['title']}</p>
                <p class="rec-message">{rec['message']}</p>
                <span class="rec-action {action_class}">{rec['action']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================
# SIDEBAR
# ==================================

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-logo">🧠</div>
            <p class="sidebar-app-name">NeuroScan AI</p>
            <p class="sidebar-app-version">{APP_VERSION}</p>
        </div>

        <div class="sidebar-section">
            <p class="sidebar-section-title">Navigation</p>
            <div class="sidebar-nav-item active">🏠 Dashboard</div>
            <div class="sidebar-nav-item">📤 Upload & Analyze</div>
            <div class="sidebar-nav-item">🔬 Explainable AI</div>
            <div class="sidebar-nav-item">📊 Analytics</div>
        </div>

        <div class="sidebar-section">
            <p class="sidebar-section-title">Model Information</p>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Architecture</span>
                <span class="sidebar-info-value">EfficientNetB0</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Framework</span>
                <span class="sidebar-info-value">TensorFlow</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Input Size</span>
                <span class="sidebar-info-value">224 × 224</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Classes</span>
                <span class="sidebar-info-value">4 Types</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Accuracy</span>
                <span class="sidebar-info-value">{MODEL_ACCURACY}</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">XAI Method</span>
                <span class="sidebar-info-value">Grad-CAM</span>
            </div>
        </div>

        <div class="sidebar-section">
            <p class="sidebar-section-title">Tumor Classes</p>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">🔴 Glioma</span>
                <span class="sidebar-info-value">Class 0</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">🟠 Meningioma</span>
                <span class="sidebar-info-value">Class 1</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">🟢 No Tumor</span>
                <span class="sidebar-info-value">Class 2</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">🟣 Pituitary</span>
                <span class="sidebar-info-value">Class 3</span>
            </div>
        </div>

        <div class="sidebar-profile">
            <div class="sidebar-avatar">RK</div>
            <p class="sidebar-dev-name">Rishi Khandelwal</p>
            <p class="sidebar-dev-role">AI & Deep Learning Engineer</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==================================
# HERO SECTION
# ==================================

st.markdown(
    """
    <div class="hero-section">
        <div class="hero-bg"></div>
        <div class="hero-content">
            <div class="hero-text">
                <div class="hero-badge">⚡ Powered by Deep Learning</div>
                <h1 class="hero-title">Brain Tumor AI Detection System</h1>
                <p class="hero-subtitle">
                    Deep Learning Powered MRI Analysis &amp; Explainable AI
                </p>
            </div>
            <svg class="brain-illustration" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle class="brain-ring" cx="100" cy="100" r="85" stroke="url(#brainGrad)" stroke-width="1.5" fill="none" opacity="0.4"/>
                <circle class="brain-ring" cx="100" cy="100" r="70" stroke="url(#brainGrad)" stroke-width="1" fill="none" opacity="0.25" style="animation-delay:-2s"/>
                <path d="M100 35 C70 35 50 60 50 90 C50 115 65 135 85 145 C80 130 78 115 80 100 C75 95 72 88 75 80 C78 72 85 68 92 70 C95 55 100 35 100 35Z" fill="url(#brainGrad)" opacity="0.7"/>
                <path d="M100 35 C130 35 150 60 150 90 C150 115 135 135 115 145 C120 130 122 115 120 100 C125 95 128 88 125 80 C122 72 115 68 108 70 C105 55 100 35 100 35Z" fill="url(#brainGrad)" opacity="0.7"/>
                <circle class="brain-synapse" cx="75" cy="85" r="4" fill="#60a5fa"/>
                <circle class="brain-synapse" cx="125" cy="85" r="4" fill="#a78bfa"/>
                <circle class="brain-synapse" cx="90" cy="110" r="3" fill="#34d399"/>
                <circle class="brain-synapse" cx="110" cy="110" r="3" fill="#f472b6"/>
                <circle class="brain-synapse" cx="100" cy="75" r="5" fill="#38bdf8"/>
                <line x1="75" y1="85" x2="100" y2="75" stroke="#60a5fa" stroke-width="1" opacity="0.4"/>
                <line x1="125" y1="85" x2="100" y2="75" stroke="#a78bfa" stroke-width="1" opacity="0.4"/>
                <line x1="90" y1="110" x2="100" y2="75" stroke="#34d399" stroke-width="1" opacity="0.4"/>
                <line x1="110" y1="110" x2="100" y2="75" stroke="#f472b6" stroke-width="1" opacity="0.4"/>
                <defs>
                    <linearGradient id="brainGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#3b82f6"/>
                        <stop offset="50%" stop-color="#06b6d4"/>
                        <stop offset="100%" stop-color="#8b5cf6"/>
                    </linearGradient>
                </defs>
            </svg>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================================
# DASHBOARD METRICS
# ==================================

st.markdown(
    f"""
    <div class="metrics-grid">
        <div class="metric-card metric-card-1">
            <span class="metric-icon">🎯</span>
            <div class="metric-value">4</div>
            <div class="metric-label">Tumor Classes</div>
        </div>
        <div class="metric-card metric-card-2">
            <span class="metric-icon">🧬</span>
            <div class="metric-value" style="font-size:1.3rem; padding-top:0.5rem;">EfficientNetB0</div>
            <div class="metric-label">Model Architecture</div>
        </div>
        <div class="metric-card metric-card-3">
            <span class="metric-icon">📐</span>
            <div class="metric-value">224²</div>
            <div class="metric-label">Input Resolution</div>
        </div>
        <div class="metric-card metric-card-4">
            <span class="metric-icon">📈</span>
            <div class="metric-value">{MODEL_ACCURACY}</div>
            <div class="metric-label">Model Accuracy</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================================
# UPLOAD SECTION
# ==================================

st.markdown(
    """
    <div class="section-header">
        <div class="section-icon">📤</div>
        <div>
            <p class="section-title">MRI Scan Upload</p>
            <p class="section-subtitle">Upload a brain MRI image for AI-powered tumor classification</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_col1, upload_col2 = st.columns([1, 1])

with upload_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="upload-zone-header">
            <div class="upload-icon-wrap">📁</div>
            <p class="upload-zone-title">Drag & Drop MRI Scan</p>
            <p class="upload-zone-hint">or click below to browse your files</p>
            <div class="upload-formats">
                <span class="format-badge">JPG</span>
                <span class="format-badge">JPEG</span>
                <span class="format-badge">PNG</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Select MRI Scan",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ==================================
# PREDICTION PIPELINE
# ==================================

if uploaded_file:
    image = Image.open(uploaded_file)
    img = image.resize((224, 224))
    img_array = np.array(img)

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    heatmap = make_gradcam_heatmap(img_array, model)

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = img_array[0].astype(np.uint8)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    with upload_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="preview-card">
                <div class="preview-label">📷 Uploaded MRI Preview</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Prediction & Chart ──
    st.markdown(
        """
        <div class="section-header">
            <div class="section-icon">🔍</div>
            <div>
                <p class="section-title">AI Diagnosis Results</p>
                <p class="section-subtitle">Real-time classification powered by EfficientNetB0</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pred_col, chart_col = st.columns([1, 1])

    with pred_col:
        render_prediction_card(predicted_class, confidence)

    with chart_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <p style="font-family:'Sora',sans-serif; font-weight:700; font-size:1rem;
            color:#e2e8f0; margin:0 0 0.5rem 0;">📊 Class Probability Distribution</p>
            <p style="font-size:0.82rem; color:#64748b; margin:0 0 1rem 0;">
                Hover over bars for detailed probability scores
            </p>
            """,
            unsafe_allow_html=True,
        )
        fig = build_probability_chart(prediction)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Explainable AI ──
    st.markdown(
        """
        <div class="section-header">
            <div class="section-icon">🔬</div>
            <div>
                <p class="section-title">Explainable AI — Grad-CAM Visualization</p>
                <p class="section-subtitle">Understand which brain regions influenced the AI's decision</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    xai_col1, xai_col2 = st.columns(2)

    with xai_col1:
        st.markdown(
            """
            <div class="xai-card">
                <div class="xai-card-header">
                    <div class="xai-card-dot" style="background:#3b82f6;"></div>
                    <span class="xai-card-title">Original MRI Scan</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(original, use_container_width=True)

    with xai_col2:
        st.markdown(
            """
            <div class="xai-card">
                <div class="xai-card-header">
                    <div class="xai-card-dot" style="background:#ef4444;"></div>
                    <span class="xai-card-title">Grad-CAM Heatmap Overlay</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(overlay, caption="Red/warm regions indicate areas the model focused on", use_container_width=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Recommendation ──
    st.markdown(
        """
        <div class="section-header">
            <div class="section-icon">🩺</div>
            <div>
                <p class="section-title">Clinical Recommendation</p>
                <p class="section-subtitle">AI-generated guidance based on classification results</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_recommendation(predicted_class)

    st.markdown(
        """
        <div class="disclaimer">
            ⚕️ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only.
            It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    with upload_col2:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:3rem 2rem; min-height:280px;
            display:flex; flex-direction:column; align-items:center; justify-content:center;">
                <div style="font-size:3rem; margin-bottom:1rem; opacity:0.4;">🧠</div>
                <p style="font-family:'Sora',sans-serif; font-weight:600; color:#64748b; margin:0 0 0.5rem 0;">
                    Awaiting MRI Upload
                </p>
                <p style="font-size:0.85rem; color:#475569; margin:0; max-width:280px; line-height:1.6;">
                    Upload a brain MRI scan to receive AI-powered tumor classification,
                    confidence scores, and Grad-CAM explainability visualizations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==================================
# FOOTER
# ==================================

st.markdown(
    f"""
    <div class="app-footer">
        <p class="footer-brand">Brain Tumor AI Detection System</p>
        <p class="footer-tagline">
            Deep Learning • TensorFlow • EfficientNetB0 • Grad-CAM Explainable AI
        </p>
        <div class="footer-links">
            <a class="footer-link" href="https://github.com/RishI-1207-MY" target="_blank">
                <span>🔗</span> GitHub
            </a>
            <a class="footer-link" href="https://github.com/RishI-1207-MY/MY-brain-tumor-detection" target="_blank">
                <span>📦</span> Repository
            </a>
            <a class="footer-link" href="https://www.linkedin.com/in/rishi-khandelwal-5b5b3a290?utm_source=share_via&utm_content=profile&utm_medium=member_android" target="_blank">
                <span>💼</span> LinkedIn
            </a>
        </div>
        <p style="font-size:0.88rem; color:#94a3b8; margin:0;">
            Developed by <strong style="color:#e2e8f0;">Rishi Khandelwal</strong>
        </p>
        <p class="footer-meta">
            {APP_VERSION} &nbsp;•&nbsp; © 2025 Rishi Khandelwal &nbsp;•&nbsp; All Rights Reserved
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
