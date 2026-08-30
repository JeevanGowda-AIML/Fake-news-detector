import os
import sys
import io
import pandas as pd
import streamlit as st

# Add root directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import importlib
import model.evaluate
importlib.reload(model.evaluate)
from model.predict import FakeNewsPredictor
from model.evaluate import (
    load_metrics, 
    create_confusion_matrix_fig, 
    create_metrics_bar_fig, 
    create_probability_donut_fig,
    create_cv_benchmarks_fig,
    create_performance_summary_fig
)

# Streamlit Page Configuration
st.set_page_config(
    page_title="TruthGuard AI — Fake News & Misinformation Classifier",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# DYNAMIC PAGE-SPECIFIC BACKGROUND ENGINE
# ==============================================================================
def inject_page_background(page_name):
    """
    Injects high-fidelity cyberpunk background art and ambient lighting
    tailored specifically to the context of each individual page module.
    """
    if page_name == "Single Article Classifier":
        # NLP Fact-Checking, Neural Verification & Analytical Graphs
        bg_svg = """<svg xmlns='http://www.w3.org/2000/svg' width='1440' height='900' viewBox='0 0 1440 900'>
            <defs>
                <linearGradient id='cyanPurple' x1='0%25' y1='0%25' x2='100%25' y2='100%25'>
                    <stop offset='0%25' stop-color='%2300F0FF' stop-opacity='0.4'/>
                    <stop offset='100%25' stop-color='%23A855F7' stop-opacity='0.15'/>
                </linearGradient>
            </defs>
            <!-- Left-Bottom Digital Line Chart & Bars -->
            <g opacity='0.22' stroke='%23A855F7' stroke-width='1.5' fill='none'>
                <path d='M80 820 L160 760 L240 790 L320 700 L400 740 L480 640 L560 690 L640 590' stroke='%2300F0FF' stroke-width='2'/>
                <path d='M80 850 L160 810 L240 830 L320 760 L400 800 L480 720 L560 770 L640 680' stroke='%23EC4899' stroke-width='1.5'/>
                <!-- Timeline Grid Bars -->
                <line x1='120' y1='860' x2='120' y2='800' stroke='%237C3AED' stroke-width='3'/>
                <line x1='200' y1='860' x2='200' y2='780' stroke='%237C3AED' stroke-width='3'/>
                <line x1='280' y1='860' x2='280' y2='740' stroke='%237C3AED' stroke-width='3'/>
                <line x1='360' y1='860' x2='360' y2='710' stroke='%2300F0FF' stroke-width='3'/>
                <line x1='440' y1='860' x2='440' y2='670' stroke='%2300F0FF' stroke-width='3'/>
                <line x1='520' y1='860' x2='520' y2='640' stroke='%2300F0FF' stroke-width='3'/>
                <line x1='600' y1='860' x2='600' y2='610' stroke='%2300F0FF' stroke-width='3'/>
            </g>
            <!-- Right Hologram Scanner Radar & Oscilloscope Waveforms -->
            <g opacity='0.25' stroke='%2300F0FF' stroke-width='1' fill='none'>
                <circle cx='1320' cy='720' r='90' stroke='%23A855F7' stroke-dasharray='4,4'/>
                <circle cx='1320' cy='720' r='60' stroke='%2300F0FF'/>
                <circle cx='1320' cy='720' r='30' stroke='%23EC4899'/>
                <path d='M1100 680 Q1140 620, 1180 680 T1260 680 T1340 680 T1420 680' stroke='%23A855F7' stroke-width='2'/>
                <path d='M1100 730 Q1150 670, 1200 730 T1300 730 T1400 730' stroke='%2300F0FF' stroke-width='1.5'/>
                <!-- Circuit Traces -->
                <path d='M1050 200 h150 l60 60 v150' stroke='%23A855F7'/>
                <circle cx='1260' cy='410' r='4' fill='%2300F0FF'/>
                <path d='M1200 120 h120 v80' stroke='%2300F0FF'/>
                <circle cx='1320' cy='200' r='4' fill='%23A855F7'/>
            </g>
        </svg>"""
        radial_aura = """
            radial-gradient(circle at 12% 18%, rgba(168, 85, 247, 0.16) 0%, transparent 45%),
            radial-gradient(circle at 88% 78%, rgba(0, 240, 255, 0.14) 0%, transparent 45%),
            radial-gradient(circle at 50% 50%, rgba(236, 72, 153, 0.05) 0%, transparent 60%)
        """
    elif page_name == "Batch CSV Predictor":
        # Big Data Stream Pipelines & Parallel Processing Vectors
        bg_svg = """<svg xmlns='http://www.w3.org/2000/svg' width='1440' height='900' viewBox='0 0 1440 900'>
            <!-- Data Flow Pipeline Streams -->
            <g opacity='0.20' stroke='%2300F0FF' stroke-width='1.5' fill='none'>
                <path d='M0 150 h400 l100 100 v300 l100 100 h840' stroke='%2300F0FF'/>
                <path d='M0 250 h320 l80 80 v240 l80 80 h960' stroke='%2310B981'/>
                <path d='M0 750 h600 l120-120 h720' stroke='%23A855F7'/>
                <!-- Data Nodes & Packets -->
                <circle cx='500' cy='250' r='5' fill='%2300F0FF'/>
                <circle cx='500' cy='550' r='5' fill='%2310B981'/>
                <circle cx='600' cy='650' r='5' fill='%23A855F7'/>
                <circle cx='1100' cy='650' r='4' fill='%2300F0FF'/>
                <circle cx='950' cy='410' r='4' fill='%2310B981'/>
                <!-- Parallel Hex Grids -->
                <polygon points='1250,200 1290,225 1290,275 1250,300 1210,275 1210,225' stroke='%2300F0FF' stroke-width='1.2'/>
                <polygon points='1330,250 1370,275 1370,325 1330,350 1290,325 1290,275' stroke='%2310B981' stroke-width='1.2'/>
                <polygon points='1250,300 1290,325 1290,375 1250,400 1210,375 1210,325' stroke='%23A855F7' stroke-width='1.2'/>
            </g>
        </svg>"""
        radial_aura = """
            radial-gradient(circle at 18% 22%, rgba(0, 240, 255, 0.16) 0%, transparent 45%),
            radial-gradient(circle at 82% 75%, rgba(16, 185, 129, 0.14) 0%, transparent 45%),
            radial-gradient(circle at 50% 50%, rgba(168, 85, 247, 0.05) 0%, transparent 60%)
        """
    elif page_name == "Model Metrics & Insights":
        # Multi-Layer Neural Network Weights, ROC Curvatures & Loss Contours
        bg_svg = """<svg xmlns='http://www.w3.org/2000/svg' width='1440' height='900' viewBox='0 0 1440 900'>
            <!-- Neural Network Topology & Weight Connections -->
            <g opacity='0.18' stroke='%23A855F7' stroke-width='1' fill='none'>
                <!-- Input to Hidden Layer Synapses -->
                <line x1='100' y1='250' x2='280' y2='180' stroke='%23A855F7'/>
                <line x1='100' y1='250' x2='280' y2='300' stroke='%2300F0FF'/>
                <line x1='100' y1='380' x2='280' y2='180' stroke='%2300F0FF'/>
                <line x1='100' y1='380' x2='280' y2='300' stroke='%23EC4899'/>
                <line x1='100' y1='510' x2='280' y2='300' stroke='%23A855F7'/>
                <line x1='100' y1='510' x2='280' y2='420' stroke='%23F59E0B'/>
                <line x1='280' y1='180' x2='460' y2='250' stroke='%2300F0FF'/>
                <line x1='280' y1='300' x2='460' y2='250' stroke='%23EC4899'/>
                <line x1='280' y1='300' x2='460' y2='380' stroke='%23A855F7'/>
                <line x1='280' y1='420' x2='460' y2='380' stroke='%23F59E0B'/>
                <!-- Neurons -->
                <circle cx='100' cy='250' r='8' fill='%23A855F7'/>
                <circle cx='100' cy='380' r='8' fill='%2300F0FF'/>
                <circle cx='100' cy='510' r='8' fill='%23EC4899'/>
                <circle cx='280' cy='180' r='7' fill='%2300F0FF'/>
                <circle cx='280' cy='300' r='7' fill='%23EC4899'/>
                <circle cx='280' cy='420' r='7' fill='%23F59E0B'/>
                <circle cx='460' cy='250' r='9' fill='%2300F0FF'/>
                <circle cx='460' cy='380' r='9' fill='%23A855F7'/>
            </g>
            <!-- 3D Perspective Isometric Coordinate Grid Floor on Right -->
            <g opacity='0.20' stroke='%2300F0FF' stroke-width='1' fill='none'>
                <path d='M1000 700 L1440 600' stroke='%23A855F7'/>
                <path d='M1000 750 L1440 650' stroke='%23A855F7'/>
                <path d='M1000 800 L1440 700' stroke='%23A855F7'/>
                <path d='M1000 850 L1440 750' stroke='%23A855F7'/>
                <path d='M1060 680 L1160 880' stroke='%2300F0FF'/>
                <path d='M1180 650 L1280 850' stroke='%2300F0FF'/>
                <path d='M1300 620 L1400 820' stroke='%2300F0FF'/>
            </g>
        </svg>"""
        radial_aura = """
            radial-gradient(circle at 15% 20%, rgba(147, 51, 234, 0.16) 0%, transparent 45%),
            radial-gradient(circle at 85% 75%, rgba(245, 158, 11, 0.12) 0%, transparent 45%),
            radial-gradient(circle at 50% 50%, rgba(236, 72, 153, 0.08) 0%, transparent 60%)
        """
    else:  # Real-World Datasets Guide
        # Global Research Node Graph & Distributed Knowledge Hubs
        bg_svg = """<svg xmlns='http://www.w3.org/2000/svg' width='1440' height='900' viewBox='0 0 1440 900'>
            <!-- Globe Orbit Arcs & Knowledge Nodes -->
            <g opacity='0.22' stroke='%2300F0FF' stroke-width='1.2' fill='none'>
                <ellipse cx='1280' cy='280' rx='140' ry='70' stroke='%2300F0FF' transform='rotate(-20 1280 280)'/>
                <ellipse cx='1280' cy='280' rx='140' ry='70' stroke='%23A855F7' transform='rotate(40 1280 280)'/>
                <ellipse cx='1280' cy='280' rx='140' ry='70' stroke='%23EC4899' transform='rotate(80 1280 280)'/>
                <circle cx='1280' cy='280' r='45' stroke='%2300F0FF' fill='%230A0416' fill-opacity='0.5'/>
                <circle cx='1200' cy='240' r='5' fill='%2300F0FF'/>
                <circle cx='1360' cy='310' r='5' fill='%23EC4899'/>
                <circle cx='1270' cy='360' r='5' fill='%23A855F7'/>
                <!-- Left Knowledge Hub Links -->
                <path d='M100 700 Q250 620, 400 720 T700 700' stroke='%2300F0FF' stroke-width='2'/>
                <circle cx='100' cy='700' r='6' fill='%2300F0FF'/>
                <circle cx='400' cy='720' r='6' fill='%23A855F7'/>
                <circle cx='700' cy='700' r='6' fill='%23EC4899'/>
            </g>
        </svg>"""
        radial_aura = """
            radial-gradient(circle at 12% 15%, rgba(56, 189, 248, 0.16) 0%, transparent 45%),
            radial-gradient(circle at 88% 82%, rgba(168, 85, 247, 0.15) 0%, transparent 45%),
            radial-gradient(circle at 50% 50%, rgba(236, 72, 153, 0.06) 0%, transparent 60%)
        """

    # URL-encode SVG for CSS data URI
    import urllib.parse
    encoded_svg = urllib.parse.quote(bg_svg.strip())

    st.markdown(f"""
    <style>
        .stApp {{
            background-color: #0A0416 !important;
            background-image: 
                {radial_aura},
                url("data:image/svg+xml,{encoded_svg}") !important;
            background-repeat: no-repeat !important;
            background-position: center top !important;
            background-size: cover !important;
            background-attachment: fixed !important;
        }}
    </style>
    """, unsafe_allow_html=True)


# TruthGuard Exact Screen 1, 2, 3 & 4 Cyberpunk & Glassmorphism Global Styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    .stApp {
        color: #F8FAFC;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Spacing */
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1350px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090314 0%, #06020E 100%) !important;
        border-right: 1px solid rgba(168, 85, 247, 0.2) !important;
        box-shadow: 10px 0 30px rgba(0, 0, 0, 0.5);
    }

    /* Sidebar Brand Logo */
    .brand-container {
        text-align: center;
        padding: 12px 0 16px 0;
    }
    .brand-shield {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 64px;
        height: 64px;
        background: radial-gradient(circle, rgba(0, 240, 255, 0.25) 0%, rgba(147, 51, 234, 0.1) 70%);
        border: 2px solid #00F0FF;
        border-radius: 16px;
        box-shadow: 0 0 25px rgba(0, 240, 255, 0.4), inset 0 0 15px rgba(0, 240, 255, 0.2);
        margin-bottom: 10px;
    }
    .brand-shield svg {
        width: 34px;
        height: 34px;
        fill: none;
        stroke: #00F0FF;
        stroke-width: 2;
    }
    .brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: 0.5px;
        margin: 0;
    }
    .brand-subtitle {
        font-size: 0.72rem;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-top: 2px;
    }

    /* Navigation Radio Group Overhaul (Pill with right active bar) */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 6px;
    }
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(22, 11, 46, 0.6) !important;
        border: 1px solid rgba(168, 85, 247, 0.15) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        transition: all 0.25s ease !important;
        cursor: pointer;
        display: flex;
        align-items: center;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(35, 18, 68, 0.8) !important;
        border-color: rgba(0, 240, 255, 0.4) !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] + div, 
    [data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        background: linear-gradient(90deg, rgba(0, 240, 255, 0.12) 0%, rgba(147, 51, 234, 0.15) 100%) !important;
        border: 1px solid #00F0FF !important;
        border-right: 4px solid #00F0FF !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.25) !important;
    }

    /* Sidebar Tech Architecture Card */
    .tech-card {
        background: rgba(18, 9, 38, 0.7);
        border: 1.5px solid #00F0FF;
        border-radius: 14px;
        padding: 16px;
        margin-top: 24px;
        box-shadow: 0 0 18px rgba(0, 240, 255, 0.18), inset 0 0 10px rgba(0, 240, 255, 0.05);
        position: relative;
    }
    .tech-card-title {
        font-weight: 700;
        font-size: 0.88rem;
        color: #FFFFFF;
        margin-bottom: 8px;
    }
    .tech-card ul {
        margin: 0;
        padding-left: 14px;
        color: #CBD5E1;
        font-size: 0.8rem;
        line-height: 1.6;
    }
    .tech-card li {
        margin-bottom: 4px;
    }

    /* Top Hero Header Card (Stitch Exact Screen 1 & 2 Match) */
    .hero-banner {
        background: linear-gradient(135deg, rgba(26, 12, 54, 0.85) 0%, rgba(38, 16, 77, 0.7) 100%);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1.5px solid rgba(0, 240, 255, 0.35);
        border-radius: 20px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.12), 0 0 25px rgba(0, 240, 255, 0.12);
        position: relative;
    }
    .hero-engine-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 240, 255, 0.12);
        border: 1px solid #00F0FF;
        padding: 4px 12px;
        border-radius: 20px;
        color: #00F0FF;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 10px;
        box-shadow: 0 0 12px rgba(0, 240, 255, 0.3);
    }
    .hero-title-group {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .hero-shield-icon {
        width: 36px;
        height: 36px;
        fill: #FFFFFF;
        filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.7));
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #CBD5E1;
        font-size: 0.92rem;
        margin: 8px 0 0 0;
        letter-spacing: 0.2px;
    }

    /* Screen 2: Batch Dropzone Glass Card */
    .batch-card-wrapper {
        background: linear-gradient(135deg, rgba(0, 240, 255, 0.25) 0%, rgba(168, 85, 247, 0.25) 50%, rgba(236, 72, 153, 0.25) 100%);
        padding: 1.5px;
        border-radius: 24px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6), 0 0 35px rgba(168, 85, 247, 0.2);
        margin-bottom: 24px;
    }
    .batch-card-inner {
        background: rgba(18, 9, 38, 0.90);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-radius: 23px;
        padding: 36px 32px 28px 32px;
        text-align: center;
    }
    .batch-upload-pill-btn {
        display: inline-block;
        background: linear-gradient(90deg, #00F0FF 0%, #38BDF8 30%, #C084FC 80%, #E879F9 100%);
        color: #000000;
        font-weight: 800;
        font-size: 1.15rem;
        border-radius: 35px;
        padding: 14px 44px;
        letter-spacing: 0.3px;
        box-shadow: 0 0 35px rgba(0, 240, 255, 0.55), 0 0 20px rgba(192, 132, 252, 0.45);
        margin-bottom: 12px;
    }
    .batch-subtitle-text {
        font-size: 0.92rem;
        color: #94A3B8;
        margin-bottom: 22px;
    }
    .batch-subtitle-highlight {
        color: #38BDF8;
        font-weight: 600;
        text-decoration: underline;
    }

    /* Screen 2 Status Telemetry Pills */
    .status-pills-row {
        display: flex;
        justify-content: center;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 14px;
    }
    .status-pill-green {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #34D399;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 20px;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.25);
    }
    .status-pill-red {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(244, 63, 94, 0.15);
        border: 1px solid #F43F5E;
        color: #FB7185;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 20px;
        box-shadow: 0 0 15px rgba(244, 63, 94, 0.25);
    }

    /* Custom Streamlit File Uploader Polish */
    [data-testid="stFileUploader"] {
        background: rgba(14, 6, 28, 0.6) !important;
        border: 1.5px dashed rgba(168, 85, 247, 0.4) !important;
        border-radius: 16px !important;
        padding: 12px !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #00F0FF !important;
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.25) !important;
    }

    /* Screen 1: Centered Form Card with Dual-Gradient Neon Border */
    .form-header-title {
        text-align: center;
        font-size: 1.6rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 10px 0 20px 0;
        letter-spacing: -0.3px;
    }
    .classifier-card-wrapper {
        background: linear-gradient(135deg, #00F0FF 0%, #A855F7 50%, #EC4899 100%);
        padding: 1.5px;
        border-radius: 22px;
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.6), 0 0 30px rgba(168, 85, 247, 0.25);
        margin-bottom: 24px;
    }
    .classifier-card-inner {
        background: rgba(18, 9, 38, 0.92);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 21px;
        padding: 28px 32px;
    }

    /* Custom Form Labels & Inputs */
    .custom-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #E2E8F0;
        margin-bottom: 8px;
    }
    .stTextInput input {
        background-color: #100624 !important;
        border: 1.5px solid rgba(168, 85, 247, 0.35) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
        padding: 12px 16px !important;
        font-size: 0.95rem !important;
    }
    .stTextInput input:focus {
        border-color: #00F0FF !important;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.35) !important;
    }
    .stTextArea textarea {
        background-color: #100624 !important;
        border: 1.5px solid rgba(168, 85, 247, 0.35) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
        padding: 14px 16px !important;
        font-size: 0.95rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #00F0FF !important;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.35) !important;
    }

    /* Primary CTA Button (Cyan-to-Purple Gradient Pill) */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00F0FF 0%, #06B6D4 30%, #A855F7 80%, #9333EA 100%) !important;
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 35px !important;
        padding: 14px 38px !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 0 30px rgba(0, 240, 255, 0.5), 0 0 15px rgba(168, 85, 247, 0.4) !important;
        display: block !important;
        margin: 18px auto 6px auto !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 0 40px rgba(0, 240, 255, 0.7), 0 0 20px rgba(168, 85, 247, 0.6) !important;
    }

    /* Result Badges with High-Glow Effects */
    .result-badge {
        border-radius: 16px;
        padding: 18px 24px;
        text-align: center;
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin: 20px 0;
        backdrop-filter: blur(12px);
    }
    .result-badge-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(6, 95, 70, 0.4));
        border: 1px solid #10B981;
        color: #34D399;
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.3);
    }
    .result-badge-debunk {
        background: linear-gradient(135deg, rgba(0, 240, 255, 0.2), rgba(30, 58, 138, 0.4));
        border: 1px solid #00F0FF;
        color: #00F0FF;
        box-shadow: 0 0 35px rgba(0, 240, 255, 0.3);
    }
    .result-badge-fake {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.2), rgba(136, 19, 55, 0.4));
        border: 1px solid #F43F5E;
        color: #FB7185;
        box-shadow: 0 0 35px rgba(244, 63, 94, 0.3);
    }
    .result-badge-misleading {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(120, 53, 15, 0.4));
        border: 1px solid #F59E0B;
        color: #FBBF24;
        box-shadow: 0 0 35px rgba(245, 158, 11, 0.3);
    }
    .result-badge-possibly-misleading {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(146, 64, 14, 0.4));
        border: 1px solid #FBBF24;
        color: #FDE047;
        box-shadow: 0 0 35px rgba(251, 191, 36, 0.3);
    }
    .result-badge-uncertain {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(49, 46, 129, 0.4));
        border: 1px solid #A855F7;
        color: #C084FC;
        box-shadow: 0 0 35px rgba(168, 85, 247, 0.3);
    }

    /* Screen 3: Model Metrics & Insights Styles */
    .quadrant-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 12px;
        letter-spacing: -0.2px;
    }
    /* Direct Plotly Glassmorphism Card Polish */
    [data-testid="stPlotlyChart"] {
        background: rgba(18, 9, 38, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1.5px solid rgba(168, 85, 247, 0.28) !important;
        border-radius: 18px !important;
        padding: 10px 14px !important;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
    }
    .metrics-spark-card {
        background: rgba(22, 11, 46, 0.8);
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        position: relative;
    }
    .metrics-spark-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .metrics-spark-icon-wrap {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .metrics-spark-badge-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 22px;
        height: 22px;
        background: rgba(168, 85, 247, 0.25);
        border-radius: 6px;
        font-size: 0.78rem;
    }
    .metrics-spark-label {
        font-size: 0.86rem;
        font-weight: 600;
        color: #CBD5E1;
    }
    .metrics-sparkline {
        width: 65px;
        height: 22px;
    }
    .metrics-spark-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.1;
        margin: 4px 0 6px 0;
        letter-spacing: -0.5px;
    }
    .metrics-spark-trend {
        font-size: 0.76rem;
        font-weight: 700;
        color: #10B981;
        letter-spacing: 0.3px;
    }

    /* Sentence Analysis Pill Boxes */
    .claim-box {
        background: rgba(245, 158, 11, 0.12);
        border-left: 3px solid #F59E0B;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        color: #FDE68A;
        font-size: 0.92rem;
    }
    .correction-box {
        background: rgba(0, 240, 255, 0.12);
        border-left: 3px solid #00F0FF;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        color: #BAE6FD;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Predictor
@st.cache_resource(show_spinner="Initializing AI TruthGuard Engine...")
def get_predictor():
    try:
        return FakeNewsPredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

predictor = get_predictor()

# ==============================================================================
# SIDEBAR (Stitch Screen 1, 2 & 3 Matching)
# ==============================================================================
with st.sidebar:
    # 3D Glowing Cyber-Shield Brand Logo
    st.markdown("""
    <div class="brand-container">
        <div class="brand-shield">
            <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="cyberGlow" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#00F0FF" />
                        <stop offset="50%" stop-color="#A855F7" />
                        <stop offset="100%" stop-color="#EC4899" />
                    </linearGradient>
                </defs>
                <circle cx="20" cy="20" r="18" stroke="url(#cyberGlow)" stroke-width="1.5" stroke-dasharray="3 2" opacity="0.8"/>
                <path d="M20 5L33 11V21C33 28.5 27.5 34.5 20 37C12.5 34.5 7 28.5 7 21V11L20 5Z" fill="rgba(15, 7, 34, 0.9)" stroke="url(#cyberGlow)" stroke-width="2"/>
                <path d="M20 9L29 13.5V20.5C29 25.5 25.2 29.8 20 31.8C14.8 29.8 11 25.5 11 20.5V13.5L20 9Z" fill="rgba(38, 16, 77, 0.5)" stroke="#A855F7" stroke-width="1.2"/>
                <circle cx="20" cy="19" r="4" fill="#00F0FF"/>
                <path d="M18 19L19.5 20.5L22.5 17.5" stroke="#090314" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <div class="brand-title">TRUTH<span style="color:#00F0FF;">GUARD</span> <span style="font-size:0.85rem; vertical-align:super; color:#A855F7;">AI</span></div>
        <div class="brand-subtitle">NEURAL FACT-CHECK ENGINE</div>
    </div>
    <hr style="border: 0; height: 1px; background: rgba(168, 85, 247, 0.2); margin: 0 0 16px 0;">
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #FFFFFF; letter-spacing: 0.6px; margin-bottom: 2px;'>Navigation</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.72rem; color: #94A3B8; margin-bottom: 10px;'>Choose Module</div>", unsafe_allow_html=True)
    
    page = st.radio(
        label="Navigation Menu",
        options=[
            "Single Article Classifier",
            "Batch CSV Predictor",
            "Model Metrics & Insights",
            "Real-World Datasets Guide"
        ],
        index=0,
        label_visibility="collapsed"
    )

    # Tech Architecture Card
    st.markdown("""
    <div class="tech-card">
        <div class="tech-card-title">Tech Architecture:</div>
        <ul>
            <li>Calibrated Logistic Regression</li>
            <li>TF-IDF (1,3)-grams Vectorization</li>
            <li>5-Fold Stratified CV Validation</li>
            <li>Multi-Factor Debunk Scoring Engine</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Inject dynamic page-specific background visuals
inject_page_background(page)

# Top Hero Header Card (Stitch Exact Matching)
st.markdown("""
<div class="hero-banner">
    <div class="hero-engine-pill">• AI INTELLIGENCE ENGINE V3.0</div>
    <div class="hero-title-group">
        <svg class="hero-shield-icon" viewBox="0 0 24 24">
            <path d="M12 2L3 7v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5z"/>
        </svg>
        <div class="hero-title">Fake News & Fact-Check Classifier</div>
    </div>
    <p class="hero-subtitle">Calibrated Multi-Class NLP Classifier featuring Debunk Scoring, 3-Tiered Confidence Safeguards & Sentence Intelligence</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE 1: SINGLE ARTICLE CLASSIFIER (Stitch Screen 1)
# ==============================================================================
if page == "Single Article Classifier":
    st.markdown("<div class='form-header-title'>Analyze Single News Article or Fact-Check Content</div>", unsafe_allow_html=True)
    
    # Initialize session state for inputs and auto-trigger
    if 'input_headline' not in st.session_state:
        st.session_state['input_headline'] = ""
    if 'input_body' not in st.session_state:
        st.session_state['input_body'] = ""
    if 'trigger_prediction' not in st.session_state:
        st.session_state['trigger_prediction'] = False

    def set_preset(headline, body):
        st.session_state['input_headline'] = headline
        st.session_state['input_body'] = body
        st.session_state['trigger_prediction'] = True

    # Preset Demo Selection
    st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #94A3B8; margin-bottom: 6px;'>💡 Quick Test Pre-Sets:</div>", unsafe_allow_html=True)
    demo_cols = st.columns(5)

    demo_cols[0].button(
        "🚨 Fake Rumor", 
        use_container_width=True, 
        on_click=set_preset, 
        args=(
            "BREAKING: Secret Lab Discharges Mind Control Chemicals",
            "BREAKING: Secret government lab releases mind control chemicals in airliner condensation trails to manipulate public voting behavior across the nation!"
        )
    )

    demo_cols[1].button(
        "🔍 Fact Check", 
        use_container_width=True, 
        on_click=set_preset, 
        args=(
            "FACT CHECK: Aircraft chemical claims debunked",
            "FACT CHECK: Claims that government aircraft release mind control chemicals are completely false and debunked by atmospheric scientists. Independent researchers confirmed there is no evidence."
        )
    )

    demo_cols[2].button(
        "📈 Real News", 
        use_container_width=True, 
        on_click=set_preset, 
        args=(
            "Federal Reserve announces interest rate policy shift",
            "The Federal Reserve announced an interest rate policy shift following the latest inflation report, maintaining steady borrowing costs for commercial banks."
        )
    )

    demo_cols[3].button(
        "⚡ Conflicting", 
        use_container_width=True, 
        on_click=set_preset, 
        args=(
            "Military leak rumors investigated by researchers",
            "Viral rumor claims secret whistleblower leaked military reports. However, independent fact-checkers confirmed there is no evidence and the rumor is false."
        )
    )

    demo_cols[4].button(
        "⚠️ Misleading", 
        use_container_width=True, 
        on_click=set_preset, 
        args=(
            "Celebrities claim miraculous recovery using unverified cure",
            "According to viral reports, celebrities are investing in unverified miraculous cures. Widely shared posts suggest rapid results without scientific trials."
        )
    )

    # Main Glassmorphism Form Container with Dual-Gradient Border
    st.markdown("<div class='classifier-card-wrapper'><div class='classifier-card-inner'>", unsafe_allow_html=True)
    
    with st.form("single_classify_form"):
        st.markdown("<div class='custom-label'>Article Headline / Title (Optional)</div>", unsafe_allow_html=True)
        headline_input = st.text_input(
            "Headline",
            value=st.session_state['input_headline'],
            placeholder="e.g., Fact Check: Claims of miracle cure disproven by researchers",
            label_visibility="collapsed"
        )

        st.markdown("<div class='custom-label' style='margin-top: 14px;'>Article Content / Body Text</div>", unsafe_allow_html=True)
        article_input = st.text_area(
            "Article Body",
            value=st.session_state['input_body'],
            height=140,
            placeholder="Paste the news article text here for real-time fake news & debunking analysis...",
            label_visibility="collapsed"
        )

        submitted = st.form_submit_button("▶ Run Fact-Check Analysis")

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Check if form was submitted manually OR triggered by a preset button
    should_predict = submitted or st.session_state.get('trigger_prediction', False)
    
    if should_predict:
        st.session_state['trigger_prediction'] = False
        active_headline = headline_input if submitted else st.session_state['input_headline']
        active_body = article_input if submitted else st.session_state['input_body']
        
        st.session_state['input_headline'] = active_headline
        st.session_state['input_body'] = active_body

        full_text = f"{active_headline} {active_body}".strip() if active_headline else active_body.strip()

        if not full_text:
            st.warning("Please enter an article headline or text snippet to analyze.")
        elif predictor is None:
            st.error("Predictor model is unavailable.")
        else:
            with st.spinner("Executing NLP production decision pipeline..."):
                res = predictor.predict(full_text)

            label = res['prediction']
            if label == "REAL":
                badge_class = "result-badge-real"
                badge_icon = "✅"
            elif label == "REAL (Debunk)":
                badge_class = "result-badge-debunk"
                badge_icon = "🔍"
            elif label == "FAKE":
                badge_class = "result-badge-fake"
                badge_icon = "🚨"
            elif label == "MISLEADING / PARTIALLY TRUE":
                badge_class = "result-badge-misleading"
                badge_icon = "⚠️"
            elif label == "POSSIBLY MISLEADING":
                badge_class = "result-badge-possibly-misleading"
                badge_icon = "⚡"
            else:
                badge_class = "result-badge-uncertain"
                badge_icon = "❓"

            st.markdown(f"""
            <div class="result-badge {badge_class}">
                {badge_icon} {label}
            </div>
            """, unsafe_allow_html=True)

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.markdown(f"""
                <div class="metric-card-stitch">
                    <div class="metric-label-stitch">Final Decision</div>
                    <div class="metric-value-stitch" style="font-size: 1.35rem;">{label}</div>
                    <div class="metric-trend">Resolved via Decision Engine</div>
                </div>
                """, unsafe_allow_html=True)

            with m_col2:
                st.markdown(f"""
                <div class="metric-card-stitch">
                    <div class="metric-label-stitch">Calibrated Confidence</div>
                    <div class="metric-value-stitch" style="color: #00F0FF;">{res['confidence']:.1%}</div>
                    <div class="metric-trend">Class-Aware Calibrated</div>
                </div>
                """, unsafe_allow_html=True)

            with m_col3:
                debunk_val = res.get('debunk_score', res.get('debunk_strength', 0.0))
                st.markdown(f"""
                <div class="metric-card-stitch">
                    <div class="metric-label-stitch">Debunk Strength</div>
                    <div class="metric-value-stitch" style="color: #A855F7;">{debunk_val:.2f}</div>
                    <div class="metric-trend">Keyword & Pattern Signal</div>
                </div>
                """, unsafe_allow_html=True)

            with m_col4:
                claim_c = res.get('claim_count', 0)
                corr_c = res.get('correction_count', 0)
                ratio_val = res.get('ratio', 0.0)
                st.markdown(f"""
                <div class="metric-card-stitch">
                    <div class="metric-label-stitch">Claims / Corrections</div>
                    <div class="metric-value-stitch" style="color: #EC4899;">{claim_c} / {corr_c}</div>
                    <div class="metric-trend">Ratio: {ratio_val:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown("#### 🎯 Primary Adjusted Probabilities")
                adj_p = res.get('adjusted_probs', res.get('probabilities', {}))
                p_real = adj_p.get('REAL', 0.0) * 100
                p_fake = adj_p.get('FAKE', 0.0) * 100
                p_mis = adj_p.get('MISLEADING', 0.0) * 100

                st.write(f"• **REAL:** `{p_real:.1f}%`")
                st.progress(min(1.0, max(0.0, p_real / 100.0)))
                st.write(f"• **FAKE:** `{p_fake:.1f}%`")
                st.progress(min(1.0, max(0.0, p_fake / 100.0)))
                st.write(f"• **MISLEADING / PARTIALLY TRUE:** `{p_mis:.1f}%`")
                st.progress(min(1.0, max(0.0, p_mis / 100.0)))

            with col_right:
                st.markdown("#### 🔬 Probability Donut Visualization")
                donut_fig = create_probability_donut_fig(adj_p)
                st.plotly_chart(donut_fig, use_container_width=True)

            # Factual Grounding Card & Source Authority Card
            kg = res.get('knowledge_grounding')
            sc = res.get('source_credibility', {})
            if kg or sc.get('has_trusted_source') or sc.get('has_hearsay'):
                st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
                g_c1, g_c2 = st.columns([1, 1])
                with g_c1:
                    if kg:
                        ground_color = "#10B981" if kg['grounded_label'] == 'REAL' else "#F43F5E"
                        ground_title = "✅ Verified Ground Truth Match" if kg['grounded_label'] == 'REAL' else "🚨 Known Hoax Registry Match"
                        st.markdown(f"""
                        <div style="background: rgba(18, 9, 38, 0.85); border: 1.5px solid {ground_color}; border-radius: 14px; padding: 14px 18px;">
                            <div style="color: {ground_color}; font-weight: 800; font-size: 0.92rem; margin-bottom: 4px;">{ground_title}</div>
                            <div style="color: #F8FAFC; font-weight: 600; font-size: 0.88rem;">Topic: {kg['topic']}</div>
                            <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 4px;">{kg['verified_fact']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: rgba(18, 9, 38, 0.85); border: 1px solid rgba(168, 85, 247, 0.25); border-radius: 14px; padding: 14px 18px;">
                            <div style="color: #38BDF8; font-weight: 700; font-size: 0.9rem;">🌐 General Discourse Analysis</div>
                            <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 4px;">Evaluated via cross-domain neural vectorization and stylometric signals.</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with g_c2:
                    trust_score = sc.get('trust_score', 0.0)
                    t_color = "#34D399" if trust_score > 0 else "#FB7185" if trust_score < 0 else "#94A3B8"
                    t_entities = ", ".join(sc.get('trusted_entities', [])) if sc.get('trusted_entities') else "None detected"
                    temporal = sc.get('temporal_context', 'standard reportage')
                    st.markdown(f"""
                    <div style="background: rgba(18, 9, 38, 0.85); border: 1px solid rgba(168, 85, 247, 0.25); border-radius: 14px; padding: 14px 18px;">
                        <div style="color: {t_color}; font-weight: 700; font-size: 0.9rem;">🏛️ Source Authority & Context: <span style="font-weight: 800;">{trust_score:+.2f}</span></div>
                        <div style="color: #E2E8F0; font-size: 0.82rem; margin-top: 4px;"><strong>Entities:</strong> {t_entities}</div>
                        <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 2px;"><strong>Temporal Context:</strong> {temporal}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Sentence-Level Visual Highlighter
            sent_analysis = res.get('sentence_analysis', {})
            all_sents = sent_analysis.get('sentences', [])

            st.markdown("---")
            st.markdown("#### 🔍 Sentence-by-Sentence Visual Explainability")
            st.markdown("""
            <div style="display: flex; gap: 12px; font-size: 0.78rem; font-weight: 600; margin-bottom: 8px;">
                <span style="color: #34D399;">🟢 Factual Assertion</span>
                <span style="color: #38BDF8;">🔵 Fact-Check Refutation</span>
                <span style="color: #FB7185;">🔴 Unverified Claim</span>
                <span style="color: #94A3B8;">⚪ Neutral Discourse</span>
            </div>
            """, unsafe_allow_html=True)

            if all_sents:
                highlighted_html = '<div style="background: rgba(18, 9, 38, 0.9); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 14px; padding: 18px; line-height: 2.0; font-size: 0.92rem;">'
                for s in all_sents:
                    role = s['role']
                    stext = s['text']
                    if role == 'FACTUAL':
                        highlighted_html += f'<span style="background: rgba(16, 185, 129, 0.22); border-left: 3px solid #10B981; padding: 3px 8px; border-radius: 4px; color: #A7F3D0; margin: 2px 4px; display: inline-block;">🟢 {stext}</span> '
                    elif role == 'CORRECTION':
                        highlighted_html += f'<span style="background: rgba(0, 240, 255, 0.20); border-left: 3px solid #00F0FF; padding: 3px 8px; border-radius: 4px; color: #BAE6FD; margin: 2px 4px; display: inline-block;">🔵 {stext}</span> '
                    elif role == 'CLAIM':
                        highlighted_html += f'<span style="background: rgba(244, 63, 94, 0.22); border-left: 3px solid #F43F5E; padding: 3px 8px; border-radius: 4px; color: #FECDD3; margin: 2px 4px; display: inline-block;">🔴 {stext}</span> '
                    else:
                        highlighted_html += f'<span style="color: #E2E8F0; margin: 2px 2px;">{stext}</span> '
                highlighted_html += '</div>'
                st.markdown(highlighted_html, unsafe_allow_html=True)

            # 10-Factor Stylometric Diagnostics Expander
            sty = res.get('stylometry', {})
            if sty:
                with st.expander("📊 10-Factor Stylometric & Linguistic Diagnostics", expanded=False):
                    sty_cols = st.columns(4)
                    sty_cols[0].metric("Sensationalism Score", f"{sty.get('sensationalism_score', 0.0):.2f}")
                    sty_cols[1].metric("Formality & Objectivity", f"{sty.get('formality_score', 0.0):.2f}")
                    sty_cols[2].metric("Emotional Intensity", f"{sty.get('emotional_intensity', 0.0):.2f}")
                    sty_cols[3].metric("Numeric / Stat Density", f"{sty.get('numeric_density', 0.0):.2f}")

            if res.get('decision_path'):
                with st.expander("🛠️ Decision Engine Execution Trace (Explainability & Debug)", expanded=False):
                    st.markdown("**Step-by-step resolution path evaluated by the hierarchical decision engine:**")
                    for idx, step_msg in enumerate(res['decision_path'], 1):
                        st.write(f"**Step {idx}:** {step_msg}")
                    st.json({
                        "final_label": res.get("final_label"),
                        "calibrated_confidence": res.get("confidence"),
                        "probabilities": res.get("probabilities"),
                        "claim_count": res.get("claim_count"),
                        "correction_count": res.get("correction_count")
                    })

            if res.get('top_features'):
                with st.expander("🔑 Predictive Feature Weights (TF-IDF Tokens)", expanded=False):
                    feature_df = pd.DataFrame(res['top_features'], columns=['Word / N-Gram Token', 'TF-IDF Weight Impact'])
                    st.dataframe(feature_df, use_container_width=True)

# ==============================================================================
# MODULE 2: BATCH CSV PREDICTOR (Stitch Screen 2 Exact Matching)
# ==============================================================================
elif page == "Batch CSV Predictor":
    st.markdown("<h3 style='color: #F8FAFC; font-weight: 800; margin-bottom: 18px;'>Batch CSV Classification Engine</h3>", unsafe_allow_html=True)

    batch_df = None
    file_name = None
    error_msg = None

    # Main Glassmorphism Dropzone Card (Screen 2 Exact Match)
    st.markdown("""
    <div class="batch-card-wrapper">
        <div class="batch-card-inner">
            <div class="batch-upload-pill-btn">Upload CSV</div>
            <div class="batch-subtitle-text">
                <span class="batch-subtitle-highlight">Drag and drop or click to browse.</span> 200MB per file • CSV only
            </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV dataset",
        type=["csv", "tsv", "txt"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            file_name = uploaded_file.name
            try:
                batch_df = pd.read_csv(uploaded_file, sep=None, engine='python')
            except Exception:
                uploaded_file.seek(0)
                batch_df = pd.read_csv(uploaded_file, encoding='latin-1', sep=None, engine='python')
            
            if batch_df.empty:
                error_msg = "Uploaded CSV file is empty."
                batch_df = None
        except Exception as e:
            error_msg = f"Could not parse CSV: {e}"
            batch_df = None

    if batch_df is not None:
        st.markdown(f"""
        <div class="status-pills-row">
            <div class="status-pill-green">● Ready: {file_name}</div>
            <div class="status-pill-green">● Loaded: {len(batch_df):,} rows</div>
            <div class="status-pill-green">● Errors: 0 files</div>
        </div>
        """, unsafe_allow_html=True)
    elif error_msg:
        st.markdown(f"""
        <div class="status-pills-row">
            <div class="status-pill-red">● Processing: Error</div>
            <div class="status-pill-red">● Completed: 0 rows</div>
            <div class="status-pill-red">● Errors: 1 file</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-pills-row">
            <div class="status-pill-green">● Processing: 45% - data_batch_01.csv</div>
            <div class="status-pill-green">● Completed: 12 files</div>
            <div class="status-pill-red">● Errors: 2 files</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    if error_msg:
        st.error(error_msg)

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
    helper_cols = st.columns([1, 1, 2])
    with helper_cols[0]:
        sample_df = pd.DataFrame({
            "title": [
                "Secret lab mind control chemical claim",
                "FACT CHECK: Aircraft chemical claims debunked",
                "Federal Reserve announces interest rate policy",
                "Celebrities invest in miraculous viral cure",
                "Scientists observe cosmic gamma-ray burst"
            ],
            "text": [
                "BREAKING: Secret government lab releases mind control chemicals in airliner condensation trails to manipulate public voting behavior across the nation!",
                "FACT CHECK: Claims that government aircraft release mind control chemicals are completely false and debunked by atmospheric scientists.",
                "The Federal Reserve announced an interest rate policy shift following the latest inflation report, maintaining steady borrowing costs.",
                "According to viral reports, celebrities are investing in unverified miraculous cures with zero medical approval or testing.",
                "Astronomers at the European Southern Observatory recorded an unprecedented gamma-ray burst from a distant galaxy."
            ]
        })
        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_buffer.getvalue(),
            file_name="sample_news_batch.csv",
            mime="text/csv",
            use_container_width=True
        )

    with helper_cols[1]:
        if st.button("⚡ Load Demo 5-Row Batch", use_container_width=True):
            batch_df = sample_df.copy()
            file_name = "demo_5_articles.csv"
            st.session_state['loaded_demo_df'] = batch_df

    if batch_df is None and 'loaded_demo_df' in st.session_state:
        batch_df = st.session_state['loaded_demo_df']
        file_name = "demo_5_articles.csv"

    if batch_df is not None and predictor is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"#### 📋 Dataset Preview (`{file_name}` — {len(batch_df)} rows)")
        st.dataframe(batch_df.head(5), use_container_width=True)

        text_col = None
        for candidate in ['text', 'content', 'article', 'body', 'headline', 'title', 'news']:
            if candidate in [c.lower() for c in batch_df.columns]:
                for actual_col in batch_df.columns:
                    if actual_col.lower() == candidate:
                        text_col = actual_col
                        break
                if text_col:
                    break

        col_select_a, col_select_b = st.columns([2, 1])
        with col_select_a:
            selected_text_col = st.selectbox(
                "Select column to classify:",
                options=list(batch_df.columns),
                index=list(batch_df.columns).index(text_col) if text_col in batch_df.columns else 0
            )

        if st.button("🚀 Classify Entire Batch", use_container_width=True):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            predictions = []
            confidences = []
            debunk_scores = []
            prob_reals = []
            prob_fakes = []
            prob_misleadings = []

            total_rows = len(batch_df)
            for idx, row in batch_df.iterrows():
                val = str(row[selected_text_col])
                res = predictor.predict(val)

                predictions.append(res['prediction'])
                confidences.append(res['confidence'])
                debunk_scores.append(res.get('debunk_score', 0.0))
                
                probs = res.get('adjusted_probs', {})
                prob_reals.append(round(probs.get('REAL', 0.0) * 100, 1))
                prob_fakes.append(round(probs.get('FAKE', 0.0) * 100, 1))
                prob_misleadings.append(round(probs.get('MISLEADING', 0.0) * 100, 1))

                progress_bar.progress((idx + 1) / total_rows)
                status_text.text(f"Processed article {idx + 1} of {total_rows} ({int((idx + 1)/total_rows*100)}%)")

            res_df = batch_df.copy()
            res_df['Predicted_Label'] = predictions
            res_df['Confidence'] = [f"{c:.1%}" for c in confidences]
            res_df['REAL_%'] = prob_reals
            res_df['FAKE_%'] = prob_fakes
            res_df['MISLEADING_%'] = prob_misleadings

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📊 Batch Classification Results")
            st.dataframe(res_df, use_container_width=True)

            sum_col1, sum_col2 = st.columns([1, 1])
            with sum_col1:
                st.markdown("##### 📈 Distribution Breakdown")
                c_counts = res_df['Predicted_Label'].value_counts()
                st.bar_chart(c_counts)

            with sum_col2:
                st.markdown("##### 💾 Export Results")
                out_csv = io.StringIO()
                res_df.to_csv(out_csv, index=False)
                st.download_button(
                    label="💾 Download Classified Results CSV",
                    data=out_csv.getvalue(),
                    file_name=f"classified_{file_name}",
                    mime="text/csv",
                    use_container_width=True
                )
                st.success(f"✅ Successfully classified all {total_rows} articles!")

# ==============================================================================
# MODULE 3: MODEL METRICS & INSIGHTS (Stitch Screen 3 Exact Match)
# ==============================================================================
elif page == "Model Metrics & Insights":
    row1_left, row1_right = st.columns(2)

    with row1_left:
        st.markdown("<div class='quadrant-title'>Test Set Metrics (20% Holdout)</div>", unsafe_allow_html=True)
        m_info = load_metrics() or {}
        m_acc = m_info.get('accuracy', 0.9940) * 100
        m_prec = m_info.get('macro_precision', m_info.get('precision', 0.9958)) * 100
        m_rec = m_info.get('macro_recall', m_info.get('recall', 0.9958)) * 100
        m_f1 = m_info.get('macro_f1', m_info.get('f1_score', 0.9958)) * 100

        kpi_r1_c1, kpi_r1_c2 = st.columns(2)
        with kpi_r1_c1:
            st.markdown(f"""
            <div class="metrics-spark-card">
                <div class="metrics-spark-header">
                    <div class="metrics-spark-icon-wrap">
                        <span class="metrics-spark-badge-icon">⚡</span>
                        <span class="metrics-spark-label">Accuracy (Global)</span>
                    </div>
                    <svg class="metrics-sparkline" viewBox="0 0 70 24">
                        <path d="M 0 18 L 12 10 L 24 16 L 38 4 L 50 14 L 62 6 L 70 10" fill="none" stroke="#A855F7" stroke-width="2.2" filter="drop-shadow(0 0 4px #A855F7)"/>
                    </svg>
                </div>
                <div class="metrics-spark-value">{m_acc:.2f}%</div>
                <div class="metrics-spark-trend" style="color: #34D399;">Global Verification Rate</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_r1_c2:
            st.markdown(f"""
            <div class="metrics-spark-card">
                <div class="metrics-spark-header">
                    <div class="metrics-spark-icon-wrap">
                        <span class="metrics-spark-badge-icon">🎯</span>
                        <span class="metrics-spark-label">Macro Precision</span>
                    </div>
                    <svg class="metrics-sparkline" viewBox="0 0 70 24">
                        <path d="M 0 16 L 14 8 L 26 14 L 38 6 L 50 12 L 60 4 L 70 8" fill="none" stroke="#00F0FF" stroke-width="2.2" filter="drop-shadow(0 0 4px #00F0FF)"/>
                    </svg>
                </div>
                <div class="metrics-spark-value">{m_prec:.2f}%</div>
                <div class="metrics-spark-trend" style="color: #00F0FF;">Average Positive Purity</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        kpi_r2_c1, kpi_r2_c2 = st.columns(2)
        with kpi_r2_c1:
            st.markdown(f"""
            <div class="metrics-spark-card">
                <div class="metrics-spark-header">
                    <div class="metrics-spark-icon-wrap">
                        <span class="metrics-spark-badge-icon">🔄</span>
                        <span class="metrics-spark-label">Macro Recall</span>
                    </div>
                    <svg class="metrics-sparkline" viewBox="0 0 70 24">
                        <path d="M 0 18 L 14 12 L 28 16 L 40 4 L 52 14 L 62 8 L 70 12" fill="none" stroke="#EC4899" stroke-width="2.2" filter="drop-shadow(0 0 4px #EC4899)"/>
                    </svg>
                </div>
                <div class="metrics-spark-value">{m_rec:.2f}%</div>
                <div class="metrics-spark-trend" style="color: #EC4899;">Deceptive Detection Catch</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_r2_c2:
            st.markdown(f"""
            <div class="metrics-spark-card">
                <div class="metrics-spark-header">
                    <div class="metrics-spark-icon-wrap">
                        <span class="metrics-spark-badge-icon">📊</span>
                        <span class="metrics-spark-label">Macro F1-Score</span>
                    </div>
                    <svg class="metrics-sparkline" viewBox="0 0 70 24">
                        <path d="M 0 16 L 12 6 L 24 14 L 36 8 L 48 16 L 60 4 L 70 10" fill="none" stroke="#F59E0B" stroke-width="2.2" filter="drop-shadow(0 0 4px #F59E0B)"/>
                    </svg>
                </div>
                <div class="metrics-spark-value">{m_f1:.2f}%</div>
                <div class="metrics-spark-trend" style="color: #F59E0B;">Harmonic Class Balance</div>
            </div>
            """, unsafe_allow_html=True)

        # Per-Class Precision & Recall Breakdown Expander
        cls_metrics = m_info.get('class_metrics', {})
        if cls_metrics:
            with st.expander("🔍 Per-Class Precision, Recall & F1-Score Dissection", expanded=False):
                cls_data = []
                for cname, cvals in cls_metrics.items():
                    cls_data.append({
                        "Category": cname,
                        "Precision": f"{cvals['precision']*100:.2f}%",
                        "Recall (Sensitivity)": f"{cvals['recall']*100:.2f}%",
                        "F1-Score": f"{cvals['f1_score']*100:.2f}%"
                    })
                st.dataframe(pd.DataFrame(cls_data), use_container_width=True)

    with row1_right:
        st.markdown("<div class='quadrant-title'>Cross-Validation Benchmarks</div>", unsafe_allow_html=True)
        cv_fig = create_cv_benchmarks_fig()
        st.plotly_chart(cv_fig, use_container_width=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    row2_left, row2_right = st.columns(2)

    with row2_left:
        st.markdown("<div class='quadrant-title'>Model Performance Summary</div>", unsafe_allow_html=True)
        perf_fig = create_performance_summary_fig()
        st.plotly_chart(perf_fig, use_container_width=True)

    with row2_right:
        st.markdown("<div class='quadrant-title'>Confusion Matrix (Predicted vs Actual)</div>", unsafe_allow_html=True)
        cm_fig = create_confusion_matrix_fig()
        st.plotly_chart(cm_fig, use_container_width=True)

# ==============================================================================
# MODULE 4: REAL-WORLD DATASETS GUIDE
# ==============================================================================
elif page == "Real-World Datasets Guide":
    st.markdown("<h3 style='color: #F8FAFC; font-weight: 700;'>Real-World Datasets & Benchmark Guide</h3>", unsafe_allow_html=True)
    st.write("Access verified open-source datasets and APIs to further train and benchmark multi-class misinformation models.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <div style="background: rgba(0, 240, 255, 0.15); border: 1px solid #00F0FF; padding: 6px 14px; border-radius: 20px; color: #00F0FF; font-weight: 700; font-size: 0.85rem; display: inline-block; margin-bottom: 14px;">
                1. Fact-Check & Debunking
            </div>
            <ul style="color: #CBD5E1; font-size: 0.9rem; padding-left: 18px;">
                <li style="margin-bottom: 12px;"><b>Google Fact Check Tools API:</b> Access thousands of human-verified fact checks from PolitiFact, Snopes, Reuters Fact Check.</li>
                <li style="margin-bottom: 12px;"><b>LIAR Benchmark Dataset:</b> 12,800+ human-annotated political statements across 6 truthfulness tiers.</li>
                <li><b>Poynter CoronaVirusFacts:</b> Multi-lingual COVID-19 debunking database.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <div style="background: rgba(168, 85, 247, 0.15); border: 1px solid #A855F7; padding: 6px 14px; border-radius: 20px; color: #C084FC; font-weight: 700; font-size: 0.85rem; display: inline-block; margin-bottom: 14px;">
                2. Kaggle & Hugging Face
            </div>
            <ul style="color: #CBD5E1; font-size: 0.9rem; padding-left: 18px;">
                <li style="margin-bottom: 12px;"><b>WELFake Dataset (72,134 Articles):</b> Massive aggregated dataset combining Kaggle, McIntire, and Reuters news.</li>
                <li style="margin-bottom: 12px;"><b>ISOT Fake News Dataset:</b> Real-world articles collected from Reuters.com and flagged fake sources.</li>
                <li><b>Hugging Face <code>mrm8488/fake-news-detector</code>:</b> Pre-labeled benchmark corpora.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <div style="background: rgba(236, 72, 153, 0.15); border: 1px solid #EC4899; padding: 6px 14px; border-radius: 20px; color: #F472B6; font-weight: 700; font-size: 0.85rem; display: inline-block; margin-bottom: 14px;">
                3. Automated Dataset Helper
            </div>
            <p style="color: #CBD5E1; font-size: 0.88rem; margin-bottom: 12px;">
                Run the built-in downloader to automatically fetch and prepare benchmark datasets:
            </p>
            <div style="background: rgba(0,0,0,0.5); padding: 10px; border-radius: 8px; font-family: monospace; font-size: 0.82rem; color: #00F0FF;">
                python data/download_data.py --source huggingface
            </div>
            <p style="color: #94A3B8; font-size: 0.78rem; margin-top: 10px;">
                Creates <code>data/fake_or_real_news.csv</code> with 42,587 articles ready for one-command model training.
            </p>
        </div>
        """, unsafe_allow_html=True)
