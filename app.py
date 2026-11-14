"""GlucoSense - Diabetes risk predictor.   Run:  streamlit run app.py"""
import base64
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import diet

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "model.joblib")
HERO_PATH = os.path.join(BASE, "assets", "hero.png")
PLATE_PATH = os.path.join(BASE, "assets", "plate.png")

st.set_page_config(page_title="GlucoSense | Diabetes Predictor", page_icon="🩺", layout="centered")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, label { font-family: 'Poppins', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; max-width: 980px; }
.hero { position: relative; border-radius: 22px; overflow: hidden; padding: 64px 44px; margin-bottom: 26px;
        background-size: cover; background-position: center; box-shadow: 0 12px 34px rgba(9,52,66,.25); }
.hero .tag { display: inline-block; background: rgba(255,255,255,.16); color: #e6fbfb; padding: 5px 14px;
             border-radius: 999px; font-size: .8rem; letter-spacing: .04em; margin-bottom: 14px; }
.hero h1 { color: #fff; font-size: 2.5rem; line-height: 1.15; margin: 0 0 10px 0; font-weight: 700; max-width: 520px; }
.hero p { color: #d3f1f2; font-size: 1.02rem; margin: 0; max-width: 480px; line-height: 1.6; }
.steps { display: flex; gap: 14px; margin-bottom: 26px; flex-wrap: wrap; }
.step { flex: 1; min-width: 200px; background: #fff; border: 1px solid #e3eef0; border-radius: 16px;
        padding: 16px 18px; box-shadow: 0 2px 10px rgba(14,124,134,.06); }
.step .n { width: 30px; height: 30px; border-radius: 50%; background: #0e7c86; color: #fff; display: inline-flex;
           align-items: center; justify-content: center; font-weight: 600; margin-bottom: 8px; }
.step b { display: block; color: #0b3b4a; } .step span { color: #5c747a; font-size: .88rem; }
.sec-title { font-size: 1.35rem; font-weight: 600; color: #0b3b4a; margin: 6px 0 2px 0; }
.sec-sub { color: #5c747a; margin-bottom: 14px; font-size: .95rem; }
div[data-testid="stForm"] { border: 1px solid #dcebed; border-radius: 20px; padding: 26px 26px 14px 26px;
                            background: #fbfefe; box-shadow: 0 4px 18px rgba(14,124,134,.07); }
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg,#0b3b4a,#0e8c94); color: #fff; border: 0; border-radius: 12px;
    padding: .7rem 1rem; font-weight: 600; font-size: 1.05rem; }
div[data-testid="stFormSubmitButton"] > button:hover { filter: brightness(1.08); color: #fff; }
.verdict { border-radius: 20px; padding: 26px 30px; margin: 26px 0 14px 0; display: flex; gap: 22px; align-items: center; }
.verdict .ico { font-size: 3rem; }
.verdict h2 { margin: 0 0 4px 0; font-size: 1.7rem; color: inherit; font-weight: 700; }
.verdict p { margin: 0; color: inherit; opacity: .9; }
.v-yes { background: linear-gradient(135deg,#fff1e8,#ffe0cc); color: #9a3412; border: 1px solid #f8c9a8; }
.v-no { background: linear-gradient(135deg,#e8f8ee,#d3f1df); color: #166534; border: 1px solid #b4e3c7; }
.meter { background: #e9f0f1; border-radius: 999px; height: 14px; overflow: hidden; margin: 6px 0 4px 0; }
.meter > div { height: 100%; border-radius: 999px; }
.meter-label { display: flex; justify-content: space-between; color: #5c747a; font-size: .85rem; }
.chips { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0 6px 0; }
.chip { flex: 1; min-width: 180px; border-radius: 14px; padding: 12px 16px; background: #fff; border: 1px solid #e3eef0; }
.chip small { color: #5c747a; display: block; } .chip b { font-size: 1.15rem; color: #0b3b4a; }
.chip em { font-style: normal; font-size: .8rem; font-weight: 600; padding: 2px 10px; border-radius: 999px; float: right; margin-top: 4px; }
.ok em { background: #dcf5e6; color: #166534; } .warn em { background: #fdf0cc; color: #854d0e; }
.bad em { background: #fde1d5; color: #9a3412; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 14px; margin: 10px 0 8px 0; }
.food { background: #fff; border: 1px solid #e0efe6; border-top: 4px solid #2e9e5b; border-radius: 16px; padding: 16px;
        box-shadow: 0 2px 10px rgba(46,158,91,.07); }
.food .e { font-size: 2.2rem; } .food b { display: block; color: #0b3b4a; margin: 4px 0 2px 0; }
.food span { color: #5c747a; font-size: .85rem; line-height: 1.45; display: block; }
.food.lim { border-top-color: #e0a100; border-color: #f2e6c4; }
.food.av { border-top-color: #d9531e; border-color: #f6d9cb; }
.tl { display: flex; gap: 14px; align-items: flex-start; background: #fff; border: 1px solid #e3eef0; border-radius: 14px;
      padding: 12px 16px; margin-bottom: 10px; }
.tl .e { font-size: 1.7rem; } .tl b { color: #0b3b4a; display: block; } .tl span { color: #4d666c; font-size: .92rem; }
.note { background: #eaf6f7; border-left: 5px solid #0e7c86; border-radius: 10px; padding: 12px 16px; margin: 8px 0; color: #0b3b4a; }
.next { background: linear-gradient(135deg,#0b3b4a,#0e7c86); color: #fff; border-radius: 18px; padding: 20px 26px; margin: 22px 0 8px 0; }
.next b { font-size: 1.1rem; } .next p { margin: 6px 0 0 0; color: #d3f1f2; }
.foot { text-align: center; color: #7b9095; font-size: .8rem; margin: 34px 0 10px 0; }

/* Make form labels darker and easier to read */
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] label,
.stNumberInput label,
.stTextInput label,
.stSelectbox label,
.stSlider label {
    color: #173b43 !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Setting up the prediction model for the first time...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        import train
        train.main()
    return joblib.load(MODEL_PATH)


def b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def cards(items, css=""):
    """items: (emoji, title, text) tuples -> HTML grid."""
    html = "".join(
        f'<div class="food {css}"><div class="e">{e}</div><b>{t}</b><span>{d}</span></div>' for e, t, d in items)
    return f'<div class="grid">{html}</div>'


model = load_model()

# ---------------------------------------------------------------- Welcome
hero_style = f"background-image:url(data:image/png;base64,{b64(HERO_PATH)})" if os.path.exists(HERO_PATH) else \
    "background:linear-gradient(135deg,#0b3b4a,#0e8c94)"
st.markdown(f"""
<div class="hero" style="{hero_style}">
<div class="tag">AI-POWERED HEALTH SCREENING</div>
<h1>Welcome to GlucoSense</h1>
<p>Enter a few health details and find out in seconds whether you may have diabetes, along with a diet plan that can help.</p>
</div>
<div class="steps">
<div class="step"><div class="n">1</div><b>Enter your details</b><span>Age, weight, glucose and a few more values.</span></div>
<div class="step"><div class="n">2</div><b>Click Predict</b><span>Our model checks your risk instantly.</span></div>
<div class="step"><div class="n">3</div><b>Get your plan</b><span>See foods that help and what to avoid.</span></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- Form
st.markdown('<div class="sec-title">🧾 Your health details</div>'
            '<div class="sec-sub">Fill in everything you know. Values from a recent lab report give the best result.</div>',
            unsafe_allow_html=True)

gender = st.radio("Gender", ["Female", "Male"], horizontal=True)

with st.form("health_form"):
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age (years)", 15, 100, 35)
    height = c2.number_input("Height (cm)", 100.0, 220.0, 165.0, step=1.0)
    weight = c3.number_input("Weight (kg)", 25.0, 250.0, 70.0, step=0.5)

    c4, c5, c6 = st.columns(3)
    glucose = c4.number_input("Blood glucose (mg/dL)", 40.0, 300.0, 110.0, step=1.0,
                              help="Plasma glucose level from a blood test.")
    bp = c5.number_input("Blood pressure - lower value (mm Hg)", 30.0, 140.0, 76.0, step=1.0,
                         help="The lower (diastolic) number, e.g. 76 in 120/76.")
    family = c6.selectbox("Diabetes in family?", [
        "No one that I know of", "Grandparent, aunt or uncle", "Parent or sibling", "Both parents / several relatives"],
        help="Used to estimate your family history score.")

    preg = 0
    if gender == "Female":
        preg = st.number_input("Number of pregnancies", 0, 20, 0, help="Enter 0 if none.")

    with st.expander("Optional lab values (leave as 0 if you don't know)"):
        o1, o2 = st.columns(2)
        ins = o1.number_input("Insulin (mu U/ml)", 0.0, 900.0, 0.0)
        skin = o2.number_input("Skin fold thickness (mm)", 0.0, 100.0, 0.0)

    submitted = st.form_submit_button("🔍  Predict", use_container_width=True)

# ---------------------------------------------------------------- Result
if submitted:
    FAMILY_SCORE = {"No one that I know of": 0.2, "Grandparent, aunt or uncle": 0.4,
                    "Parent or sibling": 0.8, "Both parents / several relatives": 1.3}
    bmi = round(weight / ((height / 100) ** 2), 1)
    row = pd.DataFrame([{
        "Pregnancies": preg, "Glucose": glucose, "BloodPressure": bp,
        "SkinThickness": np.nan if skin == 0 else skin,
        "Insulin": np.nan if ins == 0 else ins,
        "BMI": bmi, "DiabetesPedigreeFunction": FAMILY_SCORE[family], "Age": age}])
    prob = float(model.predict_proba(row)[0, 1])
    pct = round(prob * 100, 1)
    has_diabetes = prob >= 0.5

    if has_diabetes:
        st.markdown(f"""
<div class="verdict v-yes"><div class="ico">⚠️</div><div>
<h2>You may have diabetes</h2>
<p>Based on your details, the model predicts diabetes (estimated probability {pct}%). Please confirm with a blood test and consult a doctor.</p>
</div></div>""", unsafe_allow_html=True)
        bar = "linear-gradient(90deg,#f59e0b,#d9531e)"
    else:
        st.markdown(f"""
<div class="verdict v-no"><div class="ico">✅</div><div>
<h2>You do not appear to have diabetes</h2>
<p>Based on your details, the model predicts no diabetes (estimated probability of diabetes {pct}%). Keep up your healthy habits.</p>
</div></div>""", unsafe_allow_html=True)
        bar = "linear-gradient(90deg,#34d399,#2e9e5b)"

    st.markdown(f"""
<div class="meter-label"><span>Diabetes probability</span><b>{pct}%</b></div>
<div class="meter"><div style="width:{max(pct, 2)}%;background:{bar}"></div></div>
<div class="meter-label"><span>0%</span><span>100%</span></div>
""", unsafe_allow_html=True)

    g_lab, g_lvl = diet.glucose_status(glucose)
    b_lab, b_lvl = diet.bmi_status(bmi)
    p_lab, p_lvl = diet.bp_status(bp)
    st.markdown(f"""
<div class="chips">
<div class="chip {g_lvl}"><small>Blood glucose</small><em>{g_lab}</em><b>{glucose:.0f} mg/dL</b></div>
<div class="chip {b_lvl}"><small>Body mass index</small><em>{b_lab}</em><b>{bmi}</b></div>
<div class="chip {p_lvl}"><small>Blood pressure (lower)</small><em>{p_lab}</em><b>{bp:.0f} mm Hg</b></div>
</div>""", unsafe_allow_html=True)

    for n in diet.personal_notes(glucose, bmi, bp):
        st.markdown(f'<div class="note">💡 {n}</div>', unsafe_allow_html=True)

    st.divider()

    if has_diabetes:
        st.markdown('<div class="sec-title">🥗 Foods that will help you</div>'
                    '<div class="sec-sub">Build your meals around these. They digest slowly and keep blood sugar steady.</div>',
                    unsafe_allow_html=True)
        st.markdown(cards(diet.EAT_MORE), unsafe_allow_html=True)

        left, right = st.columns([1, 1.1], gap="large")
        with left:
            if os.path.exists(PLATE_PATH):
                st.image(PLATE_PATH, caption="The diabetes-friendly plate", use_container_width=True)
        with right:
            st.markdown('<div class="sec-title">⚖️ Eat in small portions</div>', unsafe_allow_html=True)
            for e, t, d in diet.LIMIT:
                st.markdown(f'<div class="tl"><div class="e">{e}</div><div><b>{t}</b><span>{d}</span></div></div>',
                            unsafe_allow_html=True)
            st.markdown('<div class="sec-title">🚫 Avoid</div>', unsafe_allow_html=True)
            for e, t in diet.AVOID:
                st.markdown(f'<div class="tl"><div class="e">{e}</div><div><b>{t}</b></div></div>',
                            unsafe_allow_html=True)

        st.markdown('<div class="sec-title">🕒 A sample day of eating</div>'
                    '<div class="sec-sub">Adjust portions with your dietitian.</div>', unsafe_allow_html=True)
        for e, t, d in diet.DAILY_PLAN:
            st.markdown(f'<div class="tl"><div class="e">{e}</div><div><b>{t}</b><span>{d}</span></div></div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="sec-title">🏃 Daily habits that make a difference</div>', unsafe_allow_html=True)
        st.markdown(cards(diet.LIFESTYLE), unsafe_allow_html=True)
        st.markdown("""<div class="next"><b>Your next step</b>
<p>Get a fasting glucose and HbA1c test and book a visit with a doctor. A prediction is not a diagnosis.</p></div>""",
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="sec-title">🌱 Stay healthy: simple prevention tips</div>'
                    '<div class="sec-sub">Small habits today keep diabetes away tomorrow.</div>', unsafe_allow_html=True)
        left, right = st.columns([1, 1.1], gap="large")
        with left:
            if os.path.exists(PLATE_PATH):
                st.image(PLATE_PATH, caption="A balanced plate", use_container_width=True)
        with right:
            for e, t, d in diet.PREVENTION:
                st.markdown(f'<div class="tl"><div class="e">{e}</div><div><b>{t}</b><span>{d}</span></div></div>',
                            unsafe_allow_html=True)

st.markdown('<div class="foot">GlucoSense is an educational screening tool and not a medical diagnosis. '
            'Always consult a qualified doctor.</div>', unsafe_allow_html=True)
