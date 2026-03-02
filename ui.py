"""
╔══════════════════════════════════════════════════════════════╗
║        EstateIQ — House Price Prediction Engine              ║
║   Multiple Linear Regression · Streamlit · Professional UI  ║
╚══════════════════════════════════════════════════════════════╝
RUN: streamlit run house_price_app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="EstateIQ · House Price Predictor",
    page_icon="🏠", layout="wide",
    initial_sidebar_state="expanded",
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ── Theme Variables ───────────────────────────────────────────
if st.session_state.dark_mode:
    BG,SURFACE,SURFACE2       = "#0d1117","#161c2a","#1e2535"
    TEXT,TEXT2,MUTED          = "#e8edf5","#a8b4c8","#5a6a82"
    ACCENT,ACCENT2            = "#4f8ef7","#3bf0c4"
    RED,BORDER                = "#f55f5f","rgba(255,255,255,0.08)"
    PLOT_THEME                = "plotly_dark"
    THEME_ICON,THEME_LABEL    = "☀️","Light Mode"
else:
    BG,SURFACE,SURFACE2       = "#f7f5f0","#ffffff","#f0ede8"
    TEXT,TEXT2,MUTED          = "#ad6e6e","#4a4a4a","#8a8a8a"
    ACCENT,ACCENT2            = "#2563eb","#059669"
    RED,BORDER                = "#dc2626","rgba(0,0,0,0.08)"
    PLOT_THEME                = "plotly_white"
    THEME_ICON,THEME_LABEL    = "🌙","Dark Mode"

# ── Global CSS ────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;1,500&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html,body,[class*="css"],.stApp{{font-family:'DM Sans',sans-serif;background-color:{BG}!important;color:{TEXT}!important;}}
.stApp{{background:{BG}!important;}}
.block-container{{padding-top:0!important;max-width:1200px;}}
section[data-testid="stSidebar"]{{background:{SURFACE}!important;border-right:1px solid {BORDER};}}
section[data-testid="stSidebar"] *{{color:{TEXT}!important;}}
.stButton>button{{background:linear-gradient(135deg,{ACCENT},{ACCENT2})!important;color:#fff!important;font-family:'DM Sans',sans-serif!important;font-size:15px!important;font-weight:600!important;border:none!important;border-radius:12px!important;padding:14px 28px!important;width:100%!important;box-shadow:0 4px 20px rgba(37,99,235,0.3)!important;transition:all 0.2s!important;}}
.stButton>button:hover{{transform:translateY(-1px)!important;}}
.stNumberInput>div>div>input{{background:{SURFACE2}!important;color:{TEXT}!important;border:1px solid {BORDER}!important;border-radius:10px!important;font-family:'JetBrains Mono',monospace!important;font-size:16px!important;padding:12px 16px!important;}}
[data-testid="metric-container"]{{background:{SURFACE};border:1px solid {BORDER};border-radius:14px;padding:16px!important;}}
[data-testid="metric-container"] label{{font-family:'JetBrains Mono',monospace!important;font-size:10px!important;letter-spacing:2px!important;color:{MUTED}!important;text-transform:uppercase!important;}}
[data-testid="metric-container"] [data-testid="stMetricValue"]{{font-family:'Playfair Display',serif!important;font-size:32px!important;color:{ACCENT}!important;}}
hr{{border:none;border-top:1px solid {BORDER}!important;margin:24px 0!important;}}
#MainMenu,footer,header{{visibility:hidden;}}
::-webkit-scrollbar{{width:5px;}}
::-webkit-scrollbar-thumb{{background:{BORDER};border-radius:3px;}}

.nav-bar{{background:{SURFACE};border-bottom:1px solid {BORDER};padding:0 32px;height:64px;display:flex;align-items:center;justify-content:space-between;margin-bottom:32px;box-shadow:0 2px 20px rgba(0,0,0,0.05);}}
.nav-logo{{font-family:'Playfair Display',serif;font-size:24px;font-weight:700;color:{TEXT};}}
.nav-logo span{{color:{ACCENT};}}
.nav-tag{{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:3px;color:{MUTED};text-transform:uppercase;}}
.live-dot{{width:7px;height:7px;border-radius:50%;background:{ACCENT2};display:inline-block;animation:pulse 2s infinite;}}
@keyframes pulse{{0%,100%{{box-shadow:0 0 0 0 rgba(5,150,105,0.4);}}50%{{box-shadow:0 0 0 5px rgba(5,150,105,0);}}}}

.hero-card{{background:linear-gradient(135deg,{ACCENT} 0%,{ACCENT2} 100%);border-radius:20px;padding:32px 36px;color:white;position:relative;overflow:hidden;margin-bottom:20px;}}
.hero-card::before{{content:'';position:absolute;top:-40px;right:-40px;width:200px;height:200px;border-radius:50%;background:rgba(255,255,255,0.06);}}
.hero-card::after{{content:'';position:absolute;bottom:-60px;right:30px;width:160px;height:160px;border-radius:50%;background:rgba(255,255,255,0.04);}}
.hero-price-label{{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:3px;color:rgba(255,255,255,0.75);text-transform:uppercase;margin-bottom:12px;}}
.hero-price{{font-family:'Playfair Display',serif;font-size:72px;font-weight:700;line-height:1;letter-spacing:-2px;color:#fff;}}
.hero-unit{{font-family:'DM Sans',sans-serif;font-size:22px;font-weight:300;vertical-align:super;margin-left:4px;}}
.hero-sub{{font-size:14px;color:rgba(255,255,255,0.8);margin-top:10px;font-weight:300;}}

.section-label{{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:3px;color:{MUTED};text-transform:uppercase;margin:24px 0 12px;}}
.feature-chip{{background:{SURFACE2};border:1px solid {BORDER};border-radius:20px;padding:6px 14px;font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:1.5px;color:{TEXT2};text-transform:uppercase;display:inline-block;margin:4px;}}
.insight-box{{background:{SURFACE2};border-left:3px solid {ACCENT};border-radius:0 10px 10px 0;padding:14px 18px;margin:12px 0;font-size:13px;color:{TEXT2};font-weight:300;line-height:1.6;}}
.info-card{{background:{SURFACE};border:1px solid {BORDER};border-radius:16px;padding:24px 28px;margin-bottom:12px;box-shadow:0 2px 12px rgba(0,0,0,0.04);}}
.card-lbl{{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:3px;color:{MUTED};text-transform:uppercase;margin-bottom:8px;}}

.coef-wrap{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px;}}
.coef-item{{background:{SURFACE2};border:1px solid {BORDER};border-radius:10px;padding:14px 16px;text-align:center;}}
.coef-name{{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:2px;color:{MUTED};margin-bottom:6px;}}
.coef-value{{font-family:'Playfair Display',serif;font-size:20px;font-weight:700;color:{ACCENT};}}

.footer{{background:{SURFACE};border-top:1px solid {BORDER};padding:36px 32px 24px;margin-top:60px;}}
.footer-grid{{display:grid;grid-template-columns:2fr 1fr 1fr;gap:32px;margin-bottom:28px;}}
.footer-logo{{font-family:'Playfair Display',serif;font-size:22px;font-weight:700;color:{TEXT};}}
.footer-logo span{{color:{ACCENT};}}
.footer-desc{{font-size:13px;color:{MUTED};font-weight:300;line-height:1.7;margin-top:8px;max-width:260px;}}
.footer-col-title{{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:3px;color:{TEXT};text-transform:uppercase;margin-bottom:12px;}}
.footer-item{{font-size:13px;color:{MUTED};font-weight:300;margin-bottom:8px;padding-left:12px;position:relative;}}
.footer-item::before{{content:'·';position:absolute;left:0;color:{ACCENT};}}
.footer-bottom{{padding-top:20px;border-top:1px solid {BORDER};display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;}}
.footer-credit{{font-size:13px;color:{MUTED};font-weight:300;}}
.footer-credit strong{{color:{TEXT};font-weight:600;}}
.footer-right{{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:1.5px;color:{MUTED};}}
</style>
""", unsafe_allow_html=True)

# ── Navbar ────────────────────────────────────────────────────
st.markdown(f"""
<div class="nav-bar">
  <div>
    <div class="nav-logo">Estate<span>IQ</span></div>
    <div class="nav-tag">House Price Intelligence · ML Regression</div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:1.5px;color:{MUTED};">
    <span class="live-dot"></span> MODEL ACTIVE
  </div>
</div>
""", unsafe_allow_html=True)

# Theme toggle
_, col_t = st.columns([7, 1])
with col_t:
    if st.button(f"{THEME_ICON}  {THEME_LABEL}"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ── Backend: Data & Model ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_dataset(n=500, seed=42):
    rng = np.random.default_rng(seed)
    size     = rng.integers(500, 4000, n).astype(float)
    bedrooms = rng.integers(1, 6, n).astype(float)
    distance = rng.uniform(1, 25, n)
    noise    = rng.normal(0, 8, n)
    price    = np.clip(0.045*size + 6.5*bedrooms - 3.2*distance + 20 + noise, 10, 300)
    return pd.DataFrame({
        "Size_sqft":   np.round(size, 0),
        "Bedrooms":    bedrooms.astype(int),
        "Distance_km": np.round(distance, 1),
        "Price":       np.round(price, 2),
    })

@st.cache_resource(show_spinner=False)
def train_model(df):
    X = df[["Size_sqft","Bedrooms","Distance_km"]]
    y = df["Price"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "r2":   round(r2_score(y_test,y_pred),4),
        "rmse": round(np.sqrt(mean_squared_error(y_test,y_pred)),2),
        "mae":  round(mean_absolute_error(y_test,y_pred),2),
    }
    return model, X_train, X_test, y_train, y_test, y_pred, metrics

# ── Load real CSV or use generated data ───────────────────────
# To use YOUR file: uncomment below and comment out generate_dataset()
# df = pd.read_csv(r"D:\AI-ML RDP\Day-8 Multiple_Linear_Regression\housing_data_500.csv")
df = generate_dataset()
model, X_train, X_test, y_train, y_test, y_pred, metrics = train_model(df)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:4px 0 20px;">
      <div style="font-family:'Playfair Display',serif;font-size:26px;font-weight:700;color:{TEXT};">
        Estate<span style="color:{ACCENT}">IQ</span>
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:3px;color:{MUTED};margin-top:4px;">
        PRICE PREDICTOR
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">// Property Details</div>', unsafe_allow_html=True)
    size_input     = st.number_input("📐 Size (sq ft)",     min_value=300.0,  max_value=5000.0, value=1500.0, step=50.0)
    bedroom_input  = st.number_input("🛏️ Bedrooms",          min_value=1,      max_value=8,      value=3,      step=1)
    distance_input = st.number_input("📍 Distance from City (km)", min_value=0.5, max_value=50.0, value=8.0, step=0.5)
    st.markdown("---")
    predict_btn = st.button("🏠  Predict Price")
    st.markdown("---")
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:1px;color:{MUTED};line-height:2.2;">
    MODEL INFO<br>· Algorithm: Linear Regression<br>· Features: 3<br>· Train split: 80%<br>· R²: {metrics['r2']}<br><br>
    DATASET<br>· Samples: {len(df)}<br>· Train rows: {len(X_train)}<br>· Test rows: {len(X_test)}<br><br>
    BUILT WITH ♥ BY NIRMALYA RAJA
    </div>
    """, unsafe_allow_html=True)

# ── Hero Intro ────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:10px 0 28px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:4px;color:{ACCENT};margin-bottom:14px;">
    // MULTIPLE LINEAR REGRESSION ENGINE
  </div>
  <div style="font-family:'Playfair Display',serif;font-size:clamp(34px,5vw,56px);font-weight:700;color:{TEXT};letter-spacing:-1px;line-height:1.1;margin-bottom:16px;">
    Predict House Prices<br><em style="color:{ACCENT};font-style:italic;">with Machine Learning</em>
  </div>
  <div style="font-size:15px;font-weight:300;color:{MUTED};max-width:520px;margin:0 auto;line-height:1.7;">
    Enter property details in the sidebar. Our regression model trained on {len(df)} samples delivers an instant price estimate.
  </div>
</div>
<div style="text-align:center;margin-bottom:32px;">
  <span class="feature-chip">📐 Size sq ft</span>
  <span class="feature-chip">🛏 Bedrooms</span>
  <span class="feature-chip">📍 Distance km</span>
  <span class="feature-chip">🤖 LinearRegression</span>
  <span class="feature-chip">📊 R² {metrics['r2']}</span>
</div>
""", unsafe_allow_html=True)

# ── Prediction Result ─────────────────────────────────────────
if predict_btn:
    new_house = pd.DataFrame(
        [[size_input, bedroom_input, distance_input]],
        columns=["Size_sqft","Bedrooms","Distance_km"]
    )
    predicted_price = max(5.0, float(model.predict(new_house)[0]))
    conf_low, conf_high = predicted_price * 0.93, predicted_price * 1.07
    price_per_sqft = predicted_price * 100000 / size_input
    zone    = "premium central zone" if distance_input<5 else ("mid-city zone" if distance_input<12 else "suburban zone")
    segment = "luxury segment" if predicted_price>150 else ("mid-premium segment" if predicted_price>80 else "affordable segment")

    st.markdown(f"""
    <div class="hero-card">
      <div class="hero-price-label">// Predicted Market Price</div>
      <div class="hero-price">₹ {predicted_price:.2f}<span class="hero-unit">L</span></div>
      <div class="hero-sub">
        Confidence range: ₹{conf_low:.1f}L — ₹{conf_high:.1f}L &nbsp;·&nbsp;
        {bedroom_input} BHK &nbsp;·&nbsp; {size_input:,.0f} sq ft &nbsp;·&nbsp; {distance_input} km from centre
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("🏠 Price / sq ft", f"₹ {price_per_sqft:,.0f}")
    with c2: st.metric("📐 Size",          f"{size_input:,.0f} sq ft")
    with c3: st.metric("🛏 Bedrooms",      f"{bedroom_input} BHK")
    with c4: st.metric("📍 Distance",      f"{distance_input} km")

    st.markdown(f"""
    <div class="insight-box">
      💡 <strong>AI Insight:</strong> This {bedroom_input}-BHK in the <strong>{zone}</strong>
      is estimated at ₹{predicted_price:.1f}L, placing it in the <strong>{segment}</strong>.
      At ₹{price_per_sqft:,.0f}/sq ft, it
      {"is above average — consider negotiating." if price_per_sqft > 5000 else "offers competitive value for the area."}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

# ── Model Performance ─────────────────────────────────────────
st.markdown(f'<div class="section-label">// Model Performance</div>', unsafe_allow_html=True)
m1,m2,m3 = st.columns(3)
with m1: st.metric("R² Score", f"{metrics['r2']}", delta="Model fit quality")
with m2: st.metric("RMSE",     f"₹ {metrics['rmse']} L", delta="Root mean squared error")
with m3: st.metric("MAE",      f"₹ {metrics['mae']} L",  delta="Mean absolute error")

# ── Charts ────────────────────────────────────────────────────
st.markdown(f'<div class="section-label">// Data Visualisation</div>', unsafe_allow_html=True)
tab1,tab2,tab3,tab4 = st.tabs(["📈 Actual vs Predicted","📊 Feature Impact","🔥 Price Heatmap","📋 Data Explorer"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
        marker=dict(color=y_pred, colorscale="Blues", size=7, opacity=0.7, line=dict(width=0.5,color="white")),
        name="Predictions", hovertemplate="Actual: ₹%{x:.1f}L<br>Predicted: ₹%{y:.1f}L<extra></extra>"))
    mn,mx = float(y_test.min()),float(y_test.max())
    fig.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode="lines",
        line=dict(color=ACCENT,dash="dash",width=2),name="Perfect Prediction"))
    fig.update_layout(template=PLOT_THEME,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Actual vs Predicted Prices",font=dict(family="Playfair Display",size=18)),
        xaxis_title="Actual Price (₹ Lakhs)",yaxis_title="Predicted Price (₹ Lakhs)",height=420)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    colors = [ACCENT2 if c>0 else RED for c in model.coef_]
    fig2 = go.Figure(go.Bar(
        x=["Size (sq ft)","Bedrooms","Distance (km)"], y=model.coef_,
        marker_color=colors, text=[f"{c:+.4f}" for c in model.coef_], textposition="outside"))
    fig2.update_layout(template=PLOT_THEME,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Feature Coefficients — Price Impact",font=dict(family="Playfair Display",size=18)),
        yaxis_title="Coefficient Value",height=380)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(f'<div class="insight-box">📊 <strong>Reading the chart:</strong> Positive bars add value (size, bedrooms), negative bars reduce it (distance). Intercept: ₹{model.intercept_:.2f}L base price.</div>', unsafe_allow_html=True)

with tab3:
    fig3 = px.density_heatmap(df, x="Size_sqft", y="Distance_km", z="Price", histfunc="avg",
        color_continuous_scale="Blues",
        labels={"Size_sqft":"Size (sq ft)","Distance_km":"Distance (km)","Price":"Avg Price (₹L)"},
        title="Average Price — Size × Distance Matrix")
    fig3.update_layout(template=PLOT_THEME,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(family="Playfair Display",size=18)),height=400)
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    cf1,cf2 = st.columns([2,1])
    with cf1:
        price_range = st.slider("Filter by Price Range (₹ Lakhs)", float(df.Price.min()), float(df.Price.max()), (float(df.Price.min()),float(df.Price.max())))
    with cf2:
        bed_filter = st.multiselect("Bedrooms", sorted(df.Bedrooms.unique()), default=sorted(df.Bedrooms.unique()))
    filtered = df[(df.Price>=price_range[0])&(df.Price<=price_range[1])&(df.Bedrooms.isin(bed_filter))]
    st.dataframe(filtered.style.format({"Price":"₹ {:.2f}L","Size_sqft":"{:,.0f}","Distance_km":"{:.1f}"}), height=320, use_container_width=True)
    st.caption(f"Showing {len(filtered)} of {len(df)} records")

# ── Regression Equation ───────────────────────────────────────
st.markdown("---")
st.markdown(f'<div class="section-label">// Regression Equation & Coefficients</div>', unsafe_allow_html=True)
c_size,c_bed,c_dist = model.coef_
intercept = model.intercept_
st.markdown(f"""
<div class="info-card" style="text-align:center;">
  <div class="card-lbl">FITTED MODEL EQUATION</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:14px;color:{ACCENT};margin:12px 0;letter-spacing:0.5px;line-height:2.2;">
    Price &nbsp;=&nbsp; {intercept:.3f}
    &nbsp;+&nbsp; ({c_size:.5f} &times; Size_sqft)
    &nbsp;+&nbsp; ({c_bed:.4f} &times; Bedrooms)
    &nbsp;+&nbsp; ({c_dist:.4f} &times; Distance_km)
  </div>
  <div style="font-size:12px;color:{MUTED};font-weight:300;">All prices in ₹ Lakhs</div>
</div>
<div class="coef-wrap">
  <div class="coef-item"><div class="coef-name">INTERCEPT β₀</div><div class="coef-value">{intercept:.3f}</div></div>
  <div class="coef-item"><div class="coef-name">SIZE β₁</div><div class="coef-value">{c_size:.5f}</div></div>
  <div class="coef-item"><div class="coef-name">BEDROOMS β₂</div><div class="coef-value">{c_bed:.4f}</div></div>
</div>
<div class="coef-wrap" style="grid-template-columns:1fr 1fr;margin-top:10px;">
  <div class="coef-item"><div class="coef-name">DISTANCE β₃</div><div class="coef-value">{c_dist:.4f}</div></div>
  <div class="coef-item"><div class="coef-name">R² SCORE</div><div class="coef-value">{metrics['r2']}</div></div>
</div>
""", unsafe_allow_html=True)

# ── Price Distribution ────────────────────────────────────────
st.markdown("---")
st.markdown(f'<div class="section-label">// Price Distribution</div>', unsafe_allow_html=True)
fig_dist = go.Figure(go.Histogram(x=df["Price"], nbinsx=40, marker_color=ACCENT, opacity=0.75,
    hovertemplate="Price: ₹%{x:.1f}L<br>Count: %{y}<extra></extra>"))
fig_dist.update_layout(template=PLOT_THEME,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
    title=dict(text="Price Distribution Across Dataset",font=dict(family="Playfair Display",size=18)),
    xaxis_title="Price (₹ Lakhs)",yaxis_title="Number of Properties",height=320,bargap=0.05)
st.plotly_chart(fig_dist, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  <div class="footer-grid">
    <div>
      <div class="footer-logo">Estate<span>IQ</span></div>
      <div class="footer-desc">An intelligent house price prediction engine powered by Multiple Linear Regression. Built for learners and data scientists who want to see ML in action.</div>
      <div style="font-family:'Playfair Display',serif;font-style:italic;font-size:13px;color:{ACCENT};margin-top:14px;">"Data is the new foundation."</div>
    </div>
    <div>
      <div class="footer-col-title">Technology</div>
      <div class="footer-item">Python 3.9+</div>
      <div class="footer-item">Streamlit</div>
      <div class="footer-item">scikit-learn</div>
      <div class="footer-item">Plotly · Pandas · NumPy</div>
    </div>
    <div>
      <div class="footer-col-title">Features</div>
      <div class="footer-item">Live Price Prediction</div>
      <div class="footer-item">Feature Coefficient View</div>
      <div class="footer-item">Price Heatmap</div>
      <div class="footer-item">Data Explorer + Filter</div>
      <div class="footer-item">Light / Dark Theme</div>
    </div>
  </div>
  <div class="footer-bottom">
    <div class="footer-credit">
      Designed &amp; developed with <span style="color:#e05252;">♥</span> by <strong>Nirmalya Raja</strong>
      &nbsp;·&nbsp; 
    </div>
    <div class="footer-right">SCIKIT-LEARN · STREAMLIT · v1.0 · 2025</div>
  </div>
</div>
""", unsafe_allow_html=True)