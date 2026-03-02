<p align="center">
  <svg width="900" height="220" viewBox="0 0 900 220" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:#2563eb;stop-opacity:1" />
        <stop offset="100%" style="stop-color:#059669;stop-opacity:1" />
      </linearGradient>
    </defs>

    <rect width="900" height="220" rx="20" fill="#0f172a"/>
    <rect x="10" y="10" width="880" height="200" rx="16" fill="url(#grad1)" opacity="0.08"/>

    <text x="50%" y="85" text-anchor="middle"
      font-family="Georgia, serif"
      font-size="48"
      fill="#ffffff"
      font-weight="bold">
      🏠 EstateIQ
    </text>

    <text x="50%" y="125" text-anchor="middle"
      font-family="Courier New, monospace"
      font-size="16"
      fill="#cbd5e1"
      letter-spacing="2">
      HOUSE PRICE PREDICTION ENGINE
    </text>

    <text x="50%" y="155" text-anchor="middle"
      font-family="Courier New, monospace"
      font-size="14"
      fill="#94a3b8">
      Multiple Linear Regression · Streamlit · scikit-learn · 2025
    </text>
  </svg>
</p>

---

# 🏠 EstateIQ — House Price Prediction Engine  
**Multiple Linear Regression · Streamlit · Professional ML Dashboard**

EstateIQ is a professionally designed house price prediction system powered by **Multiple Linear Regression**. It combines a clean synthetic (or real CSV-based) dataset, a properly trained scikit-learn model, advanced evaluation metrics, and an elegant dual-theme Streamlit interface — delivering instant market value estimates with interactive analytics.

---

## ✨ Highlights

| Feature | Description |
|----------|------------|
| 🧮 Multiple Linear Regression | Predicts house prices using 3 independent variables |
| 📊 Model Evaluation | R² Score, RMSE, MAE performance metrics |
| 🏠 Live Price Prediction | Instant estimate based on user inputs |
| 🔍 Feature Impact View | Coefficient visualization showing price contribution |
| 🔥 Price Heatmap | Size × Distance interaction analysis |
| 📈 Actual vs Predicted Chart | Model accuracy visual validation |
| 📋 Data Explorer | Filterable dataset viewer with dynamic controls |
| 🌗 Light / Dark Theme | Premium dual-mode UI |
| 🎨 Professional UI | Custom CSS styling, hero cards & branded layout |

---

## 🗂️ Project Structure


estateiq/
│
├── 📄 house_price_app.py ← Streamlit Web Application (UI + Model)
├── 📄 house_price_model.py ← Core Python ML script (CLI version)
├── 📄 housing_data_500.csv ← Dataset (optional if using real data)
├── 📄 requirements.txt ← All dependencies
└── 📄 README.md


---

## 🚀 Quick Start

### 1 · Clone the repository

```bash
git clone https://github.com/<your-username>/estateiq.git
cd estateiq
2 · Install dependencies
pip install -r requirements.txt

Python 3.9+ recommended.

Using a virtual environment:

python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
pip install -r requirements.txt
3 · Run the Streamlit Web App
streamlit run house_price_app.py

The app opens automatically at:

http://localhost:8501
4 · Run the CLI Version (Terminal Model)
python house_price_model.py

You will be prompted to enter:

Property Size (sq ft)

Number of Bedrooms

Distance from City Centre

The model will output the predicted price in ₹ Lakhs.

🧠 How It Works
Housing Dataset (CSV / Generated)
        │
        ▼
Feature Selection:
- Size_sqft
- Bedrooms
- Distance_km
        │
        ▼
Train-Test Split (80% / 20%)
        │
        ▼
Multiple Linear Regression (scikit-learn)
        │
        ▼
Metrics Evaluation (R², RMSE, MAE)
        │
        ▼
Live User Prediction
📊 Regression Equation
Price =
β₀
+ β₁ × Size_sqft
+ β₂ × Bedrooms
+ β₃ × Distance_km

Where:

β₀ → Intercept (base price)

β₁ → Impact of property size

β₂ → Impact of bedrooms

β₃ → Impact of distance from city centre

Positive coefficients increase price.
Negative coefficients reduce price.

🖥️ Interface Overview
🌤 Consumer Mode

Hero price display

Confidence range

Price per sq ft

AI-style insight summary

Market segment classification

🔬 Analytical Mode

📈 Actual vs Predicted Scatter Plot

📊 Feature Coefficient Bar Chart

🔥 Size × Distance Heatmap

📋 Filterable Dataset Explorer

📉 Price Distribution Histogram

📐 Full Regression Equation Display

📦 Dependencies
streamlit>=1.30.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
plotly>=5.18.0
matplotlib>=3.7.0

Install everything:

pip install -r requirements.txt
⚡ CLI Reference
streamlit run house_price_app.py
streamlit run house_price_app.py --server.port 8502
streamlit run house_price_app.py --server.address 0.0.0.0
Ctrl + C
🔧 Troubleshooting

❌ streamlit: command not found
→ Install Streamlit: pip install streamlit

❌ ModuleNotFoundError: No module named 'sklearn'
→ Install dependencies: pip install -r requirements.txt

❌ Port 8501 already in use
→ Use another port: --server.port 8502

❌ CSV file not found
→ Verify file path or switch to generated dataset

🗺️ Roadmap

Add Feature Scaling (StandardScaler)

Implement Ridge & Lasso Regression comparison

Add Confidence Interval using residual distribution

Add SHAP feature importance

Integrate real estate API

Deploy to Streamlit Cloud

Add Exportable PDF valuation report

📜 License

Released under the MIT License.

🙏 Acknowledgements

scikit-learn — Linear Regression

Streamlit — Web App Framework

Plotly — Interactive Visualisation

Pandas — Data Handling

NumPy — Numerical Computation

SCIKIT-LEARN · STREAMLIT · v1.0 · 2025

╔══════════════════════════════════════════════════════════╗
║ ║
║ "Data is the new foundation of modern decisions." ║
║ ║
╚══════════════════════════════════════════════════════════╝

Designed & developed with ♥ by Nirmalya Raja


---

If you want, I can now:

- 🔥 Make a more premium animated SVG version  
- 📸 Add screenshot placeholders section  
- 🌌 Make EstateIQ branding match AtmoSense even closer  
- 🧠 Upgrade wording to recruiter-optimized tone  

Just tell me 😌
