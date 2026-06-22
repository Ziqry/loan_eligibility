"""
app.py — Loan Eligibility Predictor (Streamlit)
------------------------------------------------------------------
Based on: Loan_Default_Prediction_P2.ipynb
Model    : Logistic Regression + SMOTE, decision threshold 0.635
Author   : Mohamad Ziqry Bin Zulkifli

The model predicts the PROBABILITY OF DEFAULT.
A customer is "Eligible" when that probability is below the threshold.

Run:
    pip install streamlit scikit-learn imbalanced-learn pandas numpy joblib
    streamlit run app.py

On first run, if the saved artifacts (loan_model.pkl etc.) are not
found, the app trains them from Loan_default.csv automatically.
------------------------------------------------------------------
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- config
SEED = 42
CSV_PATH = "Loan_default.csv"
DEFAULT_THRESHOLD = 0.635

NOMINAL_COLS = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]
BINARY_COLS = ["HasMortgage", "HasDependents", "HasCoSigner"]

# Categories and the baseline level that get_dummies(drop_first=True) drops
CATEGORY_OPTIONS = {
    "Education": ["Bachelor's", "High School", "Master's", "PhD"],
    "EmploymentType": ["Full-time", "Part-time", "Self-employed", "Unemployed"],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "LoanPurpose": ["Auto", "Business", "Education", "Home", "Other"],
}
BASELINE = {
    "Education": "Bachelor's",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Divorced",
    "LoanPurpose": "Auto",
}

st.set_page_config(page_title="Loan Eligibility Predictor", page_icon="🏦", layout="wide")


# ---------------------------------------------------------------- version compatibility
# Works on both old (<1.18) and new Streamlit without upgrading.
def cache_resource(func):
    """Use cache_resource if present, else fall back to older caching APIs."""
    if hasattr(st, "cache_resource"):
        return st.cache_resource(show_spinner=False)(func)
    if hasattr(st, "experimental_singleton"):
        return st.experimental_singleton(show_spinner=False)(func)
    return st.cache(allow_output_mutation=True, show_spinner=False)(func)


def divider():
    """st.divider() exists only in >=1.18; fall back to a markdown rule."""
    if hasattr(st, "divider"):
        st.divider()
    else:
        st.markdown("---")


# ---------------------------------------------------------------- model loading
@cache_resource
def load_or_train():
    """Load saved artifacts; if missing, train from CSV using the notebook pipeline."""
    have_artifacts = all(
        os.path.exists(p)
        for p in ["loan_model.pkl", "scaler.pkl", "feature_names.json"]
    )

    if have_artifacts:
        model = joblib.load("loan_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_names = json.load(open("feature_names.json"))
        threshold = DEFAULT_THRESHOLD
        if os.path.exists("threshold.json"):
            threshold = json.load(open("threshold.json")).get("threshold", DEFAULT_THRESHOLD)
        return model, scaler, feature_names, threshold, "loaded"

    # ---- train from CSV ----
    if not os.path.exists(CSV_PATH):
        return None, None, None, None, "missing_csv"

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from imblearn.over_sampling import SMOTE

    df = pd.read_csv(CSV_PATH)
    df_clean = df.drop(columns=["LoanID"]).copy()
    for col in BINARY_COLS:
        df_clean[col] = (df_clean[col] == "Yes").astype(int)
    df_clean = pd.get_dummies(df_clean, columns=NOMINAL_COLS, drop_first=True)

    X = df_clean.drop(columns=["Default"])
    y = df_clean["Default"]
    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    X_sm, y_sm = SMOTE(random_state=SEED).fit_resample(X_scaled, y)
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X_sm, y_sm)

    # cache to disk for next time
    joblib.dump(model, "loan_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    json.dump(feature_names, open("feature_names.json", "w"))
    json.dump({"threshold": DEFAULT_THRESHOLD}, open("threshold.json", "w"))

    return model, scaler, feature_names, DEFAULT_THRESHOLD, "trained"


def build_feature_vector(inputs, feature_names):
    """Turn the raw form inputs into a single-row DataFrame of the 24 model columns."""
    row = {f: 0 for f in feature_names}

    numeric = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
               "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"]
    for k in numeric:
        row[k] = inputs[k]

    for k in BINARY_COLS:
        row[k] = 1 if inputs[k] == "Yes" else 0

    for prefix in NOMINAL_COLS:
        val = inputs[prefix]
        col = f"{prefix}_{val}"
        if val != BASELINE[prefix] and col in row:
            row[col] = 1

    return pd.DataFrame([row])[feature_names]


def explain_linear(model, scaler, x_row, feature_names, top_n=6):
    """
    Exact local explanation for Logistic Regression:
    contribution to log-odds(default) = coef_i * scaled_x_i.
    Positive pushes toward default; negative pushes toward approval.
    """
    x_scaled = scaler.transform(x_row)[0]
    coefs = model.coef_[0]
    contrib = coefs * x_scaled
    s = pd.Series(contrib, index=feature_names).sort_values(key=np.abs, ascending=False)
    return s.head(top_n)


# ---------------------------------------------------------------- UI
st.title("🏦 Loan Eligibility Predictor")
st.caption(
    "Explainable ML model (Logistic Regression + SMOTE). "
    "Predicts probability of default; eligibility is decided against a tunable threshold."
)

model, scaler, feature_names, threshold, status = load_or_train()

if status == "missing_csv":
    st.error(
        "No saved model found and `Loan_default.csv` is not in this folder.\n\n"
        "Either run `python train_model.py` first, or place `Loan_default.csv` "
        "next to `app.py` so the app can train on first launch."
    )
    st.stop()

if status == "trained":
    st.info("Model trained from `Loan_default.csv` and cached for next time.")

# Sidebar: threshold control
with st.sidebar:
    st.header("⚙️ Decision settings")
    threshold = st.slider(
        "Approval threshold (P of default)",
        min_value=0.05, max_value=0.95, value=float(threshold), step=0.005,
        help="If predicted default probability is below this value, the applicant is eligible. "
             "The notebook's F1-optimal threshold is 0.635.",
    )
    st.markdown(
        "**How to read it**\n\n"
        "- Higher threshold → more lenient (more approvals)\n"
        "- Lower threshold → stricter (fewer approvals)\n"
    )
    if threshold < 0.3:
        st.warning(
            "Threshold is very low. Because the model was SMOTE-balanced, most "
            "applicants score around 0.3–0.6, so almost everyone will be rejected "
            "at this setting. The notebook's recommended cut-point is 0.635."
        )
    divider()
    st.caption(
        "Note: the model was trained on a LendingClub-style dataset (terms up to "
        "60 months). Inputs far outside that range are extrapolated and should be "
        "interpreted with caution."
    )

# Input form
st.subheader("Applicant details")
with st.form("applicant"):
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 18, 100, 43)
        income = st.number_input("Annual income", 0, 1_000_000, 82_466, step=1000)
        loan_amount = st.number_input("Loan amount", 1000, 1_000_000, 127_556, step=1000)
        dti = st.slider("Debt-to-Income (commitment) ratio", 0.0, 1.0, 0.50, 0.01,
                        help="Monthly debt commitments ÷ income. Higher = more committed.")

    with c2:
        interest_rate = st.slider("Interest rate (%)", 1.0, 30.0, 13.46, 0.01)
        loan_term = st.selectbox(
            "Loan term (months)",
            [12, 24, 36, 48, 60, 72, 84, 96, 108, 120], index=2,
            help="The model was trained on terms up to 60 months; longer terms are extrapolated.",
        )
        months_employed = st.number_input("Months employed", 0, 600, 60)
        num_credit_lines = st.number_input("Number of credit lines", 0, 20, 2)
        education = st.selectbox("Education", CATEGORY_OPTIONS["Education"])

    with c3:
        employment = st.selectbox("Employment type", CATEGORY_OPTIONS["EmploymentType"])
        marital = st.selectbox("Marital status", CATEGORY_OPTIONS["MaritalStatus"])
        purpose = st.selectbox("Loan purpose", CATEGORY_OPTIONS["LoanPurpose"])
        has_mortgage = st.selectbox("Has mortgage", ["No", "Yes"])
        has_dependents = st.selectbox("Has dependents", ["No", "Yes"])
        has_cosigner = st.selectbox("Has co-signer", ["No", "Yes"])

    submitted = st.form_submit_button("Assess eligibility")

# ---------------------------------------------------------------- prediction
if submitted:
    # CreditScore field removed from UI; held at the dataset median (574)
    # because the model still requires it as one of its 24 features.
    credit_score = 574

    inputs = {
        "Age": age, "Income": income, "LoanAmount": loan_amount,
        "CreditScore": credit_score, "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines, "InterestRate": interest_rate,
        "LoanTerm": loan_term, "DTIRatio": dti,
        "Education": education, "EmploymentType": employment,
        "MaritalStatus": marital, "LoanPurpose": purpose,
        "HasMortgage": has_mortgage, "HasDependents": has_dependents,
        "HasCoSigner": has_cosigner,
    }

    x_row = build_feature_vector(inputs, feature_names)
    x_scaled = pd.DataFrame(scaler.transform(x_row), columns=feature_names)
    prob_default = float(model.predict_proba(x_scaled)[0, 1])
    eligible = prob_default < threshold

    divider()
    left, right = st.columns([1, 1])

    with left:
        if eligible:
            st.success("### ✅ Eligible")
            st.write("Predicted default risk is **below** the approval threshold.")
        else:
            st.error("### ❌ Not eligible")
            st.write("Predicted default risk is **at or above** the approval threshold.")

        st.metric("Probability of default", f"{prob_default*100:.1f}%")
        st.metric("Approval threshold", f"{threshold*100:.1f}%")
        st.progress(min(prob_default, 1.0))

    with right:
        st.markdown("**Key factors in this decision**")
        st.caption("Logistic-regression contributions to default risk "
                   "(red = raises risk, green = lowers risk).")
        contrib = explain_linear(model, scaler, x_row, feature_names)
        exp_df = pd.DataFrame({
            "Feature": contrib.index,
            "Effect on default risk": contrib.values.round(3),
        })
        try:
            styled = (exp_df.style
                      .format({"Effect on default risk": "{:+.3f}"})
                      .background_gradient(cmap="RdYlGn_r", subset=["Effect on default risk"]))
            st.dataframe(styled)
        except Exception:
            # Fallback if matplotlib/Styler unsupported on this version
            st.table(exp_df)

    st.caption(
        "This is a statistical risk estimate, not a credit decision. "
        "The threshold is a business/policy choice, and the model reflects "
        "patterns in its training data only."
    )
