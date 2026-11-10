
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =====================================================
# 1. PROJECT PATHS
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = ROOT / "models" / "selected_svm_model.pkl"
FEATURE_FILE = ROOT / "data" / "processed" / "participant_features.csv"

STAT_FILE = (
    ROOT / "reports" / "statistical_analysis"
    / "feature_statistics.csv"
)

MODEL_RESULTS_FILE = (
    ROOT / "reports" / "model_results"
    / "repeated_model_comparison.csv"
)

FIGURE_DIR = ROOT / "reports" / "figures"


# =====================================================
# 2. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ADHD-Insight | Activity Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 3. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.2);
}

.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #172554, #312e81);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
    margin-bottom: 8px;
}

.hero p {
    color: #dbeafe;
    font-size: 15px;
}

.metric-card {
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 13px;
    padding: 17px;
}

.section-note {
    padding: 14px;
    border-radius: 10px;
    background: rgba(59,130,246,0.09);
    border-left: 4px solid #3b82f6;
}

div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 12px;
    padding: 14px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. LOAD PROJECT FILES
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_dataset():
    return pd.read_csv(FEATURE_FILE)


@st.cache_data
def load_statistics():
    if STAT_FILE.exists():
        return pd.read_csv(STAT_FILE)
    return pd.DataFrame()


@st.cache_data
def load_model_results():
    if MODEL_RESULTS_FILE.exists():
        return pd.read_csv(MODEL_RESULTS_FILE)
    return pd.DataFrame()


# =====================================================
# 5. FEATURE EXTRACTION
# =====================================================

FEATURES = [
    "mean_activity",
    "median_activity",
    "std_activity",
    "max_activity",
    "zero_activity_percentage",
    "longest_zero_bout_minutes"
]


def longest_zero_bout(values):

    longest = 0
    current = 0

    for value in values:

        if value == 0:
            current += 1
            longest = max(longest, current)

        else:
            current = 0

    return longest


def extract_features(uploaded_file):

    activity_df = pd.read_csv(uploaded_file)

    # Normalize column names
    activity_df.columns = [
        str(col).strip().upper()
        for col in activity_df.columns
    ]

    # Support alternate column names
    rename_map = {}

    if "TIME" in activity_df.columns:
        rename_map["TIME"] = "TIMESTAMP"

    if "ACC" in activity_df.columns:
        rename_map["ACC"] = "ACTIVITY"

    activity_df = activity_df.rename(columns=rename_map)

    # Validate columns
    if not {"TIMESTAMP", "ACTIVITY"}.issubset(
        activity_df.columns
    ):
        raise ValueError(
            "CSV must contain TIMESTAMP and ACTIVITY columns."
        )

    # Convert values
    activity_df["TIMESTAMP"] = pd.to_datetime(
        activity_df["TIMESTAMP"],
        errors="coerce"
    )

    activity_df["ACTIVITY"] = pd.to_numeric(
        activity_df["ACTIVITY"],
        errors="coerce"
    )

    # Remove invalid rows
    activity_df = activity_df.dropna(
        subset=["TIMESTAMP", "ACTIVITY"]
    )

    activity_df = activity_df.sort_values(
        "TIMESTAMP"
    ).reset_index(drop=True)

    if len(activity_df) < 2:
        raise ValueError(
            "At least two valid activity records are required."
        )

    if (activity_df["ACTIVITY"] < 0).any():
        raise ValueError(
            "Negative activity readings are not allowed."
        )

    if activity_df["TIMESTAMP"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found."
        )

    # Verify continuous one-minute recordings
    gaps = activity_df["TIMESTAMP"].diff().dropna()

    if not (gaps == pd.Timedelta(minutes=1)).all():
        raise ValueError(
            "Recording must contain continuous one-minute intervals, "
            "matching the training data."
        )

    values = activity_df["ACTIVITY"]

    if values.max() > 8000:
        raise ValueError(
            "Activity value exceeds the maximum observed in the "
            "training dataset. Please verify the units and format."
        )

    if values.nunique() == 1:
        raise ValueError(
            "The uploaded recording has constant activity values. "
            "Please upload a valid recording."
        )

    # Extract the six training features
    features = {
        "mean_activity": values.mean(),

        "median_activity": values.median(),

        "std_activity": values.std(),

        "max_activity": values.max(),

        "zero_activity_percentage": (
            values.eq(0).mean() * 100
        ),

        "longest_zero_bout_minutes": longest_zero_bout(
            values.tolist()
        )
    }

    return (
        pd.DataFrame([features]),
        activity_df,
        len(activity_df)
    )


# =====================================================
# 6. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("# 🧠 ADHD-Insight")

st.sidebar.caption(
    "Activity-based ML research prototype"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Activity Prediction",
        "Activity Insights",
        "Model Comparison",
        "Statistical Analysis",
        "About Project"
    ],
    index=0
)

st.sidebar.divider()

st.sidebar.markdown("**PROJECT DETAILS**")
st.sidebar.caption("Dataset: HYPERAKTIV")
st.sidebar.caption("Participants: 85")
st.sidebar.caption("Selected Model: SVM (RBF)")
st.sidebar.caption("Features: 6")

st.sidebar.divider()

st.sidebar.caption(
    "Experimental research classification. Not a medical diagnosis."
)


# =====================================================
# 7. ACTIVITY PREDICTION - MAIN PAGE
# =====================================================

if page == "Activity Prediction":

    st.markdown("""
    <div class="hero">
        <h1>🧠 ADHD-Insight</h1>
        <p>
        Understand activity patterns through machine learning.
        Upload an activity recording to generate a research
        classification result.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Upload Activity Recording")


    st.write(
        "Upload one participant's activity CSV file. The system will "
        "automatically extract six activity features and analyze the "
        "recording using the selected SVM model."
    )

    uploaded_file = st.file_uploader(
        "Choose an activity CSV file",
        type=["csv"],
        help="Required columns: TIMESTAMP and ACTIVITY"
    )

    st.caption(
        "Expected format: one activity reading per minute, "
        "with continuous timestamps."
    )

    if uploaded_file is not None:

        st.success(
            f"File uploaded: {uploaded_file.name}"
        )

        if st.button(
            "Analyze Activity",
            type="primary",
            use_container_width=True
        ):

            try:

                artifact = load_model()

                model = artifact["model"]
                model_features = artifact["features"]

                features, activity_data, record_count = (
                    extract_features(uploaded_file)
                )

                # Use exactly the training feature order
                features = features[model_features]

                prediction = model.predict(features)[0]

                st.divider()

                st.subheader("Activity Recording Overview")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Total Readings",
                    f"{record_count:,}"
                )

                duration = (
                    activity_data["TIMESTAMP"].max()
                    - activity_data["TIMESTAMP"].min()
                )

                c2.metric(
                    "Recording Duration",
                    f"{duration.total_seconds() / 3600:.1f} hours"
                )

                c3.metric(
                    "Mean Activity",
                    f"{features.iloc[0]['mean_activity']:.2f}"
                )

                # Activity time-series chart
                st.subheader("Activity Timeline")

                chart_data = activity_data.set_index(
                    "TIMESTAMP"
                )[["ACTIVITY"]]

                st.line_chart(
                    chart_data,
                    height=250
                )

                st.divider()

                st.subheader("Extracted Activity Features")

                display_features = features.T.reset_index()

                display_features.columns = [
                    "Feature",
                    "Value"
                ]

                display_features["Value"] = (
                    display_features["Value"].round(3)
                )

                st.dataframe(
                    display_features,
                    use_container_width=True,
                    hide_index=True
                )

                st.divider()

                st.subheader("Machine Learning Prediction")

                st.markdown(
                    "**Selected model:** Support Vector Machine "
                    "(RBF Kernel)"
                )

                if prediction == 1:

                    st.error(
                        "Model Prediction: ADHD-associated pattern"
                    )

                    st.write(
                        "The model predicts that this activity pattern "
                        "is more consistent with the ADHD-labelled group "
                        "in the training dataset."
                    )

                else:

                    st.success(
                        "Model Prediction: Control-associated pattern"
                    )

                    st.write(
                        "The model predicts that this activity pattern "
                        "is more consistent with the control-labelled "
                        "group in the training dataset."
                    )

                st.warning(
                    "IMPORTANT: This is an experimental group-classification "
                    "result, NOT a medical diagnosis. The current model has "
                    "limited predictive performance and cannot determine "
                    "whether a person has ADHD."
                )

                st.caption(
                    "Label interpretation assumes label 1 = ADHD group and "
                    "label 0 = control group, based on the dataset group "
                    "counts. Confirm this mapping against the original "
                    "dataset codebook before treating it as definitive."
                )

                st.caption(
                    "Synthetic sample recordings are for software "
                    "demonstration only and do not have ground-truth labels."
                )

            except Exception as error:

                st.error(
                    f"Unable to analyze the uploaded file: {error}"
                )


# =====================================================
# 8. ACTIVITY INSIGHTS
# =====================================================

elif page == "Activity Insights":

    st.title("Activity Insights")

    st.write(
        "Explore participant-level activity patterns "
        "in the HYPERAKTIV dataset."
    )

    df = load_dataset()

    c1, c2, c3 = st.columns(3)

    c1.metric("Participants", len(df))

    c2.metric(
        "Label 0",
        int((df["ADHD"] == 0).sum())
    )

    c3.metric(
        "Label 1",
        int((df["ADHD"] == 1).sum())
    )

    st.divider()

    st.subheader("Dataset Group Distribution")

    st.bar_chart(
        df["ADHD"].value_counts().sort_index()
    )

    st.subheader("Activity Feature Visualizations")

    figure_names = [
        "mean_activity_comparison.png",
        "median_activity_comparison.png",
        "std_activity_comparison.png",
        "max_activity_comparison.png",
        "activity_90th_percentile_comparison.png",
        "activity_95th_percentile_comparison.png",
        "zero_activity_percentage_comparison.png",
        "nonzero_activity_mean_comparison.png",
        "longest_zero_bout_minutes_comparison.png",
        "feature_correlation.png"
    ]

    available = [
        name for name in figure_names
        if (FIGURE_DIR / name).exists()
    ]

    for i in range(0, len(available), 2):

        cols = st.columns(2)

        for j, col in enumerate(cols):

            index = i + j

            if index < len(available):

                filename = available[index]

                with col:

                    st.image(
                        str(FIGURE_DIR / filename),
                        caption=filename.replace(
                            "_", " "
                        ).replace(".png", "").title(),
                        use_container_width=True
                    )

    with st.expander("View participant feature dataset"):

        st.dataframe(
            df,
            use_container_width=True
        )


# =====================================================
# 9. MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    st.title("Model Performance")

    st.write(
        "Comparison of six machine learning approaches "
        "using repeated stratified cross-validation."
    )

    results = load_model_results()

    if results.empty:

        st.warning(
            "Model comparison results were not found."
        )

    else:

        st.caption(
            "5 folds × 5 repeats = 25 evaluations per model."
        )

        st.subheader("Model Comparison Table")

        display_cols = [
            "model",
            "accuracy_mean",
            "balanced_accuracy_mean",
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "roc_auc_mean"
        ]

        st.dataframe(
            results[display_cols].round(3),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Accuracy Comparison")

        st.bar_chart(
            results.set_index("model")["accuracy_mean"]
        )

        st.subheader("ROC-AUC Comparison")

        st.bar_chart(
            results.set_index("model")["roc_auc_mean"]
        )

        st.info(
            "SVM RBF was selected for the prototype because it "
            "achieved the highest mean accuracy and balanced accuracy "
            "among the tested machine learning models. Its performance "
            "is close to chance and requires further investigation."
        )

        st.caption(
            "Repeated cross-validation results are exploratory and "
            "do not constitute independent external validation."
        )


# =====================================================
# 10. STATISTICAL ANALYSIS
# =====================================================

elif page == "Statistical Analysis":

    st.title("Statistical Analysis")

    stats = load_statistics()

    if stats.empty:

        st.warning(
            "Statistical analysis results were not found."
        )

    else:

        st.write(
            "Mann–Whitney U tests compare feature distributions "
            "between the two encoded dataset groups. "
            "P-values are adjusted using the "
            "Benjamini–Hochberg procedure."
        )

        significant = int(
            stats["significant_0.05"].fillna(False).sum()
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Features Tested",
            len(stats)
        )

        c2.metric(
            "Significant Features",
            significant
        )

        st.dataframe(
            stats.round(4),
            use_container_width=True,
            hide_index=True
        )

        if significant == 0:

            st.info(
                "No individual feature showed a statistically "
                "significant group difference after multiple-testing "
                "correction in this exploratory analysis."
            )


# =====================================================
# 11. ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.title("About ADHD-Insight")

    st.write(
        "ADHD-Insight is an activity-based machine learning "
        "research prototype built using the HYPERAKTIV dataset."
    )

    st.subheader("Research Objective")

    st.write(
        "To investigate whether statistical characteristics "
        "of participant activity recordings contain patterns "
        "that help distinguish the two encoded dataset groups."
    )

    st.subheader("Dataset")

    st.markdown("""
    - Dataset: HYPERAKTIV
    - Original metadata participants: 103
    - Participants with available activity recordings: 85
    - Activity sampling interval: 1 minute
    - Current classification labels: 0 and 1
    """)

    st.subheader("Machine Learning Workflow")

    st.markdown("""
    1. Activity data loading
    2. Timestamp and data validation
    3. Participant-level feature extraction
    4. Exploratory data analysis
    5. Statistical hypothesis testing
    6. Model comparison
    7. SVM-based activity classification
    """)

    st.subheader("Selected Features")

    for feature in FEATURES:
        st.markdown(f"- `{feature}`")

    st.subheader("Limitations")

    st.markdown("""
    - Small participant sample.
    - Some participants from the original metadata have no activity recordings.
    - Activity features show substantial overlap between groups.
    - Current model performance is close to chance.
    - No independent external validation has been performed.
    - Dataset labels are not a substitute for a clinical diagnosis.
    """)

    st.warning(
        "ADHD-Insight is an academic research prototype. "
        "It must not be used to diagnose ADHD or make medical decisions."
    )


# =====================================================
# 12. FOOTER
# =====================================================

st.divider()

st.caption(
    "ADHD-Insight | Academic Research Prototype | "
    "HYPERAKTIV Dataset | Not for Clinical Use"
)
