
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Classification")
st.write(
    "Machine Learning Assignment 2 - "
    "Evaluation of five classification models"
)

# -------------------------------------------------
# Model Files
# -------------------------------------------------

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine":
        "model/support_vector_machine.joblib"
}

LABEL_ENCODER_FILE = "model/label_encoder.joblib"
FEATURE_NAMES_FILE = "model/feature_names.joblib"


# -------------------------------------------------
# Load Artifacts
# -------------------------------------------------

@st.cache_resource
def load_artifacts():
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
    feature_names = joblib.load(FEATURE_NAMES_FILE)

    return label_encoder, feature_names


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


try:
    label_encoder, feature_names = load_artifacts()

except Exception as error:
    st.error("Unable to load preprocessing artifacts.")
    st.exception(error)
    st.stop()


# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.header("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys())
)

st.sidebar.write(
    "The application evaluates previously trained models."
)


# -------------------------------------------------
# File Upload
# -------------------------------------------------

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        "Please upload test_data.csv to begin evaluation."
    )

    st.stop()


# -------------------------------------------------
# Read Dataset
# -------------------------------------------------

try:

    test_data = pd.read_csv(uploaded_file)

except Exception as error:

    st.error("Unable to read the uploaded CSV.")
    st.exception(error)
    st.stop()


# Remove accidental index columns

unwanted_columns = [
    column for column in test_data.columns
    if str(column).lower() == "index"
    or str(column).lower().startswith("unnamed")
]

if unwanted_columns:

    test_data = test_data.drop(
        columns=unwanted_columns
    )


# -------------------------------------------------
# Dataset Preview
# -------------------------------------------------

st.subheader("Test Dataset Preview")

c1, c2 = st.columns(2)

c1.metric(
    "Test Instances",
    test_data.shape[0]
)

c2.metric(
    "Total Columns",
    test_data.shape[1]
)

st.dataframe(
    test_data.head(10),
    use_container_width=True
)


# -------------------------------------------------
# Validate Target
# -------------------------------------------------

TARGET_COLUMN = "Class"

if TARGET_COLUMN not in test_data.columns:

    st.error(
        "Uploaded CSV must contain the 'Class' target column."
    )
    st.stop()


# -------------------------------------------------
# Validate Features
# -------------------------------------------------

missing_features = [
    feature
    for feature in feature_names
    if feature not in test_data.columns
]

if missing_features:

    st.error(
        "Required input features are missing:"
    )

    for feature in missing_features:
        st.write(feature)

    st.stop()


# Use features in exactly the same order as training

X_test = test_data[feature_names].copy()

y_original = (
    test_data[TARGET_COLUMN]
    .astype(str)
    .str.strip()
)


# -------------------------------------------------
# Validate Classes
# -------------------------------------------------

unknown_classes = sorted(
    set(y_original)
    - set(label_encoder.classes_)
)

if unknown_classes:

    st.error(
        "Unknown target classes found: "
        + ", ".join(unknown_classes)
    )

    st.stop()


y_test = label_encoder.transform(
    y_original
)


# -------------------------------------------------
# Load Selected Saved Model
# -------------------------------------------------

model_path = MODEL_FILES[
    selected_model_name
]

try:

    model = load_model(
        model_path
    )

except Exception as error:

    st.error(
        "Unable to load the selected model."
    )

    st.exception(error)
    st.stop()


# -------------------------------------------------
# Prediction
# -------------------------------------------------

try:

    y_pred = model.predict(
        X_test
    )

    y_probability = model.predict_proba(
        X_test
    )

except Exception as error:

    st.error(
        "Prediction failed."
    )

    st.exception(error)
    st.stop()


# -------------------------------------------------
# Evaluation Metrics
# -------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_probability,
    multi_class="ovr",
    average="macro"
)

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)


# -------------------------------------------------
# Display Metrics
# -------------------------------------------------

st.subheader(
    f"Evaluation Results - {selected_model_name}"
)

m1, m2, m3 = st.columns(3)

m1.metric(
    "Accuracy",
    f"{accuracy:.4f}"
)

m2.metric(
    "AUC Score",
    f"{auc:.4f}"
)

m3.metric(
    "Precision",
    f"{precision:.4f}"
)

m4, m5, m6 = st.columns(3)

m4.metric(
    "Recall",
    f"{recall:.4f}"
)

m5.metric(
    "F1 Score",
    f"{f1:.4f}"
)

m6.metric(
    "MCC Score",
    f"{mcc:.4f}"
)


# -------------------------------------------------
# Tabs
# -------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "Confusion Matrix",
        "Classification Report",
        "Predictions"
    ]
)


# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------

with tab1:

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=list(
            range(
                len(label_encoder.classes_)
            )
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=ax
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
    )

    ax.set_title(
        "Confusion Matrix - "
        + selected_model_name
    )

    plt.xticks(
        rotation=45
    )

    plt.yticks(
        rotation=0
    )

    plt.tight_layout()

    st.pyplot(fig)


# -------------------------------------------------
# Classification Report
# -------------------------------------------------

with tab2:

    report = classification_report(
        y_test,
        y_pred,
        labels=list(
            range(
                len(label_encoder.classes_)
            )
        ),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )


# -------------------------------------------------
# Predictions
# -------------------------------------------------

with tab3:

    prediction_results = test_data.copy()

    prediction_results[
        "Predicted_Class"
    ] = label_encoder.inverse_transform(
        y_pred
    )

    prediction_results[
        "Prediction_Status"
    ] = np.where(
        prediction_results[TARGET_COLUMN]
        == prediction_results["Predicted_Class"],
        "Correct",
        "Incorrect"
    )

    st.dataframe(
        prediction_results,
        use_container_width=True
    )


st.divider()

st.caption(
    "BITS WILP - Machine Learning Assignment 2"
)
