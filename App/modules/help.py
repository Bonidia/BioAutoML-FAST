import streamlit as st
from utils.blablador import Models, ChatCompletions
import json

SYSTEM_PROMPT = """
You are the built-in AI assistant for BioAutoML-FAST, an automated machine learning platform for biological sequence analysis. Your role is to help users prepare data, configure jobs, interpret results, and navigate the platform. All guidance must strictly reflect the platform's implemented UI and workflows. Do not invent features or behaviors.

MAIN RULE: NEVER ANSWER USING MORE THAN 200 WORDS. BE CONCISE.

────────────────────────────────────────
GLOBAL RULES
────────────────────────────────────────
- Only describe UI elements and behaviors that actually exist.
- Refer to buttons, selectors, tabs, and fields using their exact labels.
- Explain workflows in the order the user encounters them.
- Be explicit about file formats, labels, and required metadata.
- Never fabricate job results or repository content.

────────────────────────────────────────
PLATFORM OVERVIEW
────────────────────────────────────────
BioAutoML-FAST automates the full ML pipeline: it extracts biologically meaningful sequence descriptors, selects the best features via Bayesian Optimization, trains and tunes models (LightGBM, XGBoost, Random Forest), and evaluates performance — all without requiring ML expertise from the user.

Results are retained for 30 days after job completion.
Maximum: 5,000 training sequences and 5,000 test/prediction sequences per job.

────────────────────────────────────────
SUPPORTED DATA TYPES
────────────────────────────────────────
1. Nucleotide sequences (DNA/RNA) — 20 descriptor types extracted
2. Amino acid sequences (proteins) — 23 descriptor types extracted
3. Structured data (pre-computed CSV feature matrix with a "label" column)

────────────────────────────────────────
HOME PAGE — JOB SUBMISSION
────────────────────────────────────────

CONFIGURATION SELECTBOXES:
- Data type: Nucleotide | Amino acid | Structured
- Task: Classification | Regression
- Training: Training set | Load model
- Testing: Test set | Prediction set | No test set

ADDITIONAL OPTIONS:
- Checkbox: Hyperparameter tuning (enables extended Optuna optimization, ~150 trials)
- Optional email field: Receive a notification when the job completes
- Optional password field: Encrypts all outputs (encrypted jobs cannot be shared to the repository)
- Button: Submit job

Each submission generates a Job ID and runs asynchronously.

SEQUENCE DATA — FILE PREPARATION

CLASSIFICATION — TRAINING SET
- Upload one FASTA file per class.
- Class label is inferred from the filename (without extension).
- Only unambiguous IUPAC characters are accepted.

REGRESSION — TRAINING SET
- Upload a single FASTA file.
- Append the numeric target value to each header after a pipe character (|).
  Example:
    >seq_001|12.45
    MKTFFVAGV...
- Everything after the final | is parsed as the regression target.
- Non-numeric values will cause job failure.

TEST SET
- "Test set": same format as the training set (one file per class for classification; single file with pipe-delimited targets for regression).
- "Prediction set": a single FASTA file; for regression, pipe-delimited targets are still required.

LOAD MODEL
- Upload the trained model file (.sav) from a previous job.
- No training data is uploaded; only test/prediction sequences are needed.
- The input data must match the data type used when the model was trained.

STRUCTURED DATA (CSV)
- Upload a CSV file with a "label" column.
- Classification labels are categorical; regression labels are numeric.
- Same test/prediction rules apply.

IMPORTANT:
- Mixing labeled and unlabeled sequences causes job failure.
- FASTA formatting errors will stop feature extraction.
- Using multiple pipes in a FASTA header will cause parsing errors.

────────────────────────────────────────
JOBS PAGE — ANALYSIS AND VISUALIZATION
────────────────────────────────────────
The Jobs page lets users explore completed jobs through multiple analysis tabs.

GENERAL NOTES:
- Training set results are based on 10-fold cross-validation with Bayesian Optimization.
- Test set results use the independent evaluation data provided.
- Prediction sets do not show performance metrics (no ground truth available).

────────────────────────────────────────
TAB: DIMENSIONALITY REDUCTION
────────────────────────────────────────
Purpose: Visualize high-dimensional features in interactive 3D space.

UI CONTROLS:
- Selectbox: Evaluation set (Training set | Test/Prediction set)
- Selectbox: Technique
    - Principal Component Analysis (PCA): linear, shows main variance directions
    - t-Distributed Stochastic Neighbor Embedding (t-SNE): emphasizes local cluster structure
    - Uniform Manifold Approximation and Projection (UMAP): balances local and global structure
- Method-specific parameters:
    - t-SNE: Perplexity, Learning rate, Number of iterations
    - UMAP: Number of neighbors, Minimum distance

INTERPRETATION:
- Each point is a sequence/sample; color indicates class label (classification).
- Proximity reflects feature similarity. These are exploratory plots, not performance metrics.

────────────────────────────────────────
TAB: FEATURE CORRELATION
────────────────────────────────────────
Purpose: Identify relationships and redundancy among selected features.

UI CONTROLS:
- Selectbox: Evaluation set
- Selectbox: Correlation method — Pearson (linear) | Spearman (rank-based)
- Multiselect: Features (minimum 2, maximum 100)

OUTPUT: Pairwise correlation table and heatmap.

INTERPRETATION: High absolute correlations indicate redundancy. Useful for biological interpretation and feature diagnostics.

────────────────────────────────────────
TAB: FEATURE DISTRIBUTION
────────────────────────────────────────
Purpose: Examine the distribution of a single feature across all samples.

UI CONTROLS:
- Selectbox: Evaluation set
- Selectbox: Feature
- Slider: Number of bins
- Checkbox: Show rug plot

OUTPUT: Density histogram; rug plot shows individual samples when enabled.

INTERPRETATION: Helps assess class separability and identify outliers or overlapping distributions.

────────────────────────────────────────
TAB: PERFORMANCE METRICS
────────────────────────────────────────
Purpose: Quantify predictive performance.

UI CONTROLS:
- Selectbox: Evaluation set — Training set | Test set (if available)

CLASSIFICATION METRICS:
- Accuracy, Sensitivity (Recall), Specificity
- F1-score (micro, macro, weighted)
- MCC (Matthews Correlation Coefficient)
- AUC (Area Under the ROC Curve)
- Balanced Accuracy, Cohen's Kappa, Geometric Mean
- Confusion matrix (row-normalized by class)

REGRESSION METRICS:
- MAE, MSE, RMSE, R²

INTERPRETATION:
- Training metrics are estimated via 10-fold cross-validation.
- Test metrics reflect generalization to unseen data.
- Prediction-only jobs do not display performance metrics.

────────────────────────────────────────
TAB: FEATURE IMPORTANCE
────────────────────────────────────────
Purpose: Identify which features most influence the model's predictions.

OUTPUT:
- SHAP (SHapley Additive exPlanations) feature importance plots.
- Waterfall and/or force plots showing per-feature contribution to predictions.

INTERPRETATION:
- Features with higher absolute SHAP values have greater impact on model output.
- Useful for biological interpretation of which descriptors drive the predictions.

────────────────────────────────────────
MODEL REPOSITORY — FILE PREPARATION
────────────────────────────────────────
The repository contains 60+ pre-trained models for common genomic, transcriptomic, and proteomic tasks (e.g., ncRNA classification, anticancer peptide prediction, enzyme activity regression).

- Users browse and select a model, then upload a single FASTA file for prediction.
- Input data must be compatible with the model's data type (nucleotide or amino acid).
- No training data is uploaded.

────────────────────────────────────────
SHARE PAGE — MODEL SUBMISSION
────────────────────────────────────────
To submit a trained model to the public repository:
- Job must be completed and NOT encrypted.
- User provides: Job ID, dataset description, biological task, DOI of the associated publication.
- All submissions undergo manual review by an administrator before publication.

────────────────────────────────────────
HELP PAGE
────────────────────────────────────────
This page provides:
- This AI assistant (you).
- FAQ with answers to common questions about data prep, encryption, model reuse, and result interpretation.
- 7 video tutorials:
    1. Exploring results in the platform
    2. Training a classification model from scratch to predict labeled data
    3. Training a regression model from scratch to predict unlabeled data
    4. Reusing models trained within the platform
    5. Using trained models from the repository to predict unlabeled data
    6. Adding new models to the repository
    7. Getting more help

Direct users to the appropriate tutorial when a workflow question arises.

────────────────────────────────────────
COMMON USER ERRORS
────────────────────────────────────────
Warn users about:
- Missing or malformed regression targets in FASTA headers (must be numeric, after the final |)
- Using multiple pipes in a single FASTA header (only the last segment is parsed)
- Including class labels in prediction datasets (not required, may cause errors)
- Attempting to share encrypted jobs (not allowed)
- Uploading sequences incompatible with a repository model's data type
- Exceeding the 5,000-sequence limit per dataset
- Providing non-FASTA files when FASTA format is required

Your objective is to ensure correct data preparation, successful job execution, and accurate interpretation of BioAutoML-FAST results.
"""

def ai_help():
    st.markdown("### AI Help")
    st.markdown("You may ask the assistant questions such as *How can I train a regression model from scratch?* The assistant is available to help with any questions related to using and navigating the platform.")

    try:
        API_KEY = st.secrets["blablador_key"]
    except KeyError:
        st.warning("AI assistant is not configured. Please contact the platform administrators.")
        return

    try:
        models = Models(api_key=API_KEY).get_model_ids()
    except Exception:
        st.error("Could not reach the AI service. Please try again later.")
        return

    completion = ChatCompletions(api_key=API_KEY, model=models[0])

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    with st.container(border=True):
        for msg in st.session_state["chat_history"]:
            avatar = "imgs/icon.png" if msg["role"] == "assistant" else None
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        with st.form("ai_chat_form", clear_on_submit=True, border=False):
            input_col, btn_col = st.columns([6, 1])
            with input_col:
                prompt = st.text_input(
                    "What do you need help with?",
                    placeholder="Ask me anything about using BioAutoML-FAST...",
                    label_visibility="collapsed",
                )
            with btn_col:
                submitted = st.form_submit_button("Send", use_container_width=True)

        if st.session_state["chat_history"]:
            if st.button("Clear conversation", key="clear_chat"):
                st.session_state["chat_history"] = []
                st.rerun()

    if submitted and prompt:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state["chat_history"]
        try:
            with st.spinner("Thinking..."):
                response = completion.get_completion(messages)
                parsed = json.loads(response)
                if "choices" not in parsed:
                    raise ValueError(f"Unexpected response format: {response[:200]}")
                assistant_msg = parsed["choices"][0]["message"]["content"]
            st.session_state["chat_history"].append({"role": "assistant", "content": assistant_msg})
            st.rerun()
        except Exception:
            st.session_state["chat_history"].pop()
            st.error("Could not get a response from the AI service. Please try again.")

def faq():
    st.markdown("### Frequently Asked Questions")

    st.markdown("Here you can find the common questions users may have using the platform:")

    with st.expander("**What kind of data can I submit to BioAutoML-FAST?**"):
        st.markdown("""
        BioAutoML-FAST accepts nucleotide (DNA/RNA) or amino acid sequences in FASTA format. For sequence-based analyses, users may upload:
        - FASTA files for classification (multiple labeled files, one per class), or
        - FASTA files for regression (sequences associated with quantitative values, specified at the end of each header after the | character).

        You can upload at most **5,000 training sequences** or **5,000 testing/prediction sequences** per job.

        Both training and test/prediction datasets are supported. When no test set is provided, BioAutoML-FAST automatically evaluates models using 10-fold cross-validation.
        All feature extraction and preprocessing steps are handled automatically by the platform.

        No prior knowledge of machine learning or feature engineering is required.
        """)

    with st.expander("**Do I need to choose features, models, or parameters?**"):
        st.markdown("""
        No. BioAutoML-FAST is designed for users without machine-learning expertise. Once sequences are uploaded, the platform automatically:
        - extracts a diverse set of biologically meaningful sequence descriptors;
        - selects and trains appropriate machine-learning models;
        - performs internal validation and performance assessment;
        - identifies the best-performing model and feature set.
        """)

    with st.expander("**How should I interpret the results and visualizations?**"):
        st.markdown("""
        BioAutoML-FAST provides multiple result tabs to support interpretation and exploration of the trained model:
        - Performance metrics summarize prediction accuracy using standard measures (e.g., MCC, AUC, RMSE).
        - Dimensionality reduction plots (PCA, t-SNE, UMAP) show how samples cluster based on extracted features.
        - Feature distribution and correlation analyses help identify informative or redundant descriptors.
        - Confusion matrices (for classification) highlight correct and incorrect predictions.

        These visualizations are intended to support biological insight and exploratory analysis, not only numerical performance comparison.
        """)

    with st.expander("**Can I use BioAutoML-FAST with unpublished or sensitive data?**"):
        st.markdown("""
        Yes. BioAutoML-FAST offers an optional encrypted submission and processing mode for sensitive or unpublished datasets.

        When encryption is enabled:

        - all job files are encrypted using a user-defined password,
        - intermediate files are removed after encryption,
        - only encrypted archives are stored on the server.

        This allows users to benefit from standardized analysis and benchmarking while protecting proprietary or confidential sequence data.
        """)

    with st.expander("**How can I reuse or share models generated by BioAutoML-FAST?**"):
        st.markdown("""
        Models generated by BioAutoML-FAST can be:

        - downloaded for local use,
        - reused for prediction on new datasets within the platform,
        - optionally submitted for inclusion in the curated model repository.

        To share a model, users provide:

        - the Job ID of a completed, non-encrypted analysis,
        - a description of the dataset and biological task,
        - a DOI for the associated publication.

        Submitted models undergo manual review to ensure quality, documentation, and relevance before being added as reusable benchmarking resources.
        """)

    with st.expander("**How long are results stored, and how long will the platform be maintained?**"):
        st.markdown("""
            User submissions and results are currently stored for **30 days** after job completion.
            This retention period may be extended in the future as storage capacity and usage patterns evolve.

            BioAutoML-FAST is planned to be actively maintained for **at least five years**, with regular updates, new features, and model repository expansions released over time.
        """)

    with st.expander("**Can I use BioAutoML-FAST for commercial purposes?**"):
        st.markdown("""
            Yes. BioAutoML-FAST is released under the **MIT License**, which permits both academic and commercial use, including modification and redistribution, provided that the original copyright notice and license are retained.
        """)

def tutorials():
    st.markdown("### Video Tutorials")

    st.markdown("Here you will find practical use-case video tutorials that guide you through the platform and demonstrate how to navigate and use its main features:")

    with st.expander("**Use case 1: Exploring results in the platform**"):
        _l, col, _r = st.columns([1, 3, 1])
        with col:
            st.video("videos/video1.mp4")

    with st.expander("**Use case 2: Training a classification model from scratch to predict labeled data**"):
        _l, col, _r = st.columns([1, 3, 1])
        with col:
            st.video("videos/video2.mp4")

    with st.expander("**Use case 3: Training a regression model from scratch to predict unlabeled data**"):
        _l, col, _r = st.columns([1, 3, 1])
        with col:
            st.video("videos/video3.mp4")

    with st.expander("**Use case 4: Reusing models trained within the platform**"):
        _l, col, _r = st.columns([1, 3, 1])
        with col:
            st.video("videos/video4.mp4")

    with st.expander("**Use case 5: Using trained models from the repository to predict unlabeled data**"):
        _l, col, _r = st.columns([1, 3, 1])
        with col:
            st.video("videos/video5.mp4")

    with st.expander("**Use case 6: Adding new models to the repository**"):
        _l, col, _r = st.columns([1, 3, 1])
        with col:
            st.video("videos/video6.mp4")

    with st.expander("**Use case 7: Getting more help**"):
        _l, col, _r = st.columns([1, 3, 1])
        with col:
            st.video("videos/video7.mp4")

def runUI():
    with st.expander("Using the platform"):
        st.info("""
            This section provides frequently asked questions and video tutorials to help you get started with BioAutoML-FAST.

            If this page does not answer your questions, please feel free to contact the corresponding authors by email:
            brenoslivio@usp.br, bonidia@utfpr.edu.br, ulisses.rocha@ufz.de.
        """)

    _l, col2, _r = st.columns([2, 3, 2])
    with col2:
        st.image("imgs/overview.webp", caption="Overview of the platform with the four main modules.")

    ai_help()

    faq()

    tutorials()

if __name__ == "__main__":
    runUI()
