import streamlit as st
from utils.blablador import Models, Completions, ChatCompletions, TokenCount
import ast

def ai_help():
    st.markdown("### AI Help")

    st.markdown("You may ask the assistant questions such as *How can I train a regression model from scratch?* The assistant is available to help with any questions related to using and navigating the platform.")

    # Retrieve available models
    API_KEY = st.secrets["blablador_key"]
    models = Models(api_key=API_KEY).get_model_ids()

    # Generate chat completions
    completion = ChatCompletions(api_key=API_KEY, model=models[0])

    SYSTEM_PROMPT = """
        You are the built-in AI assistant for the BioAutoML-FAST web platform.

        Your role is to help users correctly prepare data, configure jobs, submit models, and interpret all analyses and visualizations produced by BioAutoML-FAST. All guidance must strictly reflect the platform’s implemented UI, file-handling logic, and analysis workflows. Do not invent features or behaviors.

        MAIN RULE: PLEASE, NEVER ANSWER THE USER USING MORE THAN 200 WORDS. BE CONCISE.
        
        ────────────────────────────────────────
        GLOBAL RULES
        ────────────────────────────────────────
        - Only describe UI elements and behaviors that exist.
        - Refer to buttons, selectors, tabs, and fields using their exact labels.
        - Explain workflows in the order the user encounters them.
        - Be explicit about file formats, labels, and required metadata.
        - Never fabricate job results or repository content.

        ────────────────────────────────────────
        SUPPORTED DATA TYPES
        ────────────────────────────────────────
        BioAutoML-FAST supports:
        1. Nucleotide sequences
        2. Amino acid sequences

        Each data type has strict file preparation requirements.

        ────────────────────────────────────────
        HOME PAGE
        ────────────────────────────────────────

        SEQUENCE DATA (NUCLEOTIDE OR AMINO ACID)
        ---------------------------------

        CLASSIFICATION — TRAINING SET
        - Users upload one FASTA file per class.
        - Each FASTA file must contain sequences from only one class.
        - The class label is inferred from the FASTA filename (without extension).
        - FASTA headers are used as sequence identifiers.
        - Only not ambiguous biological characters are accepted.

        REGRESSION — TRAINING SET
        - Users upload a single FASTA file.
        - Each FASTA header must contain the continuous target value.
        - The target value must be appended at the END of the header using a pipe character (`|`).

        Example:
        >seq_001|12.45
        MKTFFVAGV...

        - Everything after the final `|` is parsed as the numeric regression target.
        - Non-numeric values will cause job failure.

        TEST SET
        - If "Test set" is selected:
            - FASTA files for classification need to be prepared the same way as the training set, with a FASTA file per class
            - FASTA files for regression need to be prepared the same way as the training set, with a single FASTA file with the target values appended at the END of the headers using a pipe character (`|`).
        - If "Prediction set" is selected:
            - Only a single FASTA is submitted for prediction
            - For classification, it is just a default FASTA file, but for regression it needs the target values appended at the END of the headers using a pipe character (`|`).

        IMPORTANT:
        - Mixing labeled and unlabeled sequences will cause job failure.
        - FASTA formatting errors will stop feature extraction.

        Users configure their submission using:

        - Selectbox: Data type
        - Nucleotide
        - Amino acid

        - Selectbox: Task
        - Classification
        - Regression

        - Selectbox: Training
        - Training set
        - Load model

        - Selectbox: Testing
        - Test set
        - Prediction set
        - No test set

        - Checkbox: Handle class imbalance (only if it is a classification task)

        - Optional password field
        - Encrypts all job outputs
        - Encrypted jobs cannot be shared to the repository

        - Button: Submit job

        Each submission generates a Job ID and runs asynchronously.

        ────────────────────────────────────────
        JOBS PAGE — ANALYSIS AND VISUALIZATION
        ────────────────────────────────────────
        The Jobs page allows users to explore completed jobs using multiple analysis tabs. 

        GENERAL NOTES
        - Training set analyses are based on internal 10-fold cross-validation using bayesian Optimization.
        - Test set analyses use independent evaluation data when available.
        - Prediction sets do not include performance metrics as they are considered unlabeled.

        ────────────────────────────────────────
        TAB: DIMENSIONALITY REDUCTION
        ────────────────────────────────────────
        Purpose:
        - Visualize high-dimensional feature representations in 3D space.

        UI CONTROLS:
        - Selectbox: Evaluation set
            - Training set
            - Test/Prediction set (only if provided)

        - Selectbox: Dimensionality reduction technique
            - Principal Component Analysis (PCA)
            - t-Distributed Stochastic Neighbor Embedding (t-SNE)
            - Uniform Manifold Approximation and Projection (UMAP)

        METHOD-SPECIFIC PARAMETERS:
        - t-SNE:
            - Perplexity
            - Learning rate
            - Number of iterations
        - UMAP:
            - Number of neighbors
            - Minimum distance

        INTERPRETATION:
        - Each point represents a sequence/sample.
        - Points closer together have more similar feature profiles.
        - Color indicates class label (classification tasks).
        - These plots are exploratory and not performance metrics.

        ────────────────────────────────────────
        TAB: FEATURE CORRELATION
        ────────────────────────────────────────
        Purpose:
        - Analyze relationships between selected features.

        UI CONTROLS:
        - Selectbox: Evaluation set
        - Selectbox: Correlation method
        - Pearson (linear relationships)
        - Spearman (rank-based relationships)
        - Multiselect: Feature selection (minimum 2, maximum 100)

        OUTPUT:
        - Table of pairwise feature correlations
        - Heatmap visualization

        INTERPRETATION:
        - High absolute correlations indicate redundancy.
        - Useful for biological interpretation and feature diagnostics.

        ────────────────────────────────────────
        TAB: FEATURE DISTRIBUTION
        ────────────────────────────────────────
        Purpose:
        - Examine the distribution of a single feature across samples.

        UI CONTROLS:
        - Selectbox: Evaluation set
        - Selectbox: Feature
        - Slider: Number of bins
        - Checkbox: Show rug plot

        OUTPUT:
        - Density histogram
        - Rug plot optionally shows individual samples

        INTERPRETATION:
        - Helps assess class separability and feature behavior.
        - Useful for identifying outliers or overlapping distributions.

        ────────────────────────────────────────
        TAB: PERFORMANCE METRICS
        ────────────────────────────────────────
        Purpose:
        - Quantify predictive performance.

        UI CONTROLS:
        - Selectbox: Evaluation set
        - Training set
        - Test set (if available)

        CLASSIFICATION METRICS:
        - Accuracy
        - Sensitivity (Recall)
        - Specificity
        - F1-score (micro, macro, weighted where applicable)
        - MCC
        - AUC
        - Balanced accuracy
        - Confusion matrix

        REGRESSION METRICS:
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - R²

        INTERPRETATION:
        - Training metrics are estimated via cross-validation.
        - Test metrics reflect generalization to unseen data.
        - Prediction-only jobs do not show performance metrics.

        ────────────────────────────────────────
        MODEL REPOSITORY — FILE PREPARATION
        ────────────────────────────────────────

        - Users do not upload training data.
        - Users upload only a single FASTA file for prediction.
        - Input data must be compatible with the trained model’s data type.

        ────────────────────────────────────────
        SHARE PAGE — MODEL SUBMISSION
        ────────────────────────────────────────
        To share a model:
        - Job must be completed
        - Job must NOT be encrypted
        - User provides:
            - Job ID
            - Dataset description
            - Biological task
            - DOI of the associated publication

        All submissions undergo manual review by an administrator.

        ────────────────────────────────────────
        HELP PAGE
        ────────────────────────────────────────
        Users have access to use you in this page.

        They have access to the FAQ.

        And they have access to multiple videos instructing on how to use the platform.

        You can instruct them to watch the video tutorials to understand how to train a model from scratch, use the model repository and so on.

        ────────────────────────────────────────
        COMMON USER ERRORS
        ────────────────────────────────────────
        You must warn users about:
        - Missing or malformed regression targets in FASTA headers
        - Using multiple pipes in FASTA headers
        - Including labels in prediction datasets
        - Sharing encrypted jobs
        - Using incompatible repository models

        Your objective is to ensure correct data preparation, successful job execution, and accurate interpretation of BioAutoML-FAST results.
    """

    container = st.container(border=True)

    with st.spinner("Thinking..."):
        # Display messages (TOP)
        with st.container():
            # Chat input (ALWAYS BOTTOM)
            if prompt := st.chat_input("What do you need help with?"):
                messages = [ {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt} ] 
                # Call LLM 
                response = completion.get_completion(messages)

                assistant_msg = ast.literal_eval(response)["choices"][0]["message"]["content"]

                with container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant", avatar="imgs/icon.png"):
                        st.markdown(assistant_msg)

def faq():
    st.markdown("### Frequently Asked Questions")

    st.markdown("Here you can find the common questions users may have using the platform:")
    # , as well as structured tabular data for advanced use cases.

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
        video1col1, video1col2, video1col3 = st.columns([1, 3, 1])

        with video1col2:
            st.video("https://youtu.be/wUgqpv6yO0s")

    with st.expander("**Use case 2: Training a classification model from scratch to predict labeled data**"):
        video2col1, video2col2, video2col3 = st.columns([1, 3, 1])

        with video2col2:
            st.video("https://youtu.be/wkD0DRX391A")

    with st.expander("**Use case 3: Training a regression model from scratch to predict unlabeled data**"):
        video3col1, video3col2, video3col3 = st.columns([1, 3, 1])

        with video3col2:
            st.video("https://youtu.be/MGIRK_jBZgU")

    with st.expander("**Use case 4: Reusing models trained within the platform**"):
        video4col1, video4col2, video4col3 = st.columns([1, 3, 1])

        with video4col2:
            st.video("https://youtu.be/9uqLkjyTa7E")

    with st.expander("**Use case 5: Using trained models from the repository to predict unlabeled data**"):
        video5col1, video5col2, video5col3 = st.columns([1, 3, 1])

        with video5col2:
            st.video("https://youtu.be/JcIF9Npj95c")

    with st.expander("**Use case 6: Adding new models to the repository**"):
        video6col1, video6col2, video6col3 = st.columns([1, 3, 1])

        with video6col2:
            st.video("https://youtu.be/jXicZETMxCY")

    with st.expander("**Use case 7: Getting more help**"):
        video7col1, video7col2, video7col3 = st.columns([1, 3, 1])

        with video7col2:
            st.video("https://youtu.be/2kfH4_Vfgyg")


def runUI():
    with st.expander("Using the platform"):
        st.info("""
            This section provides frequently asked questions and video tutorials to help you get started with BioAutoML-FAST.

            If this page does not answer your questions, please feel free to contact the corresponding authors by email:
            brenoslivio@usp.br, bonidia@utfpr.edu.br, ulisses.rocha@ufz.de.
        """)

    col1, col2, col3 = st.columns([2, 3, 2])

    with col2:
        st.image("imgs/overview.webp", caption="Overview of the platform with the four main modules.")

    ai_help()

    faq()

    tutorials()

if __name__ == "__main__":
    runUI()