import streamlit as st
import polars as pl
import pandas as pd
import plotly.graph_objects as go
from itertools import chain
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import os
import utils
import joblib
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn import tree
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from utils.tasks import manager
import tarfile
import io
import secrets
import base64
import tempfile
import shutil
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode
import shap
import csv
import gzip

def _cleanup_previous_temp():
    prev = st.session_state.get("temp_extract_path")
    if prev and os.path.exists(prev):
        try:
            shutil.rmtree(prev)
        except Exception:
            pass
        del st.session_state["temp_extract_path"]

def load_reduction_data(job_path, evaluation):
    """Load and cache data for dimensionality reduction"""
    if evaluation == "Training set":
        if "model" in st.session_state:
            features = st.session_state["model"]["train"]
            labels = pd.DataFrame(st.session_state["model"]["train_labels"], columns=["label"])["label"].tolist()
            nameseqs = st.session_state["model"]["nameseq_train"]
        else:
            features = pd.read_csv(os.path.join(job_path, "best_descriptors/best_train.csv"))
            labels = pd.read_csv(os.path.join(job_path, "feat_extraction/flabeltrain.csv"))["label"].tolist()
            nameseqs = pd.read_csv(os.path.join(job_path, "feat_extraction/fnameseqtrain.csv"))["nameseq"].tolist()
    else:
        if os.path.exists(os.path.join(job_path, "feat_extraction/test_labels.csv")):
            features = pd.read_csv(os.path.join(job_path, "feat_extraction/test.csv"))
            labels = pd.read_csv(os.path.join(job_path, "feat_extraction/test_labels.csv"))["label"].tolist()
        else:
            features = pd.read_csv(os.path.join(job_path, "best_descriptors/best_test.csv"))
            labels = pd.read_csv(os.path.join(job_path, "feat_extraction/flabeltest.csv"))["label"].tolist()
        
        nameseqs = pd.read_csv(os.path.join(job_path, "feat_extraction/fnameseqtest.csv"))["nameseq"].tolist()
    
    return features, labels, nameseqs

def scale_features(features):
    """Scale features with caching"""

    if "imputer" in st.session_state["model"]:
        features = pd.DataFrame(st.session_state["model"]["imputer"].transform(features), columns=features.columns)

    if "scaler" in st.session_state["model"]:
        features = st.session_state["model"]["scaler"].transform(features)

    return features

def create_reduction_plot(reduced_data, labels, names_df, reduction_method):
    """
    reduced_data: numpy array shape (n_samples, 3)
    labels: list/array length n_samples with label values
    names_df: DataFrame with columns ['label','nameseq'] in the same order as labels OR
              (preferred) used only to generate per-point hover text aligned with labels.
    """
    fig = go.Figure()
    reduced_data = np.asarray(reduced_data)
    if reduced_data.ndim != 2 or reduced_data.shape[1] < 3:
        raise ValueError("reduced_data must be shape (n_samples, >=3)")

    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    colors = utils.get_colors(len(unique_labels))

    # Build per-point hover text aligned with original order.
    # If names_df is the zipped DataFrame we created earlier (label, nameseq)
    if isinstance(names_df, pd.DataFrame) and 'nameseq' in names_df.columns:
        # assume names_df is aligned with labels order
        point_names = names_df['nameseq'].astype(str).tolist()
    else:
        # fallback: use index as name
        point_names = [str(i) for i in range(len(labels))]

    for i, label in enumerate(unique_labels):
        mask = (labels == label)
        xs = reduced_data[mask, 0]
        ys = reduced_data[mask, 1]
        zs = reduced_data[mask, 2]
        hover_text = [n for n in np.array(point_names)[mask]]

        fig.add_trace(go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode='markers',
            name=str(label),
            marker=dict(color=colors[i], size=5, opacity=0.8),
            hoverlabel=dict( font=dict( family="Open Sans, History Sans Pro Light", color="white", size=12 ), bgcolor="black" ),
            hovertemplate=hover_text,
            hoverinfo='text',
            showlegend=True
        ))

    fig.update_layout(
        height=600,
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3'
        ),
        title=reduction_method,
        margin=dict(t=30, b=50)
    )
    return fig

def dimensionality_reduction():

    with st.expander("What **Dimensionality Reduction** shows"):
        st.info(
            """
            This tab provides a **low-dimensional visual representation of your data**, helping you explore 
            overall structure and relationships between samples.

            High-dimensional sequence features are projected into a **3-dimensional space**, where each point 
            represents a sample. Samples that appear closer together have more similar feature profiles, while 
            distant points are more dissimilar.

            You can choose between different dimensionality reduction techniques. **PCA** summarizes the main 
            sources of variation in the data, while **t-SNE** and **UMAP** emphasize local structure and are 
            particularly useful for revealing clusters or group separation.

            Coloring by class label allows visual assessment of how well samples separate based on their 
            features. These visualizations are intended for **exploratory analysis and interpretation**, not 
            for direct performance evaluation or quantitative conclusions.
            """
        )

    dim_col1, dim_col2 = st.columns(2)

    with dim_col1:
        # Evaluation set selection
        df_job_info = pl.read_csv(os.path.join(st.session_state["job_path"], "job_info.tsv"), separator='\t')
    
        has_test_set = True if df_job_info["testing_set"].item() != "No test set" else False
    
        evaluation = st.selectbox(
            ":mag_right: Evaluation set",
            ["Training set", "Test/Prediction set"] if has_test_set else ["Training set"],
            key="reduction"
        )

        if evaluation == "Training set":
            features, labels, nameseqs = load_features(st.session_state["job_path"], True)
        else:
            features, labels, nameseqs = load_features(st.session_state["job_path"], False)
            nameseqs = nameseqs["nameseq"].tolist()

        if "label_encoder" not in st.session_state["model"]:
            class_label = st.session_state["model"]["train_stats"]["class"].item()
            labels["label"] = class_label
            
        labels = labels["label"].tolist()

        names = pd.DataFrame(list(zip(labels, nameseqs)), columns=["label", "nameseq"])
        
        # Scale features with caching
        scaled_data = scale_features(features)

        # Reduction method selection
        reduction = st.selectbox(
            "Select dimensionality reduction technique", 
            ["Principal Component Analysis (PCA)",
             "t-Distributed Stochastic Neighbor Embedding (t-SNE)",
             "Uniform Manifold Approximation and Projection (UMAP)"]
        )

        # Reduction parameters
        reducer = None

        if reduction == "t-Distributed Stochastic Neighbor Embedding (t-SNE)":
            perplexity = st.slider("Perplexity", min_value=5, max_value=50, value=30)
            learning_rate = st.slider("Learning rate", min_value=10, max_value=1000, value=200)
            max_iter = st.slider("Number of iterations", min_value=100, max_value=10000, value=1000)
            reducer = TSNE(n_components=3, perplexity=perplexity, 
                          learning_rate=learning_rate, max_iter=max_iter, n_jobs=-1)

        elif reduction == "Uniform Manifold Approximation and Projection (UMAP)":
            n_neighbors = st.slider("Number of neighbors", min_value=2, max_value=100, value=15)
            min_dist = st.slider("Minimum distance", min_value=0.0, max_value=1.0, value=0.1)
            reducer = UMAP(n_components=3, n_neighbors=n_neighbors, 
                         min_dist=min_dist, n_jobs=-1)
        else:
            reducer = PCA(n_components=3)

    with dim_col2:
        if reducer:
            with st.spinner(f'Computing {reduction}...'):
                # Compute reduction with caching

                # if "reducer" not in st.session_state:
                #     if evaluation == "Training set":
                #         st.session_state["reducer"] = reducer.fit(scaled_data)
                #         reduced_data = reducer.transform(scaled_data)
                # else:
                #     reduced_data = st.session_state["reducer"].transform(scaled_data)

                reduced_data = reducer.fit_transform(scaled_data)

                # Create plot with caching
                fig = create_reduction_plot(reduced_data, labels, names, reduction)
                st.plotly_chart(fig, use_container_width=True)

def compute_correlation_matrix(features, method):
    """Compute correlation matrix for selected features"""
    return features.corr(method=method.lower())

def create_correlation_df(corr_matrix):
    """Create long-form correlation dataframe without redundant pairs"""
    corr_df = corr_matrix.stack().reset_index()
    corr_df.columns = ["Feature 1", "Feature 2", "Correlation coefficient"]

    # Remove self-correlations
    corr_df = corr_df[corr_df["Feature 1"] != corr_df["Feature 2"]]

    # Remove redundant symmetric pairs (A,B) vs (B,A)
    corr_df[["f_min", "f_max"]] = np.sort(
        corr_df[["Feature 1", "Feature 2"]].values,
        axis=1
    )
    corr_df = corr_df.drop_duplicates(subset=["f_min", "f_max"])
    corr_df = corr_df.drop(columns=["f_min", "f_max"])

    # Sort by absolute correlation
    corr_df["abs_corr"] = corr_df["Correlation coefficient"].abs()
    corr_df = corr_df.sort_values("abs_corr", ascending=False).drop(columns="abs_corr")

    return corr_df.reset_index(drop=True)

def create_correlation_heatmap(corr_matrix):
    """Create correlation heatmap"""
    fig = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Feature correlation heatmap"
    )

    fig.update_traces(
        hovertemplate=(
            "Feature 1: %{x}<br>"
            "Feature 2: %{y}<br>"
            "Correlation: %{z:.3f}<extra></extra>"
        )
    )

    fig.update_layout(height=500, margin=dict(t=40, b=40))
    return fig

def feature_correlation():

    with st.expander("What **Feature Correlation** shows"):
        st.info(
            """
            This tab explores **how selected features relate to each other**.

            Choose a subset of features (between **2 and 100**) and a correlation method.
            Pearson captures linear relationships, while Spearman captures monotonic trends
            and is more robust to outliers.

            The table lists all pairwise correlations among the selected features,
            sorted by correlation strength, and the heatmap provides a visual overview.
            """
        )

    col1, col2 = st.columns(2)

    with col1:
        df_job_info = pl.read_csv(
            os.path.join(st.session_state["job_path"], "job_info.tsv"),
            separator="\t"
        )

        has_test_set = df_job_info["testing_set"].item() != "No test set"

        evaluation = st.selectbox(
            ":mag_right: Evaluation set",
            ["Training set", "Test/Prediction set"] if has_test_set else ["Training set"],
            key="correlation_eval"
        )

    with col2:
        correlation_method = st.selectbox(
            "Select correlation method:",
            ["Pearson", "Spearman"]
        )

    # Load features
    if evaluation == "Training set":
        features, _, _ = load_features(st.session_state["job_path"], True)
    else:
        features, _, _ = load_features(st.session_state["job_path"], False)

    if "imputer" in st.session_state["model"]:
        features = pd.DataFrame(
            st.session_state["model"]["imputer"].transform(features),
            columns=features.columns
        )

    if "mapper" in st.session_state:
        features = features.rename(columns=st.session_state["mapper"])

    feature_names = list(features.columns)

    selected_features = st.multiselect(
        "Select features for correlation analysis",
        options=feature_names,
        default=feature_names[:10],
        max_selections=100,
        help="Select between 2 and 100 features"
    )

    # Validation
    if len(selected_features) < 2:
        st.warning("Please select at least **2 features** to compute correlations.")
        return

    # Subset features
    selected_df = features[selected_features]

    # Compute correlations
    with st.spinner("Computing correlations..."):
        corr_matrix = compute_correlation_matrix(selected_df, correlation_method)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Pairwise feature correlations**", help="Sorted by absolute correlation coefficient")
        correlation_df = create_correlation_df(corr_matrix)
        st.dataframe(
            correlation_df,
            hide_index=True,
            use_container_width=True
        )

    with col2:
        with st.spinner("Generating heatmap..."):
            fig = create_correlation_heatmap(corr_matrix)
            st.plotly_chart(fig, use_container_width=True)

def load_features(job_path, training):

    if training:
        if "model" in st.session_state:
            features = st.session_state["model"]["train"]
            labels = pd.DataFrame(st.session_state["model"]["train_labels"], columns=["label"])
            nameseqs = st.session_state["model"]["nameseq_train"]
        else:
            features = pd.read_csv(os.path.join(job_path, "best_descriptors/best_train.csv"))
            labels = pd.read_csv(os.path.join(job_path, "feat_extraction/flabeltrain.csv"))
            nameseqs = pd.read_csv(os.path.join(job_path, "feat_extraction/fnameseqtrain.csv"))
    else:
        if os.path.exists(os.path.join(job_path, "feat_extraction/test_labels.csv")):
            features = pd.read_csv(os.path.join(job_path, "feat_extraction/test.csv"))
            labels = pd.read_csv(os.path.join(job_path, "feat_extraction/test_labels.csv"))
        else:
            features = pd.read_csv(os.path.join(job_path, "best_descriptors/best_test.csv"))
            labels = pd.read_csv(os.path.join(job_path, "feat_extraction/flabeltest.csv"))
        nameseqs = pd.read_csv(os.path.join(job_path, "feat_extraction/fnameseqtest.csv"))
        
    return features, labels, nameseqs

def create_distplot(fig_data, unique_labels, bin_edges, color_map, fig_rug_text, selected_feature):
    """Create and cache the distribution plot"""
    fig = ff.create_distplot(
        fig_data,
        unique_labels,
        bin_size=bin_edges,
        colors=color_map,
        rug_text=fig_rug_text if fig_rug_text is not None else False,
        histnorm="probability density",
        show_rug=fig_rug_text is not None  # Only show rug if text is provided
    )
    
    fig.update_layout(
        title=f"Feature distribution for {selected_feature}",
        xaxis_title=selected_feature,
        yaxis_title="Density",
        height=800,
        margin=dict(t=30, b=50)
    )

    return fig

def feature_distribution():
    
    with st.expander("What **Feature Distribution** shows"):
        st.info(
            """
            This tab shows how the values of a **single selected feature** are distributed across your data.

            You can choose whether to view feature distributions from the **training set** or, when available, 
            the **test/prediction set**. For classification tasks, distributions are shown separately for each class, 
            allowing you to visually compare how well a feature distinguishes between groups.

            The histogram illustrates the overall spread and frequency of feature values, while an optional 
            rug plot displays individual samples along the axis for more detailed inspection. This can help 
            identify overlap between classes, outliers, or characteristic value ranges.

            Feature distribution plots provide intuitive insight into how individual sequence descriptors 
            behave in the dataset and how they may contribute to the model’s predictions.
            """
        )

    # Determine evaluation set options
    df_job_info = pl.read_csv(os.path.join(st.session_state["job_path"], "job_info.tsv"), separator='\t')
    
    has_test_set = True if df_job_info["testing_set"].item() != "No test set" else False
    
    evaluation = st.selectbox(
        ":mag_right: Evaluation set",
        ["Training set", "Test/Prediction set"] if has_test_set else ["Training set"],
        help="Training set evaluated with 10-fold cross-validation",
        key="distribution"
    )

    if evaluation == "Training set":
        features, labels, nameseqs = load_features(st.session_state["job_path"], True)
    else:
        features, labels, nameseqs = load_features(st.session_state["job_path"], False)

    if "imputer" in st.session_state["model"]:
        features = pd.DataFrame(st.session_state["model"]["imputer"].transform(features), columns=features.columns)

    if "mapper" in st.session_state:
        features = features.rename(columns=st.session_state["mapper"])

    col1, col2 = st.columns(2)

    # Select feature to plot
    with col1:
        selected_feature = st.selectbox("Select a feature", features.columns)
        show_rug = st.checkbox("Show rug plot", value=False, 
                             help="Toggle to show/hide individual data points along the axis")

    # Get unique labels and assign colors
    if "label_encoder" not in st.session_state["model"]:
        class_label = st.session_state["model"]["train_stats"]["class"].item()
        labels["label"] = class_label
    
    unique_labels = labels["label"].unique()
    color_map = utils.get_colors(len(unique_labels))[:len(unique_labels)]

    with col2:
        num_bins = st.slider("Number of bins", min_value=5, max_value=50, value=30)

    # Prepare plot data
    with st.spinner('Preparing data...'):
        fig_data = []
        fig_rug_text = []
        feature_data = features[selected_feature].values.astype(float)

        for label in unique_labels:
            group_indices = list(chain(*(labels == label).values))
            group_data = feature_data[group_indices]
            fig_data.append(group_data)
            if show_rug:  # Only prepare rug text if needed
                group_names = nameseqs[group_indices]
                fig_rug_text.append(group_names)
            else:
                fig_rug_text.append(None)

        bin_edges = np.histogram(fig_data[0], bins=num_bins)[1]

    # Create and display plot
    with st.spinner('Generating visualization...'):
        fig = create_distplot(
            fig_data,
            unique_labels,
            bin_edges,
            color_map,
            fig_rug_text if show_rug else None,  # Pass None if rug is disabled
            selected_feature
        )
        st.plotly_chart(fig, use_container_width=True)

def create_confusion_matrix_figure(df):
    labels = df.columns[1:-1].tolist()
    values = df.iloc[0:-1, 1:-1].values.tolist()

    # Create annotations for each cell
    annotations = []
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=str(value),
                    font=dict(size=16, color='black' if value < max(map(max, values))/2 else 'white'),
                    showarrow=False
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=values,
        hoverinfo='text',
        hovertext=[ [f"{value} {'correctly classified' if i==j else 'misclassified'}" 
                for j, value in enumerate(row)] for i, row in enumerate(values) ],
        hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>%{hovertext}<extra></extra>'
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted label',
        yaxis_title='True label',
        template='plotly_white',
        margin=dict(t=30, b=50),
        annotations=annotations
    )
    
    return fig

def performance_metrics(task):

    with st.expander("What **Performance Metrics** shows"):
        st.info(
            """
            This tab summarizes **how well the trained model performs** when making predictions.

            You can choose whether to view results from the **training set** or, when available, an **independent test set**.  
            Training-set results are estimated using repeated internal validation, which helps assess how stable and reliable 
            the model is. Test-set results reflect how the model performs on previously unseen data. Note that data
            submitted to the Model Repository is considered unlabeled, so it does not show the test performance for this scenario.

            For **classification tasks**, this tab reports commonly used performance measures such as accuracy, sensitivity, 
            specificity, F1-score, and related statistics. These metrics indicate how correctly the model assigns sequences 
            to their respective classes. When multiple classes are present, averaged metrics are shown to provide a fair 
            overall evaluation.

            For **regression tasks**, error-based metrics and correlation scores are displayed, describing how close the 
            predicted values are to the true biological measurements.

            When applicable, a **confusion matrix** is also shown. This visual summary helps you understand which classes are 
            correctly predicted and where misclassifications occur, offering an intuitive view of model behavior beyond 
            single-number scores.
            """
        )

    df_job_info = pl.read_csv(os.path.join(st.session_state["job_path"], "job_info.tsv"), separator='\t')

    has_test_set = True if df_job_info["testing_set"].item() == "Test set" else False
    
    evaluation_options = ["Training set", "Test set"] if has_test_set else ["Training set"]
    evaluation = st.selectbox(
        ":mag_right: Evaluation set", 
        evaluation_options,
        help="Training set evaluated with 10-fold cross-validation"
    )

    if task == "Classification":
        col1, col2 = st.columns(2)
    else:
        col1 = st.container()  # single column
        col2 = None

    with col1:
        if evaluation == "Training set":
            if "model" in st.session_state:
                df_cv = st.session_state["model"]["cross_validation"]
            else:
                df_cv = pd.read_csv(os.path.join(st.session_state["job_path"], "training_kfold(10)_metrics.csv"))

            metrics = []

            if task == "Classification":
                if "F1_micro" not in df_cv.columns:
                    metrics.extend([
                        f"**Accuracy:** {df_cv['ACC'].item():.3f} ± {df_cv['std_ACC'].item():.3f}",
                        f"**Sensitivity:** {df_cv['Sn'].item():.3f} ± {df_cv['std_Sn'].item():.3f}",
                        f"**Specificity:** {df_cv['Sp'].item():.3f} ± {df_cv['std_Sp'].item():.3f}",
                        f"**MCC:** {df_cv['MCC'].item():.3f} ± {df_cv['std_MCC'].item():.3f}",
                        f"**AUC:** {df_cv['AUC'].item():.3f} ± {df_cv['std_AUC'].item():.3f}",
                        f"**F1-score:** {df_cv['F1'].item():.3f} ± {df_cv['std_F1'].item():.3f}",
                        f"**Balanced accuracy:** {df_cv['balanced_ACC'].item():.3f} ± {df_cv['std_balanced_ACC'].item():.3f}",
                        f"**Kappa:** {df_cv['kappa'].item():.3f} ± {df_cv['std_kappa'].item():.3f}",
                        f"**G-mean:** {df_cv['gmean'].item():.3f} ± {df_cv['std_gmean'].item():.3f}"
                    ])
                else:
                    metrics.extend([
                        f"**Accuracy:** {df_cv['ACC'].item():.3f} ± {df_cv['std_ACC'].item():.3f}",
                        f"**Sensitivity (macro avg.):** {df_cv['Sn'].item():.3f} ± {df_cv['std_Sn'].item():.3f}",
                        f"**Specificity (macro avg.):** {df_cv['Sp'].item():.3f} ± {df_cv['std_Sp'].item():.3f}",
                        f"**F1-score (micro avg.):** {df_cv['F1_micro'].item():.3f} ± {df_cv['std_F1_micro'].item():.3f}",
                        f"**F1-score (macro avg.):** {df_cv['F1_macro'].item():.3f} ± {df_cv['std_F1_macro'].item():.3f}",
                        f"**F1-score (weighted avg.):** {df_cv['F1_weighted'].item():.3f} ± {df_cv['std_F1_weighted'].item():.3f}",
                        f"**MCC:** {df_cv['MCC'].item():.3f} ± {df_cv['std_MCC'].item():.3f}",
                        f"**Kappa:** {df_cv['kappa'].item():.3f} ± {df_cv['std_kappa'].item():.3f}"
                    ])

            elif task == "Regression":
                metrics.extend([
                    f"**Mean Absolute Error:** {df_cv['mean_absolute_error'].item():.3f} ± {df_cv['std_mean_absolute_error'].item():.3f}",
                    f"**Mean Squared Error:** {df_cv['mean_squared_error'].item():.3f} ± {df_cv['std_mean_squared_error'].item():.3f}",
                    f"**Root Mean Squared Error:** {df_cv['root_mean_squared_error'].item():.3f} ± {df_cv['std_root_mean_squared_error'].item():.3f}",
                    f"**R2:** {df_cv['r2'].item():.3f} ± {df_cv['std_r2'].item():.3f}"
                ])
            
            for metric in metrics:
                st.markdown(metric)

        else:
            df_report = pd.read_csv(os.path.join(st.session_state["job_path"], "metrics_test.csv"))
            df_report = df_report.rename(columns={"Unnamed: 0": ""})

            # Format numeric columns except "support"
            for col in df_report.select_dtypes(include="number").columns:
                if col != "support":
                    df_report[col] = df_report[col].map(lambda x: f"{x:.3f}")

            df_report["support"] = df_report["support"].map(lambda x: f"{int(x)}")

            df_report.loc[df_report[""] == "accuracy", "support"] = ""

            st.dataframe(df_report, hide_index=True, use_container_width=True)

            path_metrics_other = os.path.join(os.path.join(st.session_state["job_path"], "metrics_other.csv"))

            if os.path.exists(path_metrics_other):
                df_metrics_other = pd.read_csv(path_metrics_other)
                auc_value = df_metrics_other[df_metrics_other["Metric"] == "AUC"]["Value"].item()
                st.markdown(f"**AUC:** {auc_value:.3f}")

    if task == "Classification":
        with col2:
            if evaluation == "Training set":
                if "model" in st.session_state:
                    df = st.session_state["model"]["confusion_matrix"]
                else:
                    df = pd.read_csv(os.path.join(st.session_state["job_path"], "training_confusion_matrix.csv"))
            else:
                df = pd.read_csv(os.path.join(st.session_state["job_path"], "test_confusion_matrix.csv"))

            fig = create_confusion_matrix_figure(df)

            with st.spinner('Loading visualization...'):
                st.plotly_chart(fig, use_container_width=True)

def load_predictions(job_path):
    """Load and preprocess predictions data with caching"""
    predictions = pd.read_csv(os.path.join(job_path, "test_predictions.csv"))
    predictions.iloc[:,1:-1] = predictions.iloc[:,1:-1] * 100
    return predictions

def show_predictions():

    with st.expander("What **Predictions** shows"):
        st.info(
            """
            This tab displays the **predictions generated by the trained model** for each input sample.

            Each row corresponds to a sequence or sample you submitted, while the columns show the model’s 
            predicted confidence for each possible label or outcome. These values represent how strongly 
            the model associates a given sample with each class.

            The prediction scores are shown as percentages to make them easier to interpret. Higher values 
            indicate greater confidence in a particular prediction, but they should be interpreted as 
            probabilistic support rather than absolute certainty.

            This table allows you to quickly compare predictions across samples, identify clear-cut cases, 
            and highlight ambiguous predictions that may require further biological validation or expert 
            review.
            """
        )

    # Load data with caching
    predictions = load_predictions(st.session_state["job_path"])
    labels = predictions.columns[1:-1]
    
    # Create column config once
    column_config = {
        label: st.column_config.ProgressColumn(
            help="Label probability",
            format="%.2f%%",
            min_value=0,
            max_value=100
        ) for label in labels
    }
    
    # Display with loading indicator
    with st.spinner('Displaying predictions...'):
        st.dataframe(
            predictions.rename(columns={"nameseq": "Sample name"}),
            hide_index=True,
            height=500,
            column_config=column_config,
            use_container_width=True
        )

def load_feature_importance(job_path):
    """Load and cache feature importance data"""
    if "model" in st.session_state:
        df_feat = st.session_state["model"]["feature_importance"]
    else:
        df_feat = pd.read_csv(os.path.join(job_path, "feature_importance.tsv"))

    return df_feat

def create_feature_importance_figure(df):
    """Create and cache the feature importance plot"""
    fig = go.Figure(data=go.Bar(
        x=df["Feature"],
        y=df["Importance"],
        marker=dict(color=df["Importance"], colorscale='blues'),
        hovertemplate='Feature: %{x}<br>Importance: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Importance",
        margin=dict(t=0, b=50)
    )
    
    return fig

def get_shap_data(max_samples=500):
    """
    Prepare background data for SHAP with subsampling.
    """
    X = st.session_state["model"]["train"]

    if "imputer" in st.session_state["model"]:
        X = pd.DataFrame(
            st.session_state["model"]["imputer"].transform(X),
            columns=X.columns
        )

    if "scaler" in st.session_state["model"]:
        X = st.session_state["model"]["scaler"].transform(X)
        X = pd.DataFrame(X, columns=st.session_state["model"]["train"].columns)

    if "mapper" in st.session_state:
        X = X.rename(columns=st.session_state["mapper"])

    # Subsample for performance
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)

    return X

def compute_shap_values(model, X):
    """
    Compute SHAP values for tree-based models.
    Ensures consistent shape:
    - Binary / regression: (n_samples, n_features)
    - Multiclass: (n_samples, n_features, n_classes)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # --- FIX multiclass output ---
    if isinstance(shap_values, list):
        # list[n_classes] of (n_samples, n_features)
        shap_values = np.stack(shap_values, axis=2)

    return explainer, shap_values

def shap_global_importance(shap_values, X):
    """
    Return dataframe with mean absolute SHAP values (1D).
    Works for binary, multiclass, and regression.
    """

    importance = np.mean(np.abs(shap_values), axis=0)

    return (
        pd.DataFrame({
            "Feature": X.columns.to_list(),
            "Mean |SHAP value|": importance
        })
        .sort_values("Mean |SHAP value|", ascending=False)
    )

def feature_importance():

    with st.expander("What **Feature Importance** shows"):
        st.info(
            """
            This tab explains **which features most influenced the model’s predictions** and offers
            two complementary perspectives on feature importance.

            **Tree-based importance** reflects how frequently and effectively each feature was used
            to split decision nodes during model training. Since all models available in this platform
            (Random Forest, XGBoost, and LightGBM) are tree-based, this metric summarizes how much each
            feature contributed to reducing prediction error across the ensemble. It provides a fast,
            global overview of model behavior.

            **SHAP (SHapley Additive exPlanations)** offers a model-agnostic, game-theoretic view of
            feature importance. SHAP values quantify how much each feature contributes to pushing a
            prediction away from the model’s average output for individual samples. In this tab, SHAP
            values are computed on a random subset of up to 500 training samples for efficiency.

            For multiclass models, SHAP values are calculated separately for each class. You can select
            a class to visualize how features influence predictions toward that specific label. Global
            feature importance is then obtained by averaging the absolute SHAP values across samples
            (and across classes when applicable), yielding a comparable ranking for binary, multiclass,
            and regression models.

            The **beeswarm plot** summarizes the distribution of SHAP values for the ten most impactful
            features, while the accompanying table provides a complete ranked list based on mean
            absolute SHAP values.

            Feature importance helps interpret model behavior but does not imply biological causation.
            """
        )

    feat_type = st.selectbox("Feature importance type", ["Tree-based", "SHAP (SHapley Additive exPlanations)"])

    if feat_type == "Tree-based":
        df = load_feature_importance(st.session_state["job_path"])

        if "mapper" in st.session_state:
            df["Feature"] = df["Feature"].replace(st.session_state["mapper"])

        col1, col2 = st.columns(2)
        
        with col1:
            # Create plot with caching
            fig = create_feature_importance_figure(df.iloc[:10, :])
            
            # Display with loading indicator
            st.markdown("**Importance of features regarding model training**", 
                    help="Ten most important features.")
            
            with st.spinner('Rendering feature importance...'):
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(df, hide_index=True)
    elif feat_type == "SHAP (SHapley Additive exPlanations)":
        with st.spinner("Computing SHAP values..."):
            X_shap = get_shap_data(max_samples=500)

            explainer, shap_values = compute_shap_values(
                st.session_state["model"]["clf"],
                X_shap
            )

            col1, col2 = st.columns(2)
        
            with col1:
                st.markdown("**SHAP summary (beeswarm) plot**", help="Ten most impactful features using 500 random samples.")

                shap_values = np.asarray(shap_values)

                if shap_values.ndim == 3:
                    label_choice = st.selectbox("Feature contribution per class", st.session_state["model"]["label_encoder"].inverse_transform(st.session_state["model"]["clf"].classes_))

                    shap_values = shap_values[:, :, st.session_state["model"]["label_encoder"].transform([label_choice])[0]]

                plt.figure(figsize=(8, 6))

                shap.summary_plot(
                    shap_values,
                    X_shap,
                    max_display=10,
                    show=False
                )

                st.pyplot(plt.gcf(), clear_figure=True)
            with col2:
                df_shap = shap_global_importance(shap_values, X_shap)

                st.dataframe(df_shap, hide_index=True)

def model_information(data_type, task):

    with st.expander("What **Model Information** shows"):
        st.info(
            """
            This tab provides a complete overview of the trained machine-learning model used in your analysis.

            On the left, you can see **which algorithm was selected**, the **type of task** being performed 
            (classification or regression), and the **main model settings** that were automatically optimized 
            during training. These settings control how the model learns patterns from your data, but you do 
            not need to adjust them manually.

            You can also **download the trained model file**, which allows you to reuse the model later in this 
            application or share it with collaborators.

            On the right, this tab shows the **extracted and selected features (descriptors)** used to build 
            the model. Descriptors convert biological sequences into numerical values that the model can learn from.

            You can **download the extracted feature datasets** (training and, when available, test or 
            prediction sets) as compressed files. This allows you to inspect the data, perform custom analyses, 
            or reuse the features in external tools and workflows.

            An additional section explains the biological meaning of each descriptor group, helping you 
            understand which sequence properties contributed to the final predictions.
            """
        )

    df_job_info = pl.read_csv(os.path.join(st.session_state["job_path"], "job_info.tsv"), separator='\t')

    has_test_set = True if df_job_info["testing_set"].item() != "No test set" else False

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            cont1, cont2 = st.columns([3, 1])

            with cont1:
                st.markdown("**Model**")

                st.markdown(f"**Task:** {task}")

                def show_params(title, params, keys):
                    st.markdown(f"**Algorithm:** {title}")
                    for k in keys:
                        if k in params and params[k] is not None:
                            st.markdown(f"**{k.replace('_', ' ').title()}:** {params[k]}")

                clf_str = str(st.session_state["model"]["clf"])
                params = st.session_state["model"]["clf"].get_params()

                if "RandomForest" in clf_str:
                    show_params(
                        "Random Forest",
                        params,
                        keys=[
                            "n_estimators", "criterion", "max_depth", "max_features",
                            "min_samples_split", "min_samples_leaf", "bootstrap",
                            "max_leaf_nodes", "min_impurity_decrease", "class_weight"
                        ]
                    )
                elif "XGB" in clf_str:
                    show_params(
                        "XGBoost",
                        params,
                        keys=[
                            "n_estimators", "learning_rate", "max_depth", "gamma",
                            "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
                            "min_child_weight", "objective"
                        ]
                    )
                elif "LGBM" in clf_str:
                    show_params(
                        "LightGBM",
                        params,
                        keys=[
                            "n_estimators", "learning_rate", "max_depth", "boosting_type",
                            "subsample", "colsample_bytree", "num_leaves", "min_child_samples",
                            "reg_alpha", "reg_lambda"
                        ]
                    )
            with cont2:
                with open(os.path.join(st.session_state["job_path"], "trained_model.sav"), "rb") as model_file:
                    st.download_button(
                        label="Download model",
                        data=model_file,
                        file_name="trained_model.sav",
                        mime="application/octet-stream",
                        use_container_width=True,
                        help="SAV file can be loaded into the application"
                    )

                if "RandomForest" in str(st.session_state["model"]["clf"]):
                    st.image("imgs/models/rf.png", use_container_width=True)
                elif "XGB" in str(st.session_state["model"]["clf"]):
                    st.image("imgs/models/xgboost.png", use_container_width=True)
                elif "LGBM" in str(st.session_state["model"]["clf"]):
                    st.image("imgs/models/lightgbm.png", use_container_width=True)

    with col2:
        st.markdown("**Extracted features**", help="Here you can download the features extracted from the submitted datasets. Please note that for larger models from the repository, the process may take a little longer.")

        if st.button("Prepare datasets for download", use_container_width=True):
            if has_test_set:
                download_col1, download_col2 = st.columns(2)
            with st.spinner("Compressing datasets..."):
                
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    suffix=".csv.gz"
                ) as tmp:
                    with gzip.open(tmp.name, mode="wt", newline="") as gz:
                        writer = csv.writer(gz)

                        renamed_columns = [
                            st.session_state["mapper"].get(col, col)
                            for col in st.session_state["model"]["train"].columns
                        ]

                        # Header
                        writer.writerow(
                            ["Sample name"]
                            + renamed_columns
                            + ["label"]
                        )

                        for name, row, label in zip(
                            st.session_state["model"]["nameseq_train"],
                            st.session_state["model"]["train"].itertuples(index=False),
                            st.session_state["model"]["train_labels"],
                        ):
                            writer.writerow([name, *row, label])

                    if has_test_set:
                        test_fnameseq = os.path.join(
                            st.session_state["job_path"],
                            "feat_extraction/fnameseqtest.csv",
                        )
                        test_features = os.path.join(
                            st.session_state["job_path"],
                            "best_descriptors/best_test.csv",
                        )
                        test_labels = os.path.join(
                            st.session_state["job_path"],
                            "feat_extraction/flabeltest.csv",
                        )

                        with tempfile.NamedTemporaryFile(
                            mode="wb",
                            suffix=".csv.gz"
                        ) as tmp_test:
                            with gzip.open(tmp_test.name, mode="wt", newline="") as gz:
                                writer = csv.writer(gz)

                                with (
                                    open(test_fnameseq, newline="") as f_name,
                                    open(test_features, newline="") as f_feat,
                                    open(test_labels, newline="") as f_label,
                                ):
                                    reader_name = csv.reader(f_name)
                                    reader_feat = csv.reader(f_feat)
                                    reader_label = csv.reader(f_label)

                                    # --- Read and rename test feature header ---
                                    next(reader_name, None)
                                    test_feature_header = next(reader_feat)
                                    next(reader_label, None)

                                    renamed_test_columns = [
                                        st.session_state["mapper"].get(col, col)
                                        for col in test_feature_header
                                    ]

                                    # Write final header
                                    writer.writerow(
                                        ["Sample name"]
                                        + renamed_test_columns
                                        + ["label"]
                                    )

                                    # Stream rows
                                    for name_row, feat_row, label_row in zip(
                                        reader_name, reader_feat, reader_label
                                    ):
                                        writer.writerow(
                                            [name_row[0], *feat_row, label_row[0]]
                                        )

                            with download_col1:
                                with open(tmp.name, "rb") as f:
                                    st.download_button(
                                        label="Download training set (.csv.gz)",
                                        data=f,
                                        file_name="train_dataset.csv.gz",
                                        mime="application/gzip",
                                        use_container_width=True
                                    )

                            with download_col2:
                                with open(tmp_test.name, "rb") as f:
                                    st.download_button(
                                        label="Download test/prediction set (.csv.gz)",
                                        data=f,
                                        file_name="test_dataset.csv.gz",
                                        mime="application/gzip",
                                        use_container_width=True,
                                    )
                    else:
                        with open(tmp.name, "rb") as f:
                            st.download_button(
                                label="Download training set (.csv.gz)",
                                data=f,
                                file_name="train_dataset.csv.gz",
                                mime="application/gzip",
                                use_container_width=True
                            )
                    
                    st.success("Datasets compressed successfully!")

        st.markdown("**Descriptors selected**", help="Descriptors selected as the most suitable for the training dataset")

        if "model" in st.session_state:
            df_descriptors = st.session_state["model"]["descriptors"]
        else:
            path_descriptors = os.path.join(st.session_state["job_path"], "best_descriptors/selected_descriptors.csv")
            df_descriptors = pd.read_csv(path_descriptors)

        # Replace values
        pd.set_option('future.no_silent_downcasting', True)
        df_descriptors = df_descriptors.replace({1: True, 0: False})

        # Show in Streamlit
        st.dataframe(df_descriptors.sort_index(axis=1), hide_index=True)
        
        with st.expander("**Descriptors information**"):
            if data_type == "DNA/RNA":
                st.markdown(
                    """**DNC**: Dinucleotide composition;  \n"""
                    """**Fickett**: Fickett score based on positional nucleotide features;  \n"""
                    """**FourierBinary**: Binary numerical mapping using Fourier transform;  \n"""
                    """**FourierComplex**: Complex numerical mapping using Fourier transform;  \n"""
                    """**NAC**: Nucleotide composition;  \n"""
                    """**ORF**: Open reading frame-based features;  \n"""
                    """**Shannon**: Shannon's entropy from 1-mer to 5-mer;  \n"""
                    """**TNC**: Trinucleotide composition;  \n"""
                    """**Tsallis**: Tsallis entropy from 1-mer to 5-mer with q = 2.3;  \n"""
                    """**kGap_di**: Xmer k-Spaced Ymer composition frequency with 2 after 1-gap;  \n"""
                    """**kGap_tri**: Xmer k-Spaced Ymer composition frequency with 3 after 1-gap;  \n"""
                    """**repDNA**: Comprehensive representation of DNA sequences including k-mers, autocorrelations, and physicochemical features;  \n"""
                )
            elif data_type == "Protein":
                st.markdown(
                    """**AAC**: Amino acid composition;  \n"""
                    """**CKSAAGP**: Composition of k-spaced amino acid group pairs;  \n"""
                    """**CKSAAP**: Composition of k-spaced amino acid pairs;  \n"""
                    """**CTDC**: Composition;  \n"""
                    """**CTDD**: Distribution;  \n"""
                    """**CTDT**: Transition;  \n"""
                    """**CTriad**: Conjoint triad;  \n"""
                    """**ComplexNetworks**: Complex network features from 1-mer to 5-mer;  \n"""
                    """**DDE**: Dipeptide deviation from expected mean;  \n"""
                    """**DPC**: Kmer dipeptides composition;  \n"""
                    """**Fourier_EIIP**: Electron-ion interaction potential numerical mapping using Fourier transform;  \n"""
                    """**Fourier_Integer**: Integer numerical mapping using Fourier transform;  \n"""
                    """**GAAC**: Grouped amino acid composition;  \n"""
                    """**GDPC**: Grouped dipeptide composition;  \n"""
                    """**GTPC**: Grouped tripeptide composition;  \n"""
                    """**Global**: Global one-dimensional peptide descriptors calculated from the AA sequence;  \n"""
                    """**KSCTriad**: Conjoint k-spaced Triad;  \n"""
                    """**Peptide**: AA scale based global or convoluted descriptors (auto-/cross-correlated);  \n"""
                    """**Shannon**: Shannon's entropy from 1-mer to 5-mer;  \n"""
                    """**Tsallis_23**: Tsallis's entropy from 1-mer to 5-mer with q = 2.3;  \n"""
                    """**Tsallis_30**: Tsallis's entropy from 1-mer to 5-mer with q = 3.0;  \n"""
                    """**Tsallis_40**: Tsallis's entropy from 1-mer to 5-mer with q = 4.0;  \n"""
                    """**kGap_di**: Xmer k-Spaced Ymer composition frequency with 1 after 1-gap;  \n"""
                )

def derive_key_from_password(password: str, salt: bytes, iterations: int = 390000) -> bytes:
    password_bytes = password.encode("utf-8")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
    
    return key

def decrypt_job_archive(job_path: str, password: str, target_extract_path: str) -> bool:
    enc_path = os.path.join(job_path, "job_archive.enc")
    salt_path = os.path.join(job_path, "job_salt.bin")

    with open(salt_path, "rb") as f:
        salt = f.read()

    key = derive_key_from_password(password, salt)
    fernet = Fernet(key)

    with open(enc_path, "rb") as f:
        encrypted = f.read()
    try:
        decrypted = fernet.decrypt(encrypted)
    except Exception:
        # Bad password or corrupted archive
        return False

    # Write tar bytes to memory and extract
    buf = io.BytesIO(decrypted)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        # Ensure target directory exists
        os.makedirs(target_extract_path, exist_ok=True)
        tar.extractall(path=target_extract_path)

    return True

def runUI():

    with st.expander("Viewing your submission"):
        st.info(
            """
            The **Jobs module** provides access to the results of training and prediction jobs.

            **Accessing results:**
            * **Home:** Training a model from scratch generates a *Job ID*, which can be used here to track progress and view results.
            * **Model Repository:** Running predictions with a trained model also generates a *Job ID*, allowing access to prediction outputs.

            Results are organized into interactive tabs, including model details, performance metrics, predictions, feature importance, and exploratory analyses.
            
            **Note that models from the repository with more than 5,000 training or testing/prediction sequences may have limited visualizations.**
	    """
        )

    # Uncomment to show queue
    # jobcol1, jobcol2 = st.columns(2)

    # with jobcol1:
    #     st.markdown("**Pending jobs**", help="Table displaying the first five pending jobs.")
    #     df = manager.get_pending_jobs()
    #     column_config = {
    #         "Queue position": st.column_config.TextColumn(
    #             "Queue position",
    #             help="Position in the queue for each job"
    #         ),
    #         "Job ID": st.column_config.TextColumn(
    #             "Job ID",
    #             help="Unique identifier for the job run"
    #         ),
    #         "Start": st.column_config.DateColumn(
    #             "Start",
    #             format="h:mm A. MMMM D, YYYY",
    #             help="Time when started"
    #         ),
    #         "Status": st.column_config.TextColumn(
    #             "Status",
    #             help="Job status: pending or running"
    #         )
    #     }
    #     st.dataframe(df, column_config=column_config, use_container_width=True, hide_index=True)

    # with jobcol2:
    #     st.markdown("**Last completed jobs**", help="Table displaying the most recently completed jobs, ordered from newest to oldest.")
    #     df = manager.get_recent_completed_jobs()
    #     column_config = {
    #         "Job ID": st.column_config.TextColumn(
    #             "Job ID",
    #             help="Unique identifier for the job run"
    #         ),
    #         "End": st.column_config.DateColumn(
    #             "End",
    #             format="h:mm A. MMMM D, YYYY",
    #             help="Time when finished"
    #         ),
    #         "Duration": st.column_config.TextColumn(
    #             "Duration",
    #             help="Total execution time (HH:MM:SS)"
    #         ),
    #         "Status": st.column_config.TextColumn(
    #             "Status",
    #             help="Job outcome: success or failed"
    #         )
    #     }
    #     st.dataframe(df, column_config=column_config, use_container_width=True, hide_index=True)

    def get_job_example():
        st.session_state["job_input"] = "867473e7-0dbc-4e42-81fd-985b7d1f7e64"

    with st.container(border=True):
        col1, col2 = st.columns([9, 1])

        with col2:
            st.button("Example", use_container_width=True, on_click=get_job_example)

        with st.form("jobs_submit", border=False):

            textcol1, textcol2 = st.columns(2)

            with textcol1:
                job_id = st.text_input("Enter Job ID", key="job_input")

            with textcol2:
                password = st.text_input("Password to decrypt submission (if encrypted)", type='password', help="If submission was encrypted, provide the password for decryption.")

            submitted = st.form_submit_button("Submit", use_container_width=True, type="primary")

    predict_path, dataset_path = "jobs", "datasets"

    if submitted:
        if job_id:
            job_path = ""
            if os.path.exists(os.path.join(predict_path, job_id)):
                job_path = os.path.join(predict_path, job_id)

            job = manager.get_result(job_id)

            if job:
                if job["status"] == "success":
                    _cleanup_previous_temp()

                    enc_path = os.path.join(job_path, "job_archive.enc")
                    salt_path = os.path.join(job_path, "job_salt.bin")

                    if os.path.exists(enc_path) and os.path.exists(salt_path):
                        if password:
                            temp_dir = tempfile.mkdtemp(prefix=f"extracted_job_{job_id}_")
                            succeeded = decrypt_job_archive(job_path, password, temp_dir)

                            if succeeded:
                                st.session_state["job_path"] = temp_dir
                            else:
                                st.error(f"Wrong password for descryption.")
                                if "job_path" in st.session_state:
                                    del st.session_state["job_path"]
                        else:
                            st.error("Please provide a password for decryption.")
                            if "job_path" in st.session_state:
                                del st.session_state["job_path"]
                    else:
                        if os.path.exists(os.path.join(predict_path, job_id)):
                            job_path = os.path.join(predict_path, job_id)
                            st.session_state["job_path"] = job_path
                        else:
                            if "job_path" in st.session_state:
                                del st.session_state["job_path"]
                            st.error("Job does not exist!")

                elif job["status"] == "running" or job["status"] == "pending":
                    if "job_path" in st.session_state:
                        del st.session_state["job_path"]
                    st.info(f"Job is position #{manager.get_job_position(job_id)} in the queue. Come back later.")

                elif job["status"] == "failure":
                    if "job_path" in st.session_state:
                        del st.session_state["job_path"]
                    st.info("Job failed. Try again.")
            else:
                job_path = os.path.join(dataset_path, job_id,  "runs", "run_6")

                if os.path.exists(job_path):
                    st.session_state["job_path"] = job_path
                else:
                    if "job_path" in st.session_state:
                        del st.session_state["job_path"]
                    st.error("Job does not exist!")

    if "job_path" in st.session_state:
        st.success("Job was completed with the following results")

        if "model" in st.session_state:
            del st.session_state["model"]

        if "reducer" in st.session_state:
            del st.session_state["reducer"]

        if "mapper" in st.session_state:
            del st.session_state["mapper"]

        path_model = os.path.join(st.session_state["job_path"], "trained_model.sav")
        
        if os.path.exists(path_model):
            if "model" not in st.session_state:
                with st.spinner("Loading trained model..."):
                    st.session_state["model"] = joblib.load(path_model)

                train_stats = st.session_state["model"]["train_stats"]
        else:
            train_stats = pd.read_csv(os.path.join(st.session_state["job_path"], "train_stats.csv"))

        if "label_encoder" in st.session_state["model"]:
            task = "Classification"
        else:
            task = "Regression"

        data_type = "Structured data"

        if "descriptors" in st.session_state["model"]:
            df_descriptors = st.session_state["model"]["descriptors"]

            if "NAC" in df_descriptors.columns:
                data_type = "DNA/RNA"
            else:
                data_type = "Protein"

        if "mapper" not in st.session_state:
            if data_type == "DNA/RNA":
                st.session_state["mapper"] = joblib.load("dict_nt.pkl")
            else:
                st.session_state["mapper"] = joblib.load("dict_aa.pkl")

        df_job_info = pl.read_csv(os.path.join(st.session_state["job_path"], "job_info.tsv"), separator='\t')

        with st.expander("Summary Statistics"):
            st.info(
            """
            This section summarizes the **basic characteristics of your dataset**.

            Depending on the data type, it reports the number of samples or sequences, their length 
            distribution, and relevant composition statistics (such as GC content). Results are shown 
            separately for the **training set** and, when available, the **test/prediction set**.

            These statistics help assess data quality and provide essential context for interpreting the 
            downstream analyses.
            """
            )

            str_type = {
                "DNA/RNA": ["<br><strong>gc_content</strong>: Average GC% content considering all sequences;", "Nucleotide"],
                "Protein": ["", "Amino acid"],
                "Structured data": ["<br><strong>num_samples</strong>: Number of samples", "Structured data"]
            }

            tooltip_text = """
            <strong>num_seqs</strong>: Number of sequences;<br>
            <strong>min_len</strong>: Minimum length of sequences;<br>
            <strong>max_len</strong>: Maximum length of sequences;<br>
            <strong>avg_len</strong>: Average length of sequences;<br>
            <strong>std_len</strong>: Standard deviation for length of sequences;<br>
            <strong>sum_len</strong>: Sum of length of all sequences;<br>
            <strong>Q1</strong>: 25th percentile for length of sequences;<br>
            <strong>Q2</strong>: 50th percentile for length of sequences;<br>
            <strong>Q3</strong>: 75th percentile for length of sequences;<br>
            <strong>N50</strong>: Length of the shortest read in the group of 
            longest sequences that together represent (at least) 50% of the 
            characters in the set of sequences;
            """

            if data_type == "Structured data":
                tooltip_text = "<strong>num_samples</strong>: Number of samples;"
            else:
                tooltip_text += str_type[data_type][0]

            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end">
                <div class="tooltip"> 
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#66676e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon">
                        <circle cx="12" cy="12" r="10"></circle>
                        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
                        <line x1="12" y1="17" x2="12.01" y2="17"></line>
                    </svg>
                    <span class="tooltiptext">
                        {tooltip_text}
            """, unsafe_allow_html=True)
                
            st.markdown("**Training set**")
        
            train_stats_formatted = train_stats.style.format(thousands=",")
            st.dataframe(train_stats_formatted, hide_index=True, use_container_width=True)

            if df_job_info["testing_set"].item() != "No test set":
                st.markdown("**Test/Prediction set**")
                test_stats = pd.read_csv(os.path.join(st.session_state["job_path"], "test_stats.csv"))
                test_stats_formatted = test_stats.style.format(thousands=",")
                st.dataframe(test_stats_formatted, hide_index=True, use_container_width=True)

        tabs = {}

        if df_job_info["testing_set"].item() != "No test set":
            if sum(train_stats["num_seqs"].to_list()) > 5_000 or sum(test_stats["num_seqs"].to_list()) > 5_000:
                tab_list = ["Model Information", "Performance Metrics", "Predictions",
                            "Feature Importance"]
            else:
                tab_list = ["Model Information", "Performance Metrics", "Predictions",
                            "Feature Importance", "Feature Distribution",
                            "Feature Correlation", "Dimensionality Reduction"]
        else:
            if sum(train_stats["num_seqs"].to_list()) > 5_000:
                tab_list = ["Model Information", "Performance Metrics",
                            "Feature Importance"]
            else:
                tab_list = ["Model Information", "Performance Metrics",
                            "Feature Importance", "Feature Distribution",
                            "Feature Correlation", "Dimensionality Reduction"]

        # Create the tabs dynamically
        streamlit_tabs = st.tabs(tab_list)

        # Map tab names to Streamlit tab objects
        tabs = {name: tab for name, tab in zip(tab_list, streamlit_tabs)}

        with tabs["Model Information"]:
            model_information(data_type, task)

        with tabs["Performance Metrics"]:
            performance_metrics(task)

        if "Predictions" in tabs:
            with tabs["Predictions"]:
                show_predictions()

        with tabs["Feature Importance"]:
            feature_importance()

        if "Feature Distribution" in tabs:
            with tabs["Feature Distribution"]:
                feature_distribution()

        if "Feature Correlation" in tabs:
            with tabs["Feature Correlation"]:
                feature_correlation()

        if "Dimensionality Reduction" in tabs:
            with tabs["Dimensionality Reduction"]:
                dimensionality_reduction()