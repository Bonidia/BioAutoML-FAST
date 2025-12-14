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
        features = st.session_state["model"]["imputer"].transform(features)

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
    dim_col1, dim_col2 = st.columns(2)

    with dim_col1:
        # Evaluation set selection
        df_job_info = pl.read_csv(os.path.join(st.session_state["job_path"], "job_info.tsv"), separator='\t')
    
        has_test_set = True if df_job_info["testing_set"].item() != "No test set" else False
    
        evaluation = st.selectbox(
            ":mag_right: Evaluation set",
            ["Training set", "Test set"] if has_test_set else ["Training set"],
            key="reduction"
        )

        # Load data with caching
        features, labels, nameseqs = load_reduction_data(st.session_state["job_path"], evaluation)

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

                if "reducer" not in st.session_state:
                    if evaluation == "Training set":
                        st.session_state["reducer"] = reducer.fit(scaled_data)
                        reduced_data = reducer.transform(scaled_data)
                else:
                    reduced_data = st.session_state["reducer"].transform(scaled_data)

                reduced_data = reducer.fit_transform(scaled_data)

                # Create plot with caching
                fig = create_reduction_plot(reduced_data, labels, names, reduction)
                st.plotly_chart(fig, use_container_width=True)

def load_features(job_path, evaluation):
    """Load and cache features based on evaluation set"""
    if evaluation == "Training set":
        if "model" in st.session_state:
            return st.session_state["model"]["train"]
        else:
            return pd.read_csv(os.path.join(job_path, "best_descriptors/best_train.csv")) 
    else:
        if os.path.exists(os.path.join(job_path, "feat_extraction/test_labels.csv")):
            return pd.read_csv(os.path.join(job_path, "feat_extraction/test.csv"))
        else:
            return pd.read_csv(os.path.join(job_path, "best_descriptors/best_test.csv"))

def compute_correlation_matrix(features, method):
    """Compute and cache correlation matrix"""
    return features.corr(method=method.lower())

def get_top_correlations(corr_matrix, top_n=100):
    """Get top correlated feature pairs and return matrix with features involved in top pairs"""
    # Create a copy and set diagonal to NaN to exclude self-correlations
    corr_matrix_copy = corr_matrix.copy()
    np.fill_diagonal(corr_matrix_copy.values, np.nan)
    
    # Get upper triangle only (excluding diagonal and duplicates)
    corr_pairs = corr_matrix_copy.unstack()
    corr_pairs = corr_pairs.dropna()
    
    # Sort by absolute correlation but keep original values
    sorted_pairs = corr_pairs.iloc[corr_pairs.abs().argsort()[::-1]].head(top_n)
    
    # Get all unique features from top pairs
    top_features = set()
    for idx in sorted_pairs.index:
        top_features.add(idx[0])
        top_features.add(idx[1])
    
    top_features = sorted(top_features)
    
    return corr_matrix.loc[top_features, top_features]

def create_correlation_df(corr_matrix):
    """Create formatted correlation dataframe"""
    corr_df = pd.DataFrame(corr_matrix.stack(), columns=['Correlation coefficient'])
    corr_df.reset_index(inplace=True)
    corr_df.columns = ['Feature 1', 'Feature 2', 'Correlation coefficient']
    return corr_df[corr_df['Feature 1'] != corr_df['Feature 2']]\
           .sort_values('Correlation coefficient', ascending=False)\
           .reset_index(drop=True)

def create_correlation_heatmap(corr_matrix):
    """Create and cache correlation heatmap"""
    fig = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu',
        title='Correlation heatmap'
    )
    fig.update_traces(
        hovertemplate='Feature 1 (x-axis): %{x}<br>Feature 2 (y-axis): %{y}<br>Correlation: %{z}<extra></extra>'
    )
    fig.update_layout(height=500, margin=dict(t=30, b=50))
    return fig

def feature_correlation():
    col1, col2 = st.columns(2)

    with col1:
        # Evaluation set selection
        df_job_info = pl.read_csv(os.path.join(st.session_state["job_path"], "job_info.tsv"), separator='\t')
    
        has_test_set = True if df_job_info["testing_set"].item() != "No test set" else False
    
        evaluation = st.selectbox(
            ":mag_right: Evaluation set",
            ["Training set", "Test set"] if has_test_set else ["Training set"],
            key="correlation"
        )

    with col2:
        # Correlation method selection
        correlation_method = st.selectbox(
            'Select correlation method:', 
            ['Pearson', 'Spearman']
        )

    # Load features with caching
    features = load_features(st.session_state["job_path"], evaluation)

    if "imputer" in st.session_state["model"]:
        features = pd.DataFrame(st.session_state["model"]["imputer"].transform(features), columns=features.columns)

    # Compute correlation matrix with caching
    with st.spinner('Computing correlations...'):
        corr_matrix = compute_correlation_matrix(features, correlation_method)
        top_corr_matrix = get_top_correlations(corr_matrix)

    # Display results
    with col1:
        st.markdown("**Correlation between features sorted by the correlation coefficient**")
        correlation_df = create_correlation_df(top_corr_matrix)
        st.dataframe(correlation_df, hide_index=True, use_container_width=True)

    with col2:
        with st.spinner('Generating heatmap...'):
            fig = create_correlation_heatmap(top_corr_matrix)
            st.plotly_chart(fig, use_container_width=True)

def load_training_data(job_path):
    """Load and cache training data"""

    if "model" in st.session_state:
        features = st.session_state["model"]["train"]
        labels = pd.DataFrame(st.session_state["model"]["train_labels"], columns=["label"])
        nameseqs = st.session_state["model"]["nameseq_train"]
    else:
        features = pd.read_csv(os.path.join(job_path, "best_descriptors/best_train.csv"))
        labels = pd.read_csv(os.path.join(job_path, "feat_extraction/flabeltrain.csv"))
        nameseqs = pd.read_csv(os.path.join(job_path, "feat_extraction/fnameseqtrain.csv"))
        
    return features, labels, nameseqs

def load_test_data(job_path):
    """Load and cache test data"""
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
    # Determine evaluation set options
    df_job_info = pl.read_csv(os.path.join(st.session_state["job_path"], "job_info.tsv"), separator='\t')
    
    has_test_set = True if df_job_info["testing_set"].item() != "No test set" else False
    
    evaluation = st.selectbox(
        ":mag_right: Evaluation set",
        ["Training set", "Test set"] if has_test_set else ["Training set"],
        help="Training set evaluated with 10-fold cross-validation",
        key="distribution"
    )

    # Load appropriate dataset with caching
    if evaluation == "Training set":
        features, labels, nameseqs = load_training_data(st.session_state["job_path"])

        if "imputer" in st.session_state["model"]:
            features = pd.DataFrame(st.session_state["model"]["imputer"].transform(features), columns=features.columns)
    else:
        features, labels, nameseqs = load_test_data(st.session_state["job_path"])

        if "imputer" in st.session_state["model"]:
            features = pd.DataFrame(st.session_state["model"]["imputer"].transform(features), columns=features.columns)

    col1, col2 = st.columns(2)

    # Select feature to plot
    with col1:
        selected_feature = st.selectbox("Select a feature", features.columns)
        show_rug = st.checkbox("Show rug plot", value=False, 
                             help="Toggle to show/hide individual data points along the axis")

    # Get unique labels and assign colors
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

def performance_metrics():
    df_job_info = pl.read_csv(os.path.join(st.session_state["job_path"], "job_info.tsv"), separator='\t')

    has_test_set = True if df_job_info["testing_set"].item() == "Test set" else False
    
    evaluation_options = ["Training set", "Test set"] if has_test_set else ["Training set"]
    evaluation = st.selectbox(
        ":mag_right: Evaluation set", 
        evaluation_options,
        help="Training set evaluated with 10-fold cross-validation"
    )

    task = df_job_info["task"].item()

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
                        f"**Accuracy:** {df_cv['ACC'].item()} ± {df_cv['std_ACC'].item()}",
                        f"**Sensitivity:** {df_cv['Sn'].item()} ± {df_cv['std_Sn'].item()}",
                        f"**Specificity:** {df_cv['Sp'].item()} ± {df_cv['std_Sp'].item()}",
                        f"**F1-score:** {df_cv['F1'].item()} ± {df_cv['std_F1'].item()}",
                        f"**MCC:** {df_cv['MCC'].item()} ± {df_cv['std_MCC'].item()}",
                        f"**Balanced accuracy:** {df_cv['balanced_ACC'].item()} ± {df_cv['std_balanced_ACC'].item()}",
                        f"**Kappa:** {df_cv['kappa'].item()} ± {df_cv['std_kappa'].item()}",
                        f"**G-mean:** {df_cv['gmean'].item()} ± {df_cv['std_gmean'].item()}"
                    ])
                else:
                    metrics.extend([
                        f"**Accuracy:** {df_cv['ACC'].item()} ± {df_cv['std_ACC'].item()}",
                        f"**Sensitivity (macro):** {df_cv['Sn'].item()} ± {df_cv['std_Sn'].item()}",
                        f"**Specificity (macro):** {df_cv['Sp'].item()} ± {df_cv['std_Sp'].item()}",
                        f"**F1-score (micro):** {df_cv['F1_micro'].item()} ± {df_cv['std_F1_micro'].item()}",
                        f"**F1-score (macro):** {df_cv['F1_macro'].item()} ± {df_cv['std_F1_macro'].item()}",
                        f"**F1-score (weighted):** {df_cv['F1_weighted'].item()} ± {df_cv['std_F1_weighted'].item()}",
                        f"**MCC:** {df_cv['MCC'].item()} ± {df_cv['std_MCC'].item()}",
                        f"**Kappa:** {df_cv['kappa'].item()} ± {df_cv['std_kappa'].item()}"
                    ])

            elif task == "Regression":
                metrics.extend([
                    f"**Mean Absolute Error:** {df_cv['mean_absolute_error'].item()} ± {df_cv['std_mean_absolute_error'].item()}",
                    f"**Mean Squared Error:** {df_cv['mean_squared_error'].item()} ± {df_cv['std_mean_squared_error'].item()}",
                    f"**Root Mean Squared Error:** {df_cv['root_mean_squared_error'].item()} ± {df_cv['std_root_mean_squared_error'].item()}",
                    f"**R2:** {df_cv['r2'].item()} ± {df_cv['std_r2'].item()}"
                ])
            
            for metric in metrics:
                st.markdown(metric)

        else:
            df_report = pd.read_csv(os.path.join(st.session_state["job_path"], "metrics_test.csv"))
            df_report = df_report.rename(columns={"Unnamed: 0": ""})
            st.dataframe(df_report, hide_index=True, use_container_width=True)

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

def feature_importance():
    # Load data with caching
    df = load_feature_importance(st.session_state["job_path"])

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

def model_information():

    df_job_info = pd.read_csv(os.path.join(st.session_state["job_path"], "job_info.tsv"), sep='\t')

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            cont1, cont2 = st.columns([3, 1])

            with cont1:
                st.markdown("**Model**")

                st.markdown(f"**Task:** {df_job_info['task'].item()}")

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
        st.markdown("**Descriptors selected**", help="Descriptors selected as the most suitable for the training dataset")

        path_descriptors = os.path.join(st.session_state["job_path"], "best_descriptors/selected_descriptors.csv")

        df_descriptors = pd.read_csv(path_descriptors)

        # Replace values
        pd.set_option('future.no_silent_downcasting', True)
        df_descriptors = df_descriptors.replace({1: True, 0: False})

        # Show in Streamlit
        st.dataframe(df_descriptors.sort_index(axis=1), hide_index=True)
        
        data_type = df_job_info["data_type"].item()
        
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

    st.info(
        """
        The **Jobs module** provides access to the results of training and prediction jobs.

        **Accessing results:**
        * **Home:** Training a model from scratch generates a *Job ID*, which can be used here to track progress and view results.
        * **Model Repository:** Running predictions with a trained model also generates a *Job ID*, allowing access to prediction outputs.

        **Job handling:**
        * **Pending / Running:** Displays queue position and progress.
        * **Success:** Loads all results and visualizations.
        * **Failure:** Indicates unsuccessful execution.

        Results are organized into interactive tabs, including model details, performance metrics, predictions, feature importance, and exploratory analyses.
        """
    )

    with st.expander("Job queue"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Pending jobs**", help="Table displaying the first five pending jobs.")
            df = manager.get_pending_jobs()
            st.dataframe(df, hide_index=True)

        with col2:
            st.markdown("**Last completed jobs**", help="Table displaying the most recently completed jobs, ordered from newest to oldest.")
            df = manager.get_recent_completed_jobs()
            st.dataframe(df, hide_index=True)

    def get_job_example():
        st.session_state["job_input"] = "70698bf1-a7f1-4597-bd68-5cd10f161000"

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

            submitted = st.form_submit_button("Submit", use_container_width=True,  type="primary")

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

        df_job_info = pl.read_csv(os.path.join(st.session_state["job_path"], "job_info.tsv"), separator='\t')

        with st.expander("Summary Statistics"):
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

            if df_job_info["data_type"].item() == "Structured data":
                tooltip_text = "<strong>num_samples</strong>: Number of samples;"
            else:
                tooltip_text += str_type[df_job_info["data_type"].item()][0]

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
        
            path_model = os.path.join(st.session_state["job_path"], "trained_model.sav")
            
            if os.path.exists(path_model):
                if "model" not in st.session_state:
                    with st.spinner("Loading trained model..."):
                        st.session_state["model"] = joblib.load(path_model, mmap_mode='r')

                    train_stats = st.session_state["model"]["train_stats"]
            else:
                train_stats = pd.read_csv(os.path.join(st.session_state["job_path"], "train_stats.csv"))

            train_stats_formatted = train_stats.style.format(thousands=",")
            st.dataframe(train_stats_formatted, hide_index=True, use_container_width=True)

            if df_job_info["testing_set"].item() != "No test set":
                st.markdown("**Test set**")
                test_stats = pd.read_csv(os.path.join(st.session_state["job_path"], "test_stats.csv"))
                test_stats_formatted = test_stats.style.format(thousands=",")
                st.dataframe(test_stats_formatted, hide_index=True, use_container_width=True)

        features_train, _, _ = load_training_data(st.session_state["job_path"])

        tabs = {}

        if df_job_info["testing_set"].item() != "No test set":
            if max(train_stats["num_seqs"].to_list()) > 2_000 or max(test_stats["num_seqs"].to_list()) > 2_000:
                tab_list = ["Model Information", "Performance Metrics", "Predictions",
                            "Feature Importance", "Feature Distribution"]
            else:
                tab_list = ["Model Information", "Performance Metrics", "Predictions",
                            "Feature Importance", "Feature Distribution",
                            "Feature Correlation", "Dimensionality Reduction"]
        else:
            if max(train_stats["num_seqs"].to_list()) > 2_000:
                tab_list = ["Model Information", "Performance Metrics",
                            "Feature Importance", "Feature Distribution"]
            else:
                tab_list = ["Model Information", "Performance Metrics",
                            "Feature Importance", "Feature Distribution",
                            "Feature Correlation", "Dimensionality Reduction"]

        # Create the tabs dynamically
        streamlit_tabs = st.tabs(tab_list)

        # Map tab names to Streamlit tab objects
        tabs = {name: tab for name, tab in zip(tab_list, streamlit_tabs)}

        with tabs["Model Information"]:
            model_information()

        with tabs["Performance Metrics"]:
            performance_metrics()

        if "Predictions" in tabs:
            with tabs["Predictions"]:
                show_predictions()

        with tabs["Feature Importance"]:
            feature_importance()

        with tabs["Feature Distribution"]:
            feature_distribution()

        if "Feature Correlation" in tabs:
            with tabs["Feature Correlation"]:
                feature_correlation()

        if "Dimensionality Reduction" in tabs:
            with tabs["Dimensionality Reduction"]:
                dimensionality_reduction()