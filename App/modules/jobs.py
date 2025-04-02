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
from sklearn import tree
import matplotlib.pyplot as plt

@st.cache_data
def load_reduction_data(job_path, evaluation):
    """Load and cache data for dimensionality reduction"""
    if evaluation == "Training set":
        path_best_train = os.path.join(job_path, "best_descriptors/best_train.csv")
        if os.path.exists(path_best_train):
            features = pd.read_csv(path_best_train)
            labels = pd.read_csv(os.path.join(job_path, "feat_extraction/flabeltrain.csv"))["label"].tolist()
            nameseqs = pd.read_csv(os.path.join(job_path, "feat_extraction/fnameseqtrain.csv"))["nameseq"].tolist()
        else:
            model = joblib.load(os.path.join(job_path, "trained_model.sav"))
            features = model["train"]
            labels = pd.DataFrame(model["train_labels"], columns=["label"])["label"].tolist()
            nameseqs = model["nameseq_train"]["nameseq"].tolist()
    else:
        if os.path.exists(os.path.join(job_path, "feat_extraction/test_labels.csv")):
            features = pd.read_csv(os.path.join(job_path, "feat_extraction/test.csv"))
            labels = pd.read_csv(os.path.join(job_path, "feat_extraction/test_labels.csv"))["label"].tolist()
        else:
            features = pd.read_csv(os.path.join(job_path, "best_descriptors/best_test.csv"))
            labels = pd.read_csv(os.path.join(job_path, "feat_extraction/flabeltest.csv"))["label"].tolist()
        
        nameseqs = pd.read_csv(os.path.join(job_path, "feat_extraction/fnameseqtest.csv"))["nameseq"].tolist()
    
    return features, labels, nameseqs

@st.cache_data
def scale_features(features):
    """Scale features with caching"""
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(features))

@st.cache_data(show_spinner=False)
def compute_reduction(_reducer, scaled_data, reduction_method, reduction_params):
    """Compute dimensionality reduction with caching"""
    return pd.DataFrame(_reducer.fit_transform(scaled_data))

@st.cache_data(show_spinner=False)
def create_reduction_plot(reduced_data, labels, names, reduction_method):
    """Create and cache 3D reduction plot"""
    fig = go.Figure()
    unique_labels = np.unique(labels)
    colors = utils.get_colors(len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = [True if l == label else False for l in labels]
        hover_text = names[names["label"] == label]["nameseq"].tolist()
        
        fig.add_trace(go.Scatter3d(
            x=reduced_data[mask][0],
            y=reduced_data[mask][1],
            z=reduced_data[mask][2],
            mode='markers',
            name=f'{label}',
            marker=dict(color=colors[i], size=2),
            hovertemplate=hover_text,
            hoverlabel=dict(
                font=dict(
                    family="Open Sans, History Sans Pro Light",
                    color="white",
                    size=12
                ),
                bgcolor="black"
            ),
            textposition='top center',
            hoverinfo='text'
        ))
    
    fig.update_layout(
        height=500,
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
        reduction_params = {}
        
        if reduction == "t-Distributed Stochastic Neighbor Embedding (t-SNE)":
            perplexity = st.slider("Perplexity", min_value=5, max_value=50, value=30)
            learning_rate = st.slider("Learning rate", min_value=10, max_value=1000, value=200)
            max_iter = st.slider("Number of iterations", min_value=100, max_value=10000, value=1000)
            reducer = TSNE(n_components=3, perplexity=perplexity, 
                          learning_rate=learning_rate, max_iter=max_iter, n_jobs=-1)
            reduction_params = {"perplexity": perplexity, "learning_rate": learning_rate, "max_iter": max_iter}
            
        elif reduction == "Uniform Manifold Approximation and Projection (UMAP)":
            n_neighbors = st.slider("Number of neighbors", min_value=2, max_value=100, value=15)
            min_dist = st.slider("Minimum distance", min_value=0.0, max_value=1.0, value=0.1)
            reducer = UMAP(n_components=3, n_neighbors=n_neighbors, 
                         min_dist=min_dist, n_jobs=-1)
            reduction_params = {"n_neighbors": n_neighbors, "min_dist": min_dist}
            
        else:
            reducer = PCA(n_components=3)

    with dim_col2:
        if reducer:
            with st.spinner(f'Computing {reduction}...'):
                # Compute reduction with caching
                reduced_data = compute_reduction(
                    reducer, 
                    scaled_data, 
                    reduction, 
                    tuple(reduction_params.items())  # Convert dict to tuple for hashability
                )

                # Create plot with caching
                fig = create_reduction_plot(reduced_data, labels, names, reduction)
                st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def load_features(job_path, evaluation):
    """Load and cache features based on evaluation set"""
    if evaluation == "Training set":
        path_best_train = os.path.join(job_path, "best_descriptors/best_train.csv")
        if os.path.exists(path_best_train):
            return pd.read_csv(path_best_train)
        else:
            model = joblib.load(os.path.join(job_path, "trained_model.sav"))
            return model["train"]
    else:
        if os.path.exists(os.path.join(job_path, "feat_extraction/test_labels.csv")):
            return pd.read_csv(os.path.join(job_path, "feat_extraction/test.csv"))
        else:
            return pd.read_csv(os.path.join(job_path, "best_descriptors/best_test.csv"))

@st.cache_data
def compute_correlation_matrix(features, method):
    """Compute and cache correlation matrix"""
    return features.corr(method=method.lower())

@st.cache_data
def get_top_correlations(corr_matrix, top_n=1000):
    """Get top correlated feature pairs"""
    corr_pairs = corr_matrix.abs().unstack()
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    sorted_pairs = corr_pairs.sort_values(ascending=False).head(top_n)
    
    top_features = np.unique(
        sorted_pairs.index.get_level_values(0).tolist() + 
        sorted_pairs.index.get_level_values(1).tolist()
    )[:top_n]
    
    return corr_matrix.loc[top_features, top_features]

@st.cache_data
def create_correlation_df(corr_matrix):
    """Create formatted correlation dataframe"""
    corr_df = pd.DataFrame(corr_matrix.stack(), columns=['Correlation coefficient'])
    corr_df.reset_index(inplace=True)
    corr_df.columns = ['Feature 1', 'Feature 2', 'Correlation coefficient']
    return corr_df[corr_df['Feature 1'] != corr_df['Feature 2']]\
           .sort_values('Correlation coefficient', ascending=False)\
           .reset_index(drop=True)

@st.cache_data
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

@st.cache_data
def load_training_data(job_path):
    """Load and cache training data"""
    path_best_train = os.path.join(job_path, "best_descriptors/best_train.csv")
    if os.path.exists(path_best_train):
        features = pd.read_csv(path_best_train)
        labels = pd.read_csv(os.path.join(job_path, "feat_extraction/flabeltrain.csv"))
        nameseqs = pd.read_csv(os.path.join(job_path, "feat_extraction/fnameseqtrain.csv"))
    else:
        model = joblib.load(os.path.join(job_path, "trained_model.sav"))
        features = model["train"]
        labels = pd.DataFrame(model["train_labels"], columns=["label"])
        nameseqs = model["nameseq_train"]
    return features, labels, nameseqs

@st.cache_data
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

@st.cache_data(show_spinner=False)
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
    else:
        features, labels, nameseqs = load_test_data(st.session_state["job_path"])

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
                group_names = nameseqs[group_indices]["nameseq"]
                fig_rug_text.append(group_names.tolist())
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

@st.cache_data
def load_data(path):
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.sav'):
        return joblib.load(path)
    return None

@st.cache_data
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

    col1, col2 = st.columns(2)

    with col1:
        if evaluation == "Training set":
            path_kfold = os.path.join(st.session_state["job_path"], "training_kfold(10)_metrics.csv")
            if os.path.exists(path_kfold):
                df_cv = load_data(path_kfold)
            else:
                model_data = load_data(os.path.join(st.session_state["job_path"], "trained_model.sav"))
                df_cv = model_data["cross_validation"]

            metrics = []
            if "F1_micro" not in df_cv.columns:
                metrics.extend([
                    f"**Accuracy:** {df_cv['ACC'].item()} ± {df_cv['std_ACC'].item()}",
                    f"**MCC:** {df_cv['MCC'].item()} ± {df_cv['std_MCC'].item()}",
                    f"**F1-score:** {df_cv['F1'].item()} ± {df_cv['std_F1'].item()}",
                    f"**Balanced accuracy:** {df_cv['balanced_ACC'].item()} ± {df_cv['std_balanced_ACC'].item()}",
                    f"**Kappa:** {df_cv['kappa'].item()} ± {df_cv['std_kappa'].item()}",
                    f"**G-mean:** {df_cv['gmean'].item()} ± {df_cv['std_gmean'].item()}"
                ])
            else: 
                metrics.extend([
                    f"**Accuracy:** {df_cv['ACC'].item()} ± {df_cv['std_ACC'].item()}",
                    f"**MCC:** {df_cv['MCC'].item()} ± {df_cv['std_MCC'].item()}",
                    f"**F1-score (micro avg.):** {df_cv['F1_micro'].item()} ± {df_cv['std_F1_micro'].item()}",
                    f"**F1-score (macro avg.):** {df_cv['F1_macro'].item()} ± {df_cv['std_F1_macro'].item()}",
                    f"**F1-score (weighted avg.):** {df_cv['F1_w'].item()} ± {df_cv['std_F1_w'].item()}",
                    f"**Kappa:** {df_cv['kappa'].item()} ± {df_cv['std_kappa'].item()}"
                ])
            
            for metric in metrics:
                st.markdown(metric)
        else:
            df_report = load_data(os.path.join(st.session_state["job_path"], "metrics_test.csv"))
            df_report = df_report.rename(columns={"Unnamed: 0": ""})

            st.dataframe(df_report, hide_index=True, use_container_width=True)
    
    with col2:
        if evaluation == "Training set":
            path_matrix = os.path.join(st.session_state["job_path"], "training_confusion_matrix.csv")
            if os.path.exists(path_matrix):
                df = load_data(path_matrix)
            else:
                model_data = load_data(os.path.join(st.session_state["job_path"], "trained_model.sav"))
                df = model_data["confusion_matrix"]
        else:
            df = load_data(os.path.join(st.session_state["job_path"], "test_confusion_matrix.csv"))
        
        fig = create_confusion_matrix_figure(df)

        with st.spinner('Loading visualization...'):
            st.plotly_chart(fig, use_container_width=True)

@st.cache_data
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

@st.cache_data
def load_feature_importance(job_path):
    """Load and cache feature importance data"""
    path_feat = os.path.join(job_path, "feature_importance.csv")
    if os.path.exists(path_feat):
        return pd.read_csv(path_feat, sep='\t')
    else:
        model_data = joblib.load(os.path.join(job_path, "trained_model.sav"))
        return model_data["feature_importance"]

@st.cache_data
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
    
    # Create plot with caching
    fig = create_feature_importance_figure(df)
    
    # Display with loading indicator
    st.markdown("**Importance of features regarding model training**", 
               help="It is possible to zoom in to visualize features properly.")
    
    with st.spinner('Rendering feature importance...'):
        st.plotly_chart(fig, use_container_width=True)

def model_information():

    model = joblib.load(os.path.join(st.session_state["job_path"], "trained_model.sav"))

    # st.markdown(model.keys())
    # st.markdown(model["clf"])

    col1, col2 = st.columns([1, 2])

    with col1:
        if "RandomForest" in str(model["clf"]):
            st.image("imgs/models/rf.png", use_container_width=True)
        elif "XGB" in str(model["clf"]):
            st.image("imgs/models/xgboost.png", use_container_width=True)
        elif "LGBM" in str(model["clf"]):
            st.image("imgs/models/lightgbm.png", use_container_width=True)
        elif "CatBoost" in str(model["clf"]):
            st.image("imgs/models/catboost.png", use_container_width=True)

    with col2:
        with st.container(border=True):
            cont1, cont2 = st.columns([3, 1])

            with cont1:
                st.markdown("**Model**")

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

            if "RandomForest" in str(model["clf"]):
                st.markdown("**Classifier:** Random Forest")
                params = model["clf"].get_params()
                st.markdown(f"**Number of estimators:** {params['n_estimators']}")
                st.markdown(f"**Criterion:** {params['criterion']}")
                st.markdown(f"**Max depth:** {params['max_depth']}")
                st.markdown(f"**Max features:** {params['max_features']}")
            elif "XGB" in str(model["clf"]):
                st.markdown("**Classifier:** XGBoost")
                params = model["clf"].get_params()
                st.markdown(f"**Number of estimators:** {params['n_estimators']}")
                st.markdown(f"**Learning rate:** {params['learning_rate']}")
                st.markdown(f"**Max depth:** {params['max_depth']}")
                st.markdown(f"**Gamma:** {params['gamma']}")
                st.markdown(f"**Subsample:** {params['subsample']}")
            elif "LGBM" in str(model["clf"]):
                st.markdown("**Classifier:** LightGBM")
                params = model["clf"].get_params()
                st.markdown(f"**Number of estimators:** {params['n_estimators']}")
                st.markdown(f"**Learning rate:** {params['learning_rate']}")
                st.markdown(f"**Max depth:** {params['max_depth']}")
                st.markdown(f"**Boosting type:** {params['boosting_type']}")
                st.markdown(f"**Subsample:** {params['subsample']}")
            elif "CatBoost" in str(model["clf"]):
                st.markdown("**Classifier:** CatBoost")
                params = model["clf"].get_params()
                st.markdown(f"**Number of estimators:** {params['iterations']}")
                st.markdown(f"**Learning rate:** {params['learning_rate']}")
                st.markdown(f"**Depth:** {params['depth']}")
                st.markdown(f"**L2 leaf regularization:** {params['l2_leaf_reg']}")
                st.markdown(f"**Bagging temperature:** {params['bagging_temperature']}")
                
                # fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
                # tree.plot_tree(model["clf"].estimators_[0],
                #             feature_names = model["train"].columns, 
                #             class_names=model["train_labels"],
                #             filled = True)
                # fig.savefig('rf_individualtree.png')

def runUI():
    def get_job_example():
        st.session_state["job_input"] = "SuKEVriL0frtqHPU"

    with st.container(border=True):
        col1, col2 = st.columns([9, 1])

        with col2:
            example_job = st.button("Example", use_container_width=True, on_click=get_job_example)

        with st.form("jobs_submit", border=False):
            job_id = st.text_input("Enter Job ID", key="job_input")

            submitted = st.form_submit_button("Submit", use_container_width=True,  type="primary")

    predict_path, dataset_path = "jobs", "datasets"

    if submitted:
        if job_id:
            job_path = ""
            if os.path.exists(os.path.join(predict_path, job_id)):
                job_path = os.path.join(predict_path, job_id)
            elif os.path.exists(os.path.join(dataset_path, job_id)):
                job_path = os.path.join(dataset_path, job_id, "runs", "run_1")

            if job_path:
                test_fold = os.path.join(job_path, "test")

                test_set = True if os.path.exists(test_fold) else False

                if test_set:
                    predictions = os.path.join(job_path, "test_predictions.csv")
                else:
                    predictions = os.path.join(job_path, "trained_model.sav")
                
                if os.path.exists(predictions):
                    st.session_state["job_path"] = job_path
                else:
                    if "job_path" in st.session_state:
                        del st.session_state["job_path"]
                    st.info("Job is still in progress. Come back later.")
            else:
                if "job_path" in st.session_state:
                    del st.session_state["job_path"]
                st.error("Job does not exist!")

    if "job_path" in st.session_state:
        st.success("Job was completed with the following results")

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
                    </span>
                </div>
            </div> 
            """, unsafe_allow_html=True)
                
            st.markdown("**Training set**")

            path_stats = os.path.join(st.session_state["job_path"], "train_stats.csv")
            if os.path.exists(path_stats):
                train_stats = pl.read_csv(path_stats)
            else:
                train_stats = joblib.load(os.path.join(st.session_state["job_path"], "trained_model.sav"))["train_stats"]

            st.dataframe(train_stats, hide_index=True, use_container_width=True)

            if df_job_info["testing_set"].item() != "No test set":
                st.markdown("**Test set**")
                test_stats = pl.read_csv(os.path.join(st.session_state["job_path"], "test_stats.csv"))
                st.dataframe(test_stats, hide_index=True, use_container_width=True)

        if df_job_info["testing_set"].item() != "No test set":
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Model Information", "Performance Metrics", "Predictions", "Feature Importance", "Feature Distribution", "Feature Correlation", "Dimensionality Reduction"])
        else:
            tab1, tab2, tab4, tab5, tab6, tab7 = st.tabs(["Model Information", "Performance Metrics", "Feature Importance", "Feature Distribution", "Feature Correlation", "Dimensionality Reduction"])
        
        with tab1:
            model_information()

        with tab2:
            performance_metrics()

        if df_job_info["testing_set"].item() != "No test set":
            with tab3:
                show_predictions()
        
        with tab4:
            feature_importance()

        with tab5:
            feature_distribution()

        with tab6:
            feature_correlation()
        
        with tab7:
            dimensionality_reduction()