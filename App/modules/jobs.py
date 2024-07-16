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
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def dimensionality_reduction():
    dim_col1, dim_col2 = st.columns(2)

    with dim_col1:
        evaluation = st.selectbox(":mag_right: Evaluation set", ["Training set", "Test set"], key="reduction")

        if evaluation == "Training set":
            features = pd.read_csv(os.path.join(st.session_state["job_path"], "best_descriptors/best_train.csv"))
            labels = pd.read_csv(os.path.join(st.session_state["job_path"], "feat_extraction/flabeltrain.csv"))["label"].tolist()
            nameseqs = pd.read_csv(os.path.join(st.session_state["job_path"], "feat_extraction/fnameseqtrain.csv"))["nameseq"].tolist()
        else:
            features = pd.read_csv(os.path.join(st.session_state["job_path"], "best_descriptors/best_test.csv"))
            labels = pd.read_csv(os.path.join(st.session_state["job_path"], "feat_extraction/flabeltest.csv"))["label"].tolist()
            nameseqs = pd.read_csv(os.path.join(st.session_state["job_path"], "feat_extraction/fnameseqtest.csv"))["nameseq"].tolist()
        
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(features))

        reduction = st.selectbox("Select dimensionality reduction technique", 
                                ["Principal Component Analysis (PCA)",
                                "t-Distributed Stochastic Neighbor Embedding (t-SNE)",
                                "Uniform Manifold Approximation and Projection (UMAP)"])
        
        if reduction == "t-Distributed Stochastic Neighbor Embedding (t-SNE)":
            perplexity = st.slider("Perplexity", min_value=5, max_value=50, value=30)
            learning_rate = st.slider("Learning rate", min_value=10, max_value=1000, value=200)
            n_iter = st.slider("Number of iterations", min_value=100, max_value=10000, value=1000)
            reducer = TSNE(n_components=3, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=0)
        elif reduction == "Uniform Manifold Approximation and Projection (UMAP)":
            n_neighbors = st.slider("Number of neighbors", min_value=2, max_value=100, value=15)
            min_dist = st.slider("Minimum distance", min_value=0.0, max_value=1.0, value=0.1)
            reducer = UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, random_state=0)
        else:
            reducer = PCA(n_components=3)

    names = pd.DataFrame(list(zip(labels, nameseqs)), columns=["label", "nameseq"])

    if reduction:
        with dim_col2:
            with st.spinner('Loading...'):
                reduced_data = pd.DataFrame(reducer.fit_transform(scaled_data))
                
                fig = go.Figure()

                for i, label in enumerate(np.unique(labels)):
                    mask = [True if l == label else False for l in labels]

                    fig.add_trace(go.Scatter3d(
                        x=reduced_data[mask][0],
                        y=reduced_data[mask][1],
                        z=reduced_data[mask][2],
                        mode='markers',
                        name=f'{label}',
                        marker=dict(
                            color=utils.get_colors(len(np.unique(labels)))[i], size=3,
                        ),
                        hovertemplate=names[names["label"] == label]["nameseq"].tolist(),
                        hoverlabel = dict(
                                        font=dict(
                                        family="Open Sans, History Sans Pro Light",
                                        color="white",
                                        size=12),
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
                    title=reduction,
                    margin=dict(t=30, b=50)
                )

                st.plotly_chart(fig, use_container_width=True)

def feature_correlation():
    col1, col2 = st.columns(2)

    with col1:
        evaluation = st.selectbox(":mag_right: Evaluation set", ["Training set", "Test set"], key="correlation")

        if evaluation == "Training set":
            features = pd.read_csv(os.path.join(st.session_state["job_path"], "best_descriptors/best_train.csv"))
        else:
            features = pd.read_csv(os.path.join(st.session_state["job_path"], "best_descriptors/best_test.csv"))

    with col2:
        correlation_method = st.selectbox('Select correlation method:', ['Pearson', 'Spearman'])

        if correlation_method == 'Pearson':
            correlation_matrix = features.corr(method='pearson')
        else:
            correlation_matrix = features.corr(method='spearman')
        
    with col1:
        st.markdown("**Correlation between features sorted by the correlation coefficient**")
        correlation_df = pd.DataFrame(correlation_matrix.stack(), columns=['Correlation coefficient'])

        # Reset the index to get the feature pairs as separate columns
        correlation_df.reset_index(inplace=True)

        correlation_df.columns = ['Feature 1', 'Feature 2', 'Correlation coefficient']

        correlation_df = correlation_df.drop(correlation_df[correlation_df['Feature 1'] == correlation_df['Feature 2']].index)

        # Sort the DataFrame by correlation coefficient in descending order
        correlation_df = correlation_df.sort_values('Correlation coefficient', ascending=False).reset_index(drop=True)

        # Display the sorted dataframe
        st.dataframe(correlation_df, hide_index=True, use_container_width=True)

    with col2:
        fig = px.imshow(correlation_matrix, x=correlation_matrix.columns, y=correlation_matrix.columns,
                        color_continuous_scale='RdBu', title='Correlation heatmap')
        
        fig.update_traces(hovertemplate='Feature 1 (x-axis): %{x}<br>Feature 2 (y-axis): %{y}<br>Correlation: %{z}<extra></extra>')

        fig.update_layout(height=500, margin=dict(t=30, b=50))
        st.plotly_chart(fig, use_container_width=True)

def feature_distribution():
    evaluation = st.selectbox(":mag_right: Evaluation set", ["Training set", "Test set"],
                                help="Training set evaluated with 10-fold cross-validation", key="distribution")

    if evaluation == "Training set":
        features = pd.read_csv(os.path.join(st.session_state["job_path"], "best_descriptors/best_train.csv"))
        labels = pd.read_csv(os.path.join(st.session_state["job_path"], "feat_extraction/flabeltrain.csv"))
        nameseqs = pd.read_csv(os.path.join(st.session_state["job_path"], "feat_extraction/fnameseqtrain.csv"))
    else:
        features = pd.read_csv(os.path.join(st.session_state["job_path"], "best_descriptors/best_test.csv"))
        labels = pd.read_csv(os.path.join(st.session_state["job_path"], "feat_extraction/flabeltest.csv"))
        nameseqs = pd.read_csv(os.path.join(st.session_state["job_path"], "feat_extraction/fnameseqtest.csv"))
    
    col1, col2 = st.columns(2)

    # Select feature to plot
    with col1:
        selected_feature = st.selectbox("Select a feature", features.columns)

    # Get unique labels and assign colors
    unique_labels = labels["label"].unique()
    color_map = utils.get_colors(len(unique_labels))[:len(unique_labels)]

    with col2:
        num_bins = st.slider("Number of bins", min_value=5, max_value=50, value=30)

    with st.spinner('Loading...'):
        fig_data = []
        fig_rug_text = []

        feature_data = features[selected_feature].values.astype(float)

        for label in unique_labels:
            
            group_indices = list(chain(*(labels == label).values))
            group_data = feature_data[group_indices]
            fig_data.append(group_data)

            group_names = nameseqs[group_indices]["nameseq"]
            fig_rug_text.append(group_names.tolist())

        bin_edges = np.histogram(fig_data[0], bins=num_bins)[1]

        fig = ff.create_distplot(
            fig_data,
            unique_labels,
            bin_size=bin_edges,
            colors=color_map,
            rug_text=fig_rug_text,
            histnorm="probability density"
        )

        fig.update_layout(
            title=f"Feature distribution for {selected_feature}",
            xaxis_title=selected_feature,
            yaxis_title="Density",
            height=800,
            margin=dict(t=30, b=50)
        )

        st.plotly_chart(fig, use_container_width=True)

def performance_metrics():
    evaluation = st.selectbox(":mag_right: Evaluation set", ["Training set", "Test set"],
                                        help="Training set evaluated with 10-fold cross-validation")

    col1, col2 = st.columns(2)

    with col1:
        if evaluation == "Training set":
            df_cv = pl.read_csv(os.path.join(st.session_state["job_path"], "training_kfold(10)_metrics.csv"))

            st.markdown(f"""**Accuracy:** {df_cv['ACC'].item()} ± {df_cv['std_ACC'].item()}""")
            st.markdown(f"""**MCC:** {df_cv['MCC'].item()} ± {df_cv['std_MCC'].item()}""")
            st.markdown(f"""**F1-score (micro avg.):** {df_cv['F1_micro'].item()} ± {df_cv['std_F1_micro'].item()}""")
            st.markdown(f"""**F1-score (macro avg.):** {df_cv['F1_macro'].item()} ± {df_cv['std_F1_macro'].item()}""")
            st.markdown(f"""**F1-score (weighted avg.):** {df_cv['F1_w'].item()} ± {df_cv['std_F1_w'].item()}""")
            st.markdown(f"""**Kappa:** {df_cv['kappa'].item()} ± {df_cv['std_kappa'].item()}""")
        else:
            df_report = pl.read_csv(os.path.join(st.session_state["job_path"], "metrics_test.csv"))
            st.dataframe(df_report, hide_index=True, use_container_width=True)
    with col2:
        if evaluation == "Training set":
            df = pd.read_csv(os.path.join(st.session_state["job_path"], "training_confusion_matrix.csv"))

            # Extract labels and confusion matrix values
            labels = df.columns[1:-1].tolist()
            
            values = df.iloc[0:-1, 1:-1].values.tolist()

            fig = go.Figure(data=go.Heatmap(
                z=values,
                x=labels,
                y=labels,
                colorscale='Blues'
            ))

            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted label',
                yaxis_title='True label',
                margin=dict(t=30, b=50)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            df = pd.read_csv(os.path.join(st.session_state["job_path"], "test_confusion_matrix.csv"))

            # Extract labels and confusion matrix values
            labels = df.columns[1:-1].tolist()
            
            values = df.iloc[0:-1, 1:-1].values.tolist()

            fig = go.Figure(data=go.Heatmap(
                z=values,
                x=labels,
                y=labels,
                colorscale='Blues'
            ))

            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted label',
                yaxis_title='True label',
                margin=dict(t=30, b=50)
            )

            st.plotly_chart(fig, use_container_width=True)

def show_predictions():
    predictions = pd.read_csv(os.path.join(st.session_state["job_path"], "test_predictions.csv"))
    predictions.iloc[:,1:-1] = predictions.iloc[:,1:-1]*100
    labels = predictions.columns[1:-1]

    st.dataframe(
        predictions,
        hide_index=True,
        height=500,
        column_config = {label: st.column_config.ProgressColumn(
                            help="Label probability",
                            format="%.2f%%",
                            min_value=0,
                            max_value=100
                        ) for label in labels},
        use_container_width=True
    )

def feature_importance():
    df = pd.read_csv(os.path.join(st.session_state["job_path"], "feature_importance.csv"), sep=' ', header=None)

    features = df[2].str.extract(r'\((.*?)\)')[0][::-1]

    score_importances = df[3].str.extract(r'\((.*?)\)')[0].values.astype(float)[::-1]

    fig = go.Figure(data=go.Bar(
        x=features[::-1],
        y=score_importances[::-1],
        marker=dict(color=score_importances[::-1], colorscale='blues'),
        hovertemplate='Feature: %{x}<br>Importance: %{y}<extra></extra>'
    ))

    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Importance",
        margin=dict(t=0, b=50)
    )

    st.markdown("**Importance of features regarding model training**", help="It is possible to zoom in to visualize features properly.")
    st.plotly_chart(fig, use_container_width=True)

def runUI():
    if not st.session_state["queue"]:
        st.session_state["queue"] = True

    def get_job_example():
        st.session_state["job_input"] = "V2B4LkGYfa7gEJtd"

    with st.container(border=True):
        col1, col2 = st.columns([9, 1])

        with col2:
            example_job = st.button("Example", use_container_width=True, on_click=get_job_example)

        with st.form("jobs_submit", border=False):
            job_id = st.text_input("Enter Job ID", key="job_input")

            submitted = st.form_submit_button("Submit", use_container_width=True,  type="primary")

    predict_path = "jobs"

    if submitted:
        if job_id:
            job_path = os.path.join(predict_path, job_id)
            if os.path.exists(job_path):
                test_fold = os.path.join(job_path, "test")

                test_set = True if os.path.exists(test_fold) else False

                if test_set:
                    predictions = os.path.join(job_path, "test_confusion_matrix.csv")
                else:
                    predictions = os.path.join(job_path, "training_confusion_matrix.csv")
                
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

        with st.expander("Summary Statistics"):
            st.markdown("**Training set**")
            train_stats = pl.read_csv(os.path.join(st.session_state["job_path"], "train_stats.csv"))
            st.dataframe(train_stats, hide_index=True, use_container_width=True)

            test_fold = os.path.join(st.session_state["job_path"], "test")
            test_set = True if os.path.exists(test_fold) else False

            if test_set:
                st.markdown("**Test set**")
                test_stats = pl.read_csv(os.path.join(st.session_state["job_path"], "test_stats.csv"))
                st.dataframe(test_stats, hide_index=True, use_container_width=True)

        if test_set:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Performance Metrics", "Predictions", "Feature Importance", "Feature Distribution", "Feature Correlation", "Dimensionality Reduction"])
        else:
            tab1, tab3, tab4, tab5, tab6 = st.tabs(["Performance Metrics", "Feature Importance", "Feature Distribution", "Feature Correlation", "Dimensionality Reduction"])
        
        with tab1:
            performance_metrics()

        if test_set:
            with tab2:
                show_predictions()
        
        with tab3:
            feature_importance()

        with tab4:
            feature_distribution()

        with tab5:
            feature_correlation()
        
        with tab6:
            dimensionality_reduction()