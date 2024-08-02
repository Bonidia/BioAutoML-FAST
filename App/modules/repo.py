import streamlit as st

def runUI():
    model = st.selectbox("Select trained model", [
                'bitter peptides',
                'LncRNA Subcellular Localization ',
                'lncRNA',
                'anti-coronavirus peptides',
                'proinflammatory peptides',
                'Identification of Bacteriophage-Host Interaction',
                'DNA N4-methylcytosine sites',
                'Identifying Antioxidant Proteins',
                'N4-methylcytosine site prediction',
                'eukaryotic sequences',
                'Protein - Pvp-svm',
                'lncRNA vs. sncRNA',
                'antiviral and anti-coronavirus',
                'circRNA',
                'Recombination Spots',
                'RNA pseudouridine sites',
                'pyfeat - Datasets',
                'viral sequences',
                'Prediction of Anticancer Peptides',
                'DNA N 6 -Methyladenine',
                'COVID-19',
                'antibacterial peptides',
                'Identifying enhancers',
                'Neuropeptides',
                'therapeutic peptides targeting SARS-CoV-2',
                'Anticancer Peptides',
                'iDNA-MS',
                'Design powerful predictor for mRNA subcellular location prediction in Homo sapiens',
                'PLRPIM-master',
                'DNA-Binding Proteins',
                'phage virion proteins',
                'neuropeptides',
                'Recombination Hotspot',
                'mAML_ an automated machine learning pipeline with a microbiome repository for human disease classification',
                'CircRNA vs. lncRNA',
                'Prediction of Protein Pupylation Sites',
                'Sequence-Based Prediction of Type IV Secreted',
                'Therapeutic Peptides',
                'lncRNA-protein Interaction',
                'cancerous genomic sequences',
                'Cancer',
                'localization of mRNAs'])
    
    if model == "bitter peptides":
        st.info("""
                **Data set from the following paper:** Charoenkwan, P., Nantasenamat, C., Hasan, M. M., Manavalan, B., & Shoombuatong, W. 
                (2021). BERT4Bitter: a bidirectional encoder representations from transformers 
                (BERT)-based model for improving the prediction of bitter peptides. Bioinformatics, 
                37(17), 2556-2562.
                """)
    else:
        st.info("test")

    with st.form("repo_submit", clear_on_submit=True):

        test_files = st.file_uploader("FASTA file for prediction", accept_multiple_files=False, help="Single file for prediction (e.g. predict.fasta)")

        submitted = st.form_submit_button("Submit", use_container_width=True, type="primary")
