#!/usr/bin/env nextflow

// nextflow -log /dev/null run main.nf -with-conda  

process setupEnvironmentAndRunApp {
    conda params.envFile

    """
    streamlit run $params.appFile
    """
}

workflow {

    params.envFile = "${projectDir}/BioAutoML-env.yml"
    params.appFile = "${projectDir}/App/app.py"

    setupEnvironmentAndRunApp()
}