#!/usr/bin/env nextflow

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