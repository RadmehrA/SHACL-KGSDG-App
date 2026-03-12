import streamlit as st
import requests
import json
import base64
import pandas as pd
from sseclient import SSEClient
from rdflib import Graph, Namespace
import pandas as pd
import networkx as nx

def get_base64_of_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    
def fetch_models(model_type):
    try:
        if model_type == "GAN":
            resp = requests.get(f"{API_BASE}/gan_models")
        elif model_type == "VAE":
            resp = requests.get(f"{API_BASE}/vae_models")
        else:
            return []
        
        if resp.status_code == 200:
            return resp.json().get("saved_models", [])
        return []
    except Exception as e:
        st.error(f"Failed to fetch {model_type} models: {e}")
        return []

def render_custom_progress_bar(progress):
    percentage = int(progress)
    bar = f"""
    <div style="background-color: #e0e0e0; border-radius: 8px; height: 24px; width: 100%; margin-top: 20px;">
        <div style="
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            width: {percentage}%;
            height: 100%;
            border-radius: 8px;
            text-align: center;
            color: white;
            font-weight: bold;
            line-height: 24px;">
            {percentage}%
        </div>
    </div>
    """
    st.markdown(bar, unsafe_allow_html=True)

API_BASE = "http://fastapi-backend:8000"
background_path = "./background.png"
gif_path = "./srdfgen.gif"

base64_background = get_base64_of_file(background_path)
encoded_gif = get_base64_of_file(gif_path)

st.markdown(f"""
<style>
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: url("data:image/png;base64,{base64_background}");
    background-size: 1000px;
    background-position: center;
    background-attachment: fixed;
}}

.top-wrapper {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin-top: 30px;
    margin-bottom: 40px;
    text-align: center;
}}
.top-wrapper img {{ 
    width: 280px; 
    max-width: 90%; 
}}
.top-wrapper h1 {{
    font-family: 'Poppins', sans-serif !important;
    font-size: 28px !important;     
    font-weight: 700 !important;
    margin-top: 15px;
    color: #1f2937;
}}

.stButton>button {{
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    border-radius: 12px;
    font-size: 15px;
    padding: 10px 16px;
    background: linear-gradient(135deg, #ff4b4b, #ff6b6b);
    color: white;
    border: none;
}}
</style>

<div class="top-wrapper">
    <img src="data:image/gif;base64,{encoded_gif}" />
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["SHACL to RDF Generator", "RDF Explorer", "Ontology Trainer"])

with tab1:
    st.markdown("""
    <h1 style="font-family: 'Poppins', sans-serif; font-size: 28px; font-weight: 700; 
                background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;">
        SHACL to RDF Generator
    </h1>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload SHACL (.ttl)", type=["ttl"])
    parsed_schema = None

    if uploaded_file:
        with st.spinner("Uploading and parsing SHACL..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{API_BASE}/upload_shacl_and_extract_schema", files=files)
            if response.status_code == 200:
                parsed_schema = response.json()["json_schema"]
                st.success("SHACL parsed successfully!")

    if parsed_schema:
        st.markdown("""
        <h2 style="font-family: 'Poppins', sans-serif; font-size: 22px; font-weight: 600; 
                    background: linear-gradient(90deg, #ff6b6b, #ff4b4b);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;">
            Configure Generation Parameters
        </h2>
        """, unsafe_allow_html=True)

        num_samples = st.number_input("Number of samples", min_value=1, value=3)

        for idx, prop in enumerate(parsed_schema):
            with st.expander(f"Property: {prop['path'].split('#')[-1]}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    prop["model_type"] = st.selectbox(
                        "Model Type", ["LLM", "VAE", "GAN"], key=f"model_type_{idx}"
                    )
                with col2:
                    default_name = "" if prop["model_type"] == "LLM" else prop.get("model_name", "")
                    

                    if prop["model_type"] in ["VAE", "GAN"]:
                        model_options = fetch_models(prop["model_type"])
                        prop["model_name"] = st.selectbox(
                            "Model Name",
                            options=model_options,
                            index=0 if model_options else -1,
                            key=f"model_name_{idx}"
                        )
                    else:
                        prop["model_name"] = st.text_input(
                            "Model Name", value=default_name, key=f"model_name_{idx}"
                        )

                if prop["distribution_type"] == "categorical":
                    allowed_list = prop["distribution_params"].get("allowed_list", [])
                    probabilities = prop["distribution_params"].get("probabilities", [])
                    st.text_area(f"Allowed Values", value=", ".join(allowed_list), key=f"allowed_{idx}")
                    st.text_area(f"Probabilities", value=", ".join(probabilities), key=f"prob_{idx}")
                elif prop["distribution_type"] == "numeric":
                    mean = prop["distribution_params"].get("mean", 0)
                    std = prop["distribution_params"].get("std", 1)
                    min_val = prop["distribution_params"].get("min", 0)
                    max_val = prop["distribution_params"].get("max", 10)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        prop["distribution_params"]["mean"] = st.number_input(f"Mean", value=float(mean), key=f"mean_{idx}")
                    with col2:
                        prop["distribution_params"]["std"] = st.number_input(f"Std", value=float(std), key=f"std_{idx}")
                    with col3:
                        prop["distribution_params"]["min"] = st.number_input(f"Min", value=float(min_val), key=f"min_{idx}")
                    with col4:
                        prop["distribution_params"]["max"] = st.number_input(f"Max", value=float(max_val), key=f"max_{idx}")

    
    if parsed_schema and st.button("Generate RDF Samples (Real-Time)"):
        st.markdown("""
        <h3 style="font-family: 'Roboto', sans-serif; font-size: 18px; font-weight: 500; 
                    background: linear-gradient(90deg, #ff7f7f, #ff4b4b);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;">
            Generating RDF...
        </h3>
        """, unsafe_allow_html=True)

        
        for idx, prop in enumerate(parsed_schema):
            if prop["distribution_type"] == "categorical":
                allowed_str = st.session_state.get(f"allowed_{idx}", "")
                prob_str = st.session_state.get(f"prob_{idx}", "")
                prop["distribution_params"]["allowed_list"] = [v.strip() for v in allowed_str.split(",") if v.strip()]
                if prob_str:
                    prop["distribution_params"]["probabilities"] = [v.strip() for v in prob_str.split(",") if v.strip()]

        url = f"{API_BASE}/generate_from_shacl_stream?num_samples={num_samples}"
        response = requests.post(url, json=parsed_schema, stream=True)
        client = SSEClient(response)

        progress_bar = st.progress(0)
        generated_rdf = []
        generated_data = []

        for event in client.events():
            data = json.loads(event.data)
            progress = data.get("progress", 0)
            progress_bar.progress(progress)

            if "rdf_turtle_samples" in data:
                generated_rdf = data["rdf_turtle_samples"]
                generated_data = data["generated_data_samples"]

        if generated_rdf and generated_data:
            st.session_state['generated_rdf'] = generated_rdf
            st.session_state['generated_data'] = generated_data
            st.success("RDF generation completed!")

    
    if 'generated_rdf' in st.session_state and 'generated_data' in st.session_state:
        rdf_list = st.session_state['generated_rdf']
        data_list = st.session_state['generated_data']

        st.markdown("<h3>Generated RDF Samples (Turtle)</h3>", unsafe_allow_html=True)
        for idx, rdf in enumerate(rdf_list):
            st.text_area(f"Sample {idx+1}", rdf, height=150)

        st.markdown("<h3>Generated Data Samples (JSON-LD)</h3>", unsafe_allow_html=True)
        st.json(data_list)

        
        st.download_button("Download TTL File", data="\n\n".join(rdf_list).encode("utf-8"),
                           file_name="generated_samples.ttl", mime="text/turtle")
        st.download_button("Download JSON-LD File", data=json.dumps(data_list, indent=2).encode("utf-8"),
                           file_name="generated_samples.jsonld", mime="application/ld+json")

import streamlit as st
from rdflib import Graph
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components

with tab2:
    st.header("Upload RDF Data for Visualization & Query (Interactive)")

    uploaded_rdf = st.file_uploader("Upload RDF (.ttl or .jsonld)", type=["ttl", "jsonld"])

    if uploaded_rdf:
        g = Graph()
        file_type = "ttl" if uploaded_rdf.name.endswith(".ttl") else "json-ld"
        g.parse(data=uploaded_rdf.getvalue(), format=file_type)
        st.success(f"RDF data loaded successfully ({len(g)} triples)")

        
        st.subheader("Namespaces in Graph")
        namespaces = {prefix: ns for prefix, ns in g.namespaces()}
        st.json(namespaces)

        
        default_query = """
PREFIX pr: <http://purl.obolibrary.org/obo/pr#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?protein ?protein_label ?GO ?synonym
WHERE {
  ?protein rdfs:label ?protein_label .

  OPTIONAL { 
    ?protein pr:has_GO_term ?GO .   # Use the correct property for GO annotations
  }

  OPTIONAL {
    ?protein oboInOwl:hasExactSynonym ?synonym .
  }
}
LIMIT 100
"""

        st.subheader("Run SPARQL Query")
        query_input = st.text_area("Enter your SPARQL query", value=default_query, height=250)
        if st.button("Execute Query"):
            if not query_input.strip():
                st.warning("Please enter a SPARQL query first.")
            else:
                try:
                    results = g.query(query_input)
                    df = pd.DataFrame([row.asdict() for row in results])
                    if df.empty:
                        st.info("Query returned no results.")
                    else:
                        st.subheader("Query Results")
                        st.dataframe(df)

                        
                        st.subheader("Interactive Graph Visualization")

                        net = Network(height="600px", width="100%", notebook=False, directed=False)

                        
                        main_col = df.columns[0]
                        for _, row in df.iterrows():
                            main_node = row[main_col]
                            net.add_node(main_node, label=main_node, color='lightblue')
                            for col in df.columns[1:]:
                                value = row[col]
                                if pd.notna(value):
                                    for v in str(value).split(", "):
                                        net.add_node(v, label=v, color='lightgreen')
                                        net.add_edge(main_node, v)

                        
                        net.save_graph("rdf_graph.html")
                        HtmlFile = open("rdf_graph.html", 'r', encoding='utf-8').read()
                        components.html(HtmlFile, height=650)
                except Exception as e:
                    st.error(f"Error executing query: {e}")

import streamlit as st
import requests

with tab3:
    st.header("Upload Ontology & Train Graph Models")

    st.markdown("""
    Upload your ontology file and train a GraphVAE or GraphGAN model.
    Provide a unique model name before training.
    """)

    model_name = st.text_input("Model Name", placeholder="Enter a unique model name")
    uploaded_file = st.file_uploader("Upload Ontology (.owl, .ttl)", type=["owl", "ttl"])

    model_type = st.selectbox("Select Model Type", ["GraphVAE", "GraphGAN"])

    if st.button("Train Model"):
        if not model_name:
            st.warning("Please enter a model name.")
        elif not uploaded_file:
            st.warning("Please upload an ontology file.")
        else:
            with st.spinner(f"Training {model_type} model..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    endpoint = (
                        "http://fastapi-backend:8000/graphvae/upload_and_train"
                        if model_type == "GraphVAE"
                        else "http://fastapi-backend:8000/graphgan/upload_and_train"
                    )
                    params = {"model_name": model_name}
                    response = requests.post(endpoint, params=params, files=files)

                    if response.status_code == 200:
                        st.success(response.json().get("message", "Model trained successfully!"))
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")