
import streamlit as st
import requests
import json
import base64
import pandas as pd

def get_base64_of_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

background_path = "./background.png"
gif_path = "./srdfgen.gif"

base64_background = get_base64_of_file(background_path)
encoded_gif = get_base64_of_file(gif_path)

st.markdown(f"""
<style>
/* Overall font */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* App background */
.stApp {{
    background: url("data:image/png;base64,{base64_background}");
    background-size: 1000px;
    background-position: center;
    background-attachment: fixed;
}}

/* Top GIF wrapper */
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

/* Title (st.title) */
.stTitle h1 {{
    font-family: 'Poppins', sans-serif !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

/* Header (st.header) */
.stHeader h2 {{
    font-family: 'Poppins', sans-serif !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    background: linear-gradient(90deg, #ff6b6b, #ff4b4b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

/* Subheader (st.subheader) */
.stSubheader h3 {{
    font-family: 'Roboto', sans-serif !important;
    font-size: 18px !important;
    font-weight: 500 !important;
    background: linear-gradient(90deg, #ff7f7f, #ff4b4b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

/* Button styling */
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


import streamlit as st
import requests
import json
from sseclient import SSEClient  # pip install sseclient-py

API_BASE = "http://fastapi-backend:8000"

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
                    prop["distribution_params"]["mean"] = st.number_input(
                        f"Mean", value=float(mean), key=f"mean_{idx}"
                    )
                with col2:
                    prop["distribution_params"]["std"] = st.number_input(
                        f"Std", value=float(std), key=f"std_{idx}"
                    )
                with col3:
                    prop["distribution_params"]["min"] = st.number_input(
                        f"Min", value=float(min_val), key=f"min_{idx}"
                    )
                with col4:
                    prop["distribution_params"]["max"] = st.number_input(
                        f"Max", value=float(max_val), key=f"max_{idx}"
                    )


if parsed_schema and st.button("Generate RDF Samples (Real-Time)"):

    st.markdown("""
    <h3 style="font-family: 'Roboto', sans-serif; font-size: 18px; font-weight: 500; 
                background: linear-gradient(90deg, #ff7f7f, #ff4b4b);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;">
        Generating RDF...)
    </h3>
    """, unsafe_allow_html=True)

    for idx, prop in enumerate(parsed_schema):
        if prop["distribution_type"] == "categorical":
            allowed_str = st.session_state.get(f"allowed_{idx}", "")
            prob_str = st.session_state.get(f"prob_{idx}", "")
            prop["distribution_params"]["allowed_list"] = [
                v.strip() for v in allowed_str.split(",") if v.strip()
            ]
            if prob_str:
                prop["distribution_params"]["probabilities"] = [
                    v.strip() for v in prob_str.split(",") if v.strip()
                ]

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

    st.markdown("""
    <h3 style="font-family: 'Roboto', sans-serif; font-size: 18px; font-weight: 500; 
                background: linear-gradient(90deg, #ff7f7f, #ff4b4b);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;">
        Generated RDF Samples (Turtle)
    </h3>
    """, unsafe_allow_html=True)

    for idx, rdf in enumerate(rdf_list):
        st.text_area(f"Sample {idx+1}", rdf, height=150)


    st.markdown("""
    <h3 style="font-family: 'Roboto', sans-serif; font-size: 18px; font-weight: 500; 
                background: linear-gradient(90deg, #ff7f7f, #ff4b4b);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;">
        Generated Data Samples (JSON-LD)
    </h3>
    """, unsafe_allow_html=True)
    st.json(data_list)

    # Download buttons
    ttl_bytes = "\n\n".join(rdf_list).encode("utf-8")
    st.download_button(
        label="Download TTL File",
        data=ttl_bytes,
        file_name="generated_samples.ttl",
        mime="text/turtle"
    )

    jsonld_bytes = json.dumps(data_list, indent=2).encode("utf-8")
    st.download_button(
        label="Download JSON-LD File",
        data=jsonld_bytes,
        file_name="generated_samples.jsonld",
        mime="application/ld+json"
    )