
import streamlit as st
import requests
import time
import json
import base64
import rdflib


# Function to encode the image
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Path to your local image
image_path = "background.png"  # Ensure this exists!
base64_image = get_base64_of_image(image_path)


# Read and encode your gif
with open("srdfgen.gif", "rb") as f:
    gif_bytes = f.read()
    encoded_gif = base64.b64encode(gif_bytes).decode()


# Encode the background image
with open("./background.png", "rb") as img_file:
    base64_image = base64.b64encode(img_file.read()).decode()

# Inject custom CSS and top-centered GIF + Title
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Impact&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');
        html, body, [class*="css"] {{
            font-family: 'Poppins', sans-serif;
        }}
        .stApp {{
            background: url("data:image/png;base64,{base64_image}");
            background-size: 1000px;
            background-position: center;
            background-attachment: fixed;
        }}
        .top-center-container {{
            position: absolute;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            text-align: center;
            z-index: 100;
        }}
        .top-center-container img {{
            width: 1100px;
        }}
        .top-center-container h1 {{
            margin-top: 1px;
            white-space: nowrap;
            font-family: 'Impact', sans-serif !important;
            font-size: 30px !important;
            font-weight: normal !important;
            color: #333333;
        }}
        .stButton>button {{
            background-color: #ff4b4b;
            color: white;
            border-radius: 10px;
            font-size: 18px;
            padding: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
            transition: 0.3s;
        }}
        .stButton>button:hover {{
            background-color: #e60000;
            box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.3);
        }}
        .stFileUploader {{
            border: 2px dashed #ff4b4b;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }}
        .subheader-style {{
            text-align: left;
            white-space: nowrap;
            font-size: 36px; /* Adjusted font size */
            font-family: 'Roboto', sans-serif; /* Changed font family */
            font-weight: bold; /* Optional: Makes it bold */
            margin-top: 200px;
            color: #2c3e50; /* Optional: Changed text color */
        }}
    </style>

    <div class='top-center-container'>
        <img src="data:image/gif;base64,{encoded_gif}" />
        <h1>SHACL-based Synthetic RDF Generator</h1>
    </div>
""", unsafe_allow_html=True)


# Sidebar
st.sidebar.header("⚙️ Settings")

# Upload SHACL
shacl_file = st.sidebar.file_uploader(" Upload SHACL File", type=["ttl", "shacl"])
available_properties = []
def render_shacl_tree(shape_map):
    for shape in shape_map:
        # Get the root node (target class)
        target_class = shape.get("target_classes", ["Unknown"])[0].split("/")[-1]
        st.sidebar.markdown(f"### ⊛ {target_class}")
        
        for prop in shape.get("properties", []):
            constraints = {list(c.keys())[0]: list(c.values())[0] for c in prop["constraints"]}
            path = constraints.get("http://www.w3.org/ns/shacl#path", "unknown")
            datatype = constraints.get("http://www.w3.org/ns/shacl#datatype", "unknown")
            
            # Simplify path and datatype display
            prop_name = path.split("/")[-1]
            datatype_name = datatype.split("#")[-1]
            
            st.sidebar.markdown(f"  - `{prop_name}`: *{datatype_name}*")

# Fetch saved models using the FastAPI endpoint
def fetch_saved_models():
    response = requests.get("http://localhost:8000/models/saved")  # Adjust URL if needed
    if response.status_code == 200:
        return response.json().get("saved_models", [])
    else:
        st.error("Failed to fetch models")
        return []

if shacl_file:
    files = {"file": shacl_file.getvalue()}
    response = requests.post("http://fastapi-backend:8000/upload_shacl", files=files)
    if response.status_code == 200:
        st.sidebar.success("✅ SHACL File Uploaded Successfully")
        data = response.json()
        shape_map = data.get("shape_map", [])
        render_shacl_tree(shape_map)

        # Get available properties
        try:
            prop_response = requests.get("http://fastapi-backend:8000/get_shacl_properties")
            if prop_response.status_code == 200:
                available_properties = prop_response.json().get("properties", [])
                st.sidebar.success("✅ Retrieved SHACL properties")
            else:
                st.sidebar.warning("⚠️ Could not fetch properties")
        except Exception as e:
            st.sidebar.error(f"❌ Error fetching SHACL properties: {e}")
    else:
        st.sidebar.error("❌ Error uploading SHACL File")

# Show multiselect for properties if available
selected_properties = st.sidebar.multiselect(
    " Select Properties to Include in Generation",
    options=available_properties,
    default=available_properties
)

st.sidebar.markdown("### Select Model & Distribution per Property")

property_model_map = {}
property_distribution_map = {}
model_options = ["LLM", "GAN", "VAE"]
distribution_options = ["Normal", "Uniform", "Custom"]

gan_saved_models = requests.get("http://fastapi-backend:8000/models/saved/gan").json().get("saved_models", [])
vae_saved_models = requests.get("http://fastapi-backend:8000/models/saved/vae").json().get("saved_models", [])
saved_gan_models = [model for model in gan_saved_models if model.lower().endswith("gan")]
saved_vae_models = [model for model in vae_saved_models if model.lower().endswith("vae")]


for prop in selected_properties:
    path = prop["path"]
    
    model_key = f"model_select_{path}"
    selected_model = st.sidebar.selectbox(f"Model for `{path}`", model_options, key=model_key)
    

    if path not in property_model_map:
        property_model_map[path] = {}

    property_model_map[path]["type"] = selected_model

    # Handle GAN selection
    if selected_model == "GAN" and saved_gan_models:
        gan_model_key = f"gan_model_select_{path}"
        selected_gan_model = st.sidebar.selectbox(
            f"Select saved GAN model for `{path}`",
            ["all"] + saved_gan_models,
            key=gan_model_key
        )
        property_model_map[path]["name"] = selected_gan_model

    # Handle VAE selection
    elif selected_model == "VAE" and saved_vae_models:
        vae_model_key = f"vae_model_select_{path}"
        selected_vae_model = st.sidebar.selectbox(
            f"Select saved VAE model for `{path}`",
            ["string_vae"] + saved_vae_models,
            key=vae_model_key
        )
        property_model_map[path]["name"] = selected_vae_model


    # Select distribution per property
    dist_key = f"distribution_select_{path}"
    selected_distribution = st.sidebar.selectbox(f"Distribution for `{path}`", distribution_options, key=dist_key)
    
    # Add distribution-specific parameters
    dist_params = {}
    if selected_distribution == "Normal":
        dist_params["mean"] = st.sidebar.number_input(f"Mean for `{path}`", value=0.0, key=f"mean_{path}")
        dist_params["stddev"] = st.sidebar.number_input(f"Std Dev for `{path}`", value=1.0, key=f"std_{path}")
    elif selected_distribution == "Uniform":
        dist_params["low"] = st.sidebar.number_input(f"Low for `{path}`", value=0.0, key=f"low_{path}")
        dist_params["high"] = st.sidebar.number_input(f"High for `{path}`", value=1.0, key=f"high_{path}")
    elif selected_distribution == "Skewed":
        dist_params["custom_param"] = st.sidebar.text_input(f"Custom Param for `{path}`", key=f"custom_{path}")
        dist_params["low"] = st.sidebar.number_input(f"Low for `{path}`", value=0.0, key=f"low_{path}")
        dist_params["high"] = st.sidebar.number_input(f"High for `{path}`", value=1.0, key=f"high_{path}")

    # Store the distribution selection and its parameters
    property_distribution_map[path] = {
        "type": selected_distribution,
        "parameters": dist_params
    }


# Distribution selection for all properties
distribution_type = st.sidebar.selectbox("Select Data Distribution", ["Normal", "Uniform", "Skewed"])
num_samples = st.sidebar.number_input("Number of Samples", min_value=1, value=10, step=1)

# Distribution parameters
parameters = {}
if distribution_type == "Normal":
    parameters["mean"] = st.sidebar.number_input("Mean", value=0.0)
    parameters["stddev"] = st.sidebar.number_input("Standard Deviation", value=1.0)
elif distribution_type == "Uniform":
    parameters["low"] = st.sidebar.number_input("Low", value=0.0)
    parameters["high"] = st.sidebar.number_input("High", value=1.0)
elif distribution_type == "skewed":
    parameters["custom_param"] = st.sidebar.text_input("Custom Parameter")
    parameters["low"] = st.sidebar.number_input("Low", value=0.0)
    parameters["high"] = st.sidebar.number_input("High", value=1.0)

# Theme toggle
theme_mode = st.sidebar.radio("Theme", ["Light", "Dark"])
if theme_mode == "Light":
    st.markdown("<style>body { background: white; color: black; }</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body { background: #1e1e1e; color: white; }</style>", unsafe_allow_html=True)

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



def render_custom_progress_bar(progress, placeholder):
    percentage = int(progress)
    if percentage > 100:
        percentage = 100
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
    placeholder.markdown(bar, unsafe_allow_html=True)


from rdflib import Graph, URIRef, Literal, Namespace, RDF

EX = Namespace("http://example.org/")

def convert_to_jsonld(data):
    graph = Graph()
    graph.bind("ex", EX)

    for i, item in enumerate(data):
        subject = URIRef(f"http://example.org/item/{i}")
        graph.add((subject, RDF.type, EX.SyntheticEntity))

        for key, value in item.items():
            predicate = URIRef(f"http://example.org/property/{key}")
            object_ = Literal(value)
            graph.add((subject, predicate, object_))

    return graph.serialize(format="json-ld", indent=2)

def convert_to_ttl(data):
    graph = Graph()
    graph.bind("ex", EX)

    for i, item in enumerate(data):
        subject = URIRef(f"http://example.org/item/{i}")
        graph.add((subject, RDF.type, EX.SyntheticEntity))

        for key, value in item.items():
            predicate = URIRef(f"http://example.org/property/{key}")
            object_ = Literal(value)
            graph.add((subject, predicate, object_))

    return graph.serialize(format="turtle")


from io import BytesIO

def get_serialized_jsonld_bytes(synthetic_data):
    return BytesIO(convert_to_jsonld(synthetic_data).encode("utf-8"))

def get_serialized_ttl_bytes(synthetic_data):
    return BytesIO(convert_to_ttl(synthetic_data).encode("utf-8"))

def get_serialized_json_bytes(synthetic_data):
    return BytesIO(json.dumps(synthetic_data, indent=2).encode("utf-8"))



# Function to handle synthetic data generation
def generate_synthetic_data(progress_placeholder, num_samples, payload):
    try:
        selected_model = payload.get("model_name", ["string_vae"])[0]

        if selected_model == "VAE":
            url = "http://fastapi-backend:8000/generate_vae"
            vae_payload = {
                "model_name": selected_model,
                "subject": payload["subject_input"],
                "predicate": payload["predicate_input"],
                "num_samples": payload["num_samples"],
                "distribution": payload.get("distribution_type", "normal")
            }
            headers = {"accept": "application/json"}  # VAE doesn't stream
            stream_response = False
        else:
            url = "http://fastapi-backend:8000/generate_data"
            vae_payload = payload
            headers = {"Accept": "text/event-stream"}
            stream_response = True

        with requests.post(
            url,
            json=vae_payload,
            headers=headers,
            stream=stream_response
        ) as response:
            if response.status_code == 200:
                synthetic_data = []
                last_progress = 0

                if selected_model == "VAE":
                    synthetic_data = response.json().get("synthetic_data", [])
                else:
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8")
                            if decoded_line.startswith("data: "):
                                event_data = json.loads(decoded_line[6:])
                                if event_data.get("type") == "progress_update":
                                    progress = event_data["progress"]
                                    if abs(progress - last_progress) > 0.01:
                                        last_progress = progress
                                        render_custom_progress_bar(progress, progress_placeholder)
                                        time.sleep(0.1)
                                elif event_data.get("type") == "final_result":
                                    synthetic_data = event_data["synthetic_data"]

                return synthetic_data

            else:
                st.error(f"❌ Backend error: {response.status_code}")
                st.text(response.text)
    except Exception as e:
        st.error(f"❌ Error during data generation: {e}")

# UI Trigger for generation
if "synthetic_data" not in st.session_state:
    st.session_state.synthetic_data = None
user_message = st.sidebar.text_area("💬 Use this Prompt for batch property", placeholder="E.g., Generate realistic company names...")

# Check if the button has been clicked
if st.sidebar.button("⚉ Generate Synthetic Data (Batch Request)"):
    st.markdown("""
        <h2 style="text-align: left; font-family: 'Roboto', sans-serif; font-size: 14px; font-weight: bold; color: #2c3e50; margin-top: 200px;">
            ⚇ Generating Synthetic RDF Data (Batch Mode)...
        </h2>
    """, unsafe_allow_html=True)

    model_name = property_model_map[path].get("name", "string_vae")

    # Extract the last part of the predicate URL (after the last "/")
    predicate_input = path.split('/')[-1]  # Get the part after the last "/"

    payload = {
        "num_samples": num_samples,
        "distribution_type": distribution_type.lower(),
        "parameters": parameters,
        "property_model_map": property_model_map,
        "user_message": user_message,
        "model_name": [model_name],
        "subject_input": prop['shape'].split('/')[-1].replace('Shape', ''),
        "predicate_input": path.split('/')[-1]  # Send only the part after the last '/'
    }

    # Progress placeholder for UI updates during data generation
    progress_placeholder = st.empty()

    # Check if synthetic data is already generated or needs to be generated
    if st.session_state.synthetic_data is None:
        # Generate synthetic data
        st.session_state.synthetic_data = generate_synthetic_data(progress_placeholder, num_samples, payload)

# Retrieve synthetic data from session state
synthetic_data = st.session_state.synthetic_data

if synthetic_data:
    st.success("✅ Synthetic data generated successfully!")
    st.subheader("Generated Data")
    st.json(synthetic_data)

    # Generate downloadable data for buttons
    json_data = get_serialized_json_bytes(synthetic_data)
    jsonld_data = get_serialized_jsonld_bytes(synthetic_data)
    ttl_data = get_serialized_ttl_bytes(synthetic_data)

    # Add download buttons for each format
    st.download_button(
        label="📥 Download Synthetic Data (JSON)",
        data=json_data,
        file_name="synthetic_data_batch.json",
        mime="application/json"
    )

    st.download_button(
        label="📥 Download Synthetic Data (JSON-LD)",
        data=jsonld_data,
        file_name="synthetic_data_batch.jsonld",
        mime="application/ld+json"
    )

    st.download_button(
        label="📥 Download Synthetic Data (TTL)",
        data=ttl_data,
        file_name="synthetic_data_batch.ttl",
        mime="text/turtle"
    )
