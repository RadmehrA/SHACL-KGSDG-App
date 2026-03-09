
# # # import streamlit as st
# # # import requests
# # # import time
# # # import json
# # # import base64
# # # import rdflib


# # # # Function to encode the image
# # # def get_base64_of_image(image_path):
# # #     with open(image_path, "rb") as image_file:
# # #         return base64.b64encode(image_file.read()).decode()

# # # # Path to your local image
# # # image_path = "background.png"  # Ensure this exists!
# # # base64_image = get_base64_of_image(image_path)


# # # # Read and encode your gif
# # # with open("srdfgen.gif", "rb") as f:
# # #     gif_bytes = f.read()
# # #     encoded_gif = base64.b64encode(gif_bytes).decode()


# # # # Encode the background image
# # # with open("./background.png", "rb") as img_file:
# # #     base64_image = base64.b64encode(img_file.read()).decode()


# # # st.markdown(f"""
# # # <style>
# # # @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

# # # /* ---------------- GLOBAL FONT ---------------- */
# # # html, body, [class*="css"] {{
# # #     font-family: 'Inter', sans-serif;
# # #     font-size: 15px;
# # #     letter-spacing: 0.2px;
# # # }}

# # # /* ---------------- BACKGROUND ---------------- */
# # # .stApp {{
# # #     background: url("data:image/png;base64,{base64_image}");
# # #     background-size: 1000px;
# # #     background-position: center;
# # #     background-attachment: fixed;
# # # }}

# # # /* ---------------- TOP CENTER HERO ---------------- */
# # # .top-wrapper {{
# # #     display: flex;
# # #     flex-direction: column;
# # #     align-items: center;
# # #     justify-content: center;
# # #     margin-top: 30px;
# # #     margin-bottom: 40px;
# # #     text-align: center;
# # # }}

# # # .top-wrapper img {{
# # #     width: 280px;      /* smaller GIF size */
# # #     max-width: 90%;
# # #     /* shadow removed */
# # # }}

# # # .top-wrapper h1 {{
# # #     font-family: 'Poppins', sans-serif !important;
# # #     font-size: 32px !important;
# # #     font-weight: 700 !important;
# # #     margin-top: 15px;
# # #     color: #1f2937;
# # # }}

# # # /* ---------------- HEADINGS ---------------- */
# # # h1, h2, h3, .stMarkdown h3 {{
# # #     font-family: 'Poppins', sans-serif !important;
# # #     font-weight: 600 !important;
# # #     color: #111827;
# # # }}

# # # /* ---------------- SIDEBAR HEADERS ---------------- */
# # # section[data-testid="stSidebar"] h1,
# # # section[data-testid="stSidebar"] h2,
# # # section[data-testid="stSidebar"] h3 {{
# # #     font-family: 'Poppins', sans-serif !important;
# # #     font-weight: 600 !important;
# # # }}

# # # /* ---------------- LABELS ---------------- */
# # # label {{
# # #     font-weight: 500 !important;
# # #     font-size: 14px !important;
# # # }}

# # # /* ---------------- CODE / RDF PATHS ---------------- */
# # # code {{
# # #     font-family: 'JetBrains Mono', monospace !important;
# # #     font-size: 13px;
# # #     background-color: #f3f4f6;
# # #     padding: 3px 6px;
# # #     border-radius: 6px;
# # # }}

# # # /* ---------------- BUTTONS ---------------- */
# # # .stButton>button {{
# # #     font-family: 'Poppins', sans-serif;
# # #     font-weight: 600;
# # #     border-radius: 12px;
# # #     font-size: 15px;
# # #     padding: 10px 16px;
# # #     background: linear-gradient(135deg, #ff4b4b, #ff6b6b);
# # #     color: white;
# # #     border: none;
# # #     transition: 0.2s ease-in-out;
# # # }}

# # # .stButton>button:hover {{
# # #     transform: translateY(-2px);
# # #     box-shadow: 0 6px 14px rgba(0,0,0,0.2);
# # # }}

# # # /* ---------------- FILE UPLOADER ---------------- */
# # # .stFileUploader {{
# # #     font-family: 'Inter', sans-serif;
# # #     border-radius: 12px;
# # # }}

# # # /* ---------------- SIDEBAR BACKGROUND ---------------- */
# # # section[data-testid="stSidebar"] {{
# # #     background-color: #f8fafc;
# # # }}

# # # </style>

# # # <div class="top-wrapper">
# # #     <img src="data:image/gif;base64,{encoded_gif}" />
# # # </div>
# # # """, unsafe_allow_html=True)

# # # # Sidebar
# # # st.sidebar.header("⚙️ Settings")

# # # # Upload SHACL
# # # shacl_file = st.sidebar.file_uploader(" Upload SHACL File", type=["ttl", "shacl"])
# # # available_properties = []


# # # def render_shacl_tree(shape_map):
# # #     DATATYPE_MAP = {
# # #         "http://www.w3.org/2001/XMLSchema#string": "string",
# # #         "http://www.w3.org/2001/XMLSchema#integer": "integer",
# # #         "http://www.w3.org/2001/XMLSchema#decimal": "decimal number",
# # #         "http://www.w3.org/2001/XMLSchema#boolean": "true/false value",
# # #         "http://www.w3.org/2001/XMLSchema#date": "date (YYYY-MM-DD)",
# # #         "http://www.w3.org/2001/XMLSchema#dateTime": "datetime (YYYY-MM-DDTHH:MM:SS)",
# # #         "IRI": "IRI (Internationalized Resource Identifier)"
# # #     }

# # #     for shape in shape_map:
# # #         target_class = (shape.get("target_classes") or ["Unknown"])[0].split("/")[-1]
# # #         st.sidebar.markdown(f"### ⊛ {target_class}")
        
# # #         for prop in shape.get("properties", []):
# # #             constraints = {list(c.keys())[0]: list(c.values())[0] for c in prop["constraints"]}
            
# # #             path = constraints.get("http://www.w3.org/ns/shacl#path", "unknown")
            
# # #             datatype = constraints.get("http://www.w3.org/ns/shacl#datatype")
# # #             if not datatype and constraints.get("http://www.w3.org/ns/shacl#nodeKind") == "http://www.w3.org/ns/shacl#IRI":
# # #                 datatype = "IRI"
# # #             if not datatype:
# # #                 datatype = "http://www.w3.org/2001/XMLSchema#string"
            
# # #             prop_name = path.split("/")[-1]
# # #             datatype_name = DATATYPE_MAP.get(datatype, datatype.split("#")[-1] if "#" in datatype else datatype)
            
# # #             st.sidebar.markdown(f"  - `{prop_name}`: *{datatype_name}*")


# # # # Fetch saved models using the FastAPI endpoint
# # # def fetch_saved_models():
# # #     response = requests.get("http://localhost:8000/models/saved")  # Adjust URL if needed
# # #     if response.status_code == 200:
# # #         return response.json().get("saved_models", [])
# # #     else:
# # #         st.error("Failed to fetch models")
# # #         return []

# # # if shacl_file:
# # #     files = {"file": shacl_file.getvalue()}
# # #     response = requests.post("http://fastapi-backend:8000/upload_shacl", files=files)
# # #     if response.status_code == 200:
# # #         st.sidebar.success("✅ SHACL File Uploaded Successfully")
# # #         data = response.json()
# # #         shape_map = data.get("shape_map", [])
# # #         render_shacl_tree(shape_map)

# # #         # Get available properties
# # #         try:
# # #             prop_response = requests.get("http://fastapi-backend:8000/get_shacl_properties")
# # #             if prop_response.status_code == 200:
# # #                 available_properties = prop_response.json().get("properties", [])
# # #                 st.sidebar.success("✅ Retrieved SHACL properties")
# # #             else:
# # #                 st.sidebar.warning("⚠️ Could not fetch properties")
# # #         except Exception as e:
# # #             st.sidebar.error(f"❌ Error fetching SHACL properties: {e}")
# # #     else:
# # #         st.sidebar.error("❌ Error uploading SHACL File")

# # # # Show multiselect for properties if available
# # # selected_properties = st.sidebar.multiselect(
# # #     " Select Properties to Include in Generation",
# # #     options=available_properties,
# # #     default=available_properties
# # # )

# # # st.sidebar.markdown("### Select Model & Distribution per Property")

# # # property_model_map = {}
# # # property_distribution_map = {}
# # # model_options = ["LLM", "GAN", "VAE"]
# # # distribution_options = ["Normal", "Uniform", "Custom"]

# # # gan_saved_models = requests.get("http://fastapi-backend:8000/gan_models").json().get("saved_models", [])
# # # vae_saved_models = requests.get("http://fastapi-backend:8000/vae_models").json().get("saved_models", [])
# # # saved_gan_models = [model for model in gan_saved_models if model.lower().endswith("gan")]
# # # saved_vae_models = [model for model in vae_saved_models if model.lower().endswith("vae")]

# # # import hashlib

# # # def make_streamlit_key(prefix, path):
# # #     # Use hash to ensure uniqueness and avoid problematic characters
# # #     hashed = hashlib.md5(path.encode()).hexdigest()
# # #     return f"{prefix}_{hashed}"

# # # # Helper to fetch models dynamically based on type
# # # def fetch_models(model_type):
# # #     try:
# # #         response = requests.get("http://fastapi-backend:8000/models", params={"model_type": model_type})
# # #         if response.status_code == 200:
# # #             models = response.json()
# # #             return models if models else []
# # #         else:
# # #             return []
# # #     except Exception as e:
# # #         st.warning(f"Could not fetch {model_type} models: {e}")
# # #         return []

# # # # --- Loop over selected properties ---
# # # for i, prop in enumerate(selected_properties):
# # #     st.write("DEBUG PROP:", prop)
# # #     path = prop["path"]
    
# # #     # Unique key for the property model select
# # #     model_key = make_streamlit_key("model_select", f"{i}_{path}")
# # #     selected_model = st.sidebar.selectbox(f"Model for `{path}`", model_options, key=model_key)
    
# # #     property_model_map[path] = {"type": selected_model}

# # #     # If GAN or VAE is selected, fetch models dynamically
# # #     if selected_model in ["GAN", "VAE"]:
# # #         available_models = fetch_models(selected_model)
# # #         options = ["default_model"] + available_models if available_models else ["No models found"]
# # #         model_select_key = make_streamlit_key(f"{selected_model.lower()}_model_select", f"{i}_{path}")
# # #         selected_model_name = st.sidebar.selectbox(
# # #             f"Select saved {selected_model} model for `{path}`",
# # #             options,
# # #             key=model_select_key
# # #         )
# # #         property_model_map[path]["name"] = selected_model_name
# # #     else:
# # #         # Default for LLM
# # #         property_model_map[path]["name"] = "LLM_default"

# # #     # --- Distribution Selection ---
# # #     dist_key = make_streamlit_key("distribution_select", f"{i}_{path}")
# # #     selected_distribution = st.sidebar.selectbox(f"Distribution for `{path}`", distribution_options, key=dist_key)
    
# # #     mean_key = make_streamlit_key("mean", f"{i}_{path}")
# # #     std_key = make_streamlit_key("std", f"{i}_{path}")
# # #     low_key = make_streamlit_key("low", f"{i}_{path}")
# # #     high_key = make_streamlit_key("high", f"{i}_{path}")
# # #     custom_key = make_streamlit_key("custom", f"{i}_{path}")

# # #     dist_params = {}
# # #     if selected_distribution == "Normal":
# # #         dist_params["mean"] = st.sidebar.number_input(f"Mean for `{path}`", value=0.0, key=mean_key)
# # #         dist_params["stddev"] = st.sidebar.number_input(f"Std Dev for `{path}`", value=1.0, key=std_key)
# # #     elif selected_distribution == "Uniform":
# # #         dist_params["low"] = st.sidebar.number_input(f"Low for `{path}`", value=0.0, key=low_key)
# # #         dist_params["high"] = st.sidebar.number_input(f"High for `{path}`", value=1.0, key=high_key)
# # #     elif selected_distribution == "Skewed":
# # #         dist_params["custom_param"] = st.sidebar.text_input(f"Custom Param for `{path}`", key=custom_key)
# # #         dist_params["low"] = st.sidebar.number_input(f"Low for `{path}`", value=0.0, key=low_key)
# # #         dist_params["high"] = st.sidebar.number_input(f"High for `{path}`", value=1.0, key=high_key)

# # #     property_distribution_map[path] = {
# # #         "type": selected_distribution,
# # #         "parameters": dist_params
# # #     }


# # # # Distribution selection for all properties
# # # distribution_type = st.sidebar.selectbox("Select Data Distribution", ["Normal", "Uniform", "Skewed"])
# # # num_samples = st.sidebar.number_input("Number of Samples", min_value=1, value=10, step=1)

# # # # Distribution parameters
# # # parameters = {}
# # # if distribution_type == "Normal":
# # #     parameters["mean"] = st.sidebar.number_input("Mean", value=0.0)
# # #     parameters["stddev"] = st.sidebar.number_input("Standard Deviation", value=1.0)
# # # elif distribution_type == "Uniform":
# # #     parameters["low"] = st.sidebar.number_input("Low", value=0.0)
# # #     parameters["high"] = st.sidebar.number_input("High", value=1.0)
# # # elif distribution_type == "skewed":
# # #     parameters["custom_param"] = st.sidebar.text_input("Custom Parameter")
# # #     parameters["low"] = st.sidebar.number_input("Low", value=0.0)
# # #     parameters["high"] = st.sidebar.number_input("High", value=1.0)

# # # # Theme toggle
# # # theme_mode = st.sidebar.radio("Theme", ["Light", "Dark"])
# # # if theme_mode == "Light":
# # #     st.markdown("<style>body { background: white; color: black; }</style>", unsafe_allow_html=True)
# # # else:
# # #     st.markdown("<style>body { background: #1e1e1e; color: white; }</style>", unsafe_allow_html=True)


# # # def render_custom_progress_bar(progress):
# # #     percentage = int(progress)
# # #     bar = f"""
# # #     <div style="background-color: #e0e0e0; border-radius: 8px; height: 24px; width: 100%; margin-top: 20px;">
# # #         <div style="
# # #             background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
# # #             width: {percentage}%;
# # #             height: 100%;
# # #             border-radius: 8px;
# # #             text-align: center;
# # #             color: white;
# # #             font-weight: bold;
# # #             line-height: 24px;">
# # #             {percentage}%
# # #         </div>
# # #     </div>
# # #     """
# # #     st.markdown(bar, unsafe_allow_html=True)



# # # def render_custom_progress_bar(progress, placeholder):
# # #     percentage = int(progress)
# # #     if percentage > 100:
# # #         percentage = 100
# # #     bar = f"""
# # #     <div style="background-color: #e0e0e0; border-radius: 8px; height: 24px; width: 100%; margin-top: 20px;">
# # #         <div style="
# # #             background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
# # #             width: {percentage}%;
# # #             height: 100%;
# # #             border-radius: 8px;
# # #             text-align: center;
# # #             color: white;
# # #             font-weight: bold;
# # #             line-height: 24px;">
# # #             {percentage}%
# # #         </div>
# # #     </div>
# # #     """
# # #     placeholder.markdown(bar, unsafe_allow_html=True)


# # # from rdflib import Graph, URIRef, Literal, Namespace, RDF

# # # EX = Namespace("http://example.org/")

# # # def convert_to_jsonld(data):
# # #     graph = Graph()
# # #     graph.bind("ex", EX)

# # #     for i, item in enumerate(data):
# # #         subject = URIRef(f"http://example.org/item/{i}")
# # #         graph.add((subject, RDF.type, EX.SyntheticEntity))

# # #         for key, value in item.items():
# # #             predicate = URIRef(f"http://example.org/property/{key}")
# # #             object_ = Literal(value)
# # #             graph.add((subject, predicate, object_))

# # #     return graph.serialize(format="json-ld", indent=2)

# # # def convert_to_ttl(data):
# # #     graph = Graph()
# # #     graph.bind("ex", EX)

# # #     for i, item in enumerate(data):
# # #         subject = URIRef(f"http://example.org/item/{i}")
# # #         graph.add((subject, RDF.type, EX.SyntheticEntity))

# # #         for key, value in item.items():
# # #             predicate = URIRef(f"http://example.org/property/{key}")
# # #             object_ = Literal(value)
# # #             graph.add((subject, predicate, object_))

# # #     return graph.serialize(format="turtle")


# # # from io import BytesIO

# # # def get_serialized_jsonld_bytes(synthetic_data):
# # #     return BytesIO(convert_to_jsonld(synthetic_data).encode("utf-8"))

# # # def get_serialized_ttl_bytes(synthetic_data):
# # #     return BytesIO(convert_to_ttl(synthetic_data).encode("utf-8"))

# # # def get_serialized_json_bytes(synthetic_data):
# # #     return BytesIO(json.dumps(synthetic_data, indent=2).encode("utf-8"))



# # # def iri_to_fragment(iri: str) -> str:
# # #     """
# # #     Converts a full IRI to a fragment usable by GAN/VAE generators.
# # #     Example: "http://example.org/ontology#Veneziana" → "#Veneziana"
# # #     """
# # #     if "#" in iri:
# # #         return "#" + iri.split("#")[-1]
# # #     else:
# # #         return iri.split("/")[-1]
    

# # # from rdflib import Graph, RDF, RDFS, URIRef

# # # def generate_instance_shapes(ontology_graph: Graph, base_prefix="http://example.org/") -> list:
# # #     """
# # #     Generates a SHACL NodeShape for every individual instance in the ontology.
# # #     Returns a list of dicts compatible with your Streamlit SHACL renderer.
# # #     """
# # #     shapes = []

# # #     # Iterate over all individuals
# # #     for s in ontology_graph.subjects(RDF.type, None):
# # #         s_uri = str(s)
# # #         s_fragment = iri_to_fragment(s_uri)
# # #         s_class = None

# # #         # Try to get the class (rdf:type) of the instance
# # #         for o in ontology_graph.objects(s, RDF.type):
# # #             if isinstance(o, URIRef):
# # #                 s_class = iri_to_fragment(str(o))
# # #                 break

# # #         # Collect all outgoing predicates
# # #         properties = []
# # #         for p, o in ontology_graph.predicate_objects(s):
# # #             prop_dict = {
# # #                 "constraints": [{str(RDF.type): str(o)}],  # placeholder
# # #                 "path": str(p)
# # #             }
# # #             properties.append(prop_dict)

# # #         shape = {
# # #             "target_classes": [s_class or s_fragment],
# # #             "properties": properties
# # #         }
# # #         shapes.append(shape)

# # #     return shapes





# # # import requests
# # # import json
# # # import time
# # # import streamlit as st

# # # def generate_synthetic_data(progress_placeholder, num_samples, payload):
# # #     """
# # #     Generates synthetic data based on the selected model type:
# # #     - VAE → calls /generate_vae
# # #     - GAN → calls /gan/load-and-generate
# # #     - LLM → retains previous streaming behavior
# # #     The progress bar now updates per sample.
# # #     """
# # #     try:
# # #         selected_model_type = payload.get("model_name", ["LLM"])[0]
# # #         selected_model_type_upper = selected_model_type.upper()

# # #         # VAE endpoint
# # #         if selected_model_type_upper == "VAE":
# # #             url = "http://fastapi-backend:8000/generate_vae"
# # #             payload_backend = {
# # #                 "model_name": selected_model_type,
# # #                 "subject": payload["subject_input"],
# # #                 "predicate": payload["predicate_input"],
# # #                 "num_samples": payload["num_samples"],
# # #                 "distribution": payload.get("distribution_type", "normal").lower(),
# # #                 "dist_params": payload.get("parameters", {})
# # #             }
# # #             response = requests.post(url, json=payload_backend, headers={"accept": "application/json"})
# # #             if response.status_code == 200:
# # #                 generated_objects = response.json().get("generated_objects", [])
# # #                 # Option 2: increment progress per sample
# # #                 total_samples = len(generated_objects)
# # #                 for i, _ in enumerate(generated_objects, start=1):
# # #                     progress = (i / total_samples) * 100
# # #                     render_custom_progress_bar(progress, progress_placeholder)
# # #                     time.sleep(0.05)  # small delay for smooth UI
# # #                 return generated_objects
# # #             else:
# # #                 st.error(f"❌ VAE backend error {response.status_code}: {response.text}")
# # #                 return []

# # #         # GAN endpoint
# # #         elif selected_model_type_upper == "GAN":
# # #             url = "http://fastapi-backend:8000/gan/load-and-generate"
# # #             payload_backend = {
# # #                 "model_name": selected_model_type,
# # #                 "subject": payload["subject_input"],
# # #                 "predicate": payload["predicate_input"],
# # #                 "num_samples": payload["num_samples"],
# # #                 "distribution": payload.get("distribution_type", "normal").lower(),
# # #                 "dist_params": payload.get("parameters", {})
# # #             }
# # #             response = requests.post(url, json=payload_backend, headers={"accept": "application/json"})
# # #             if response.status_code == 200:
# # #                 generated_objects = response.json().get("generated_objects", [])
# # #                 # Option 2: increment progress per sample
# # #                 total_samples = len(generated_objects)
# # #                 for i, _ in enumerate(generated_objects, start=1):
# # #                     progress = (i / total_samples) * 100
# # #                     render_custom_progress_bar(progress, progress_placeholder)
# # #                     time.sleep(0.05)
# # #                 return generated_objects
# # #             else:
# # #                 st.error(f"❌ GAN backend error {response.status_code}: {response.text}")
# # #                 return []

# # #         # LLM streaming unchanged
# # #         else:
# # #             url = "http://fastapi-backend:8000/generate_data"
# # #             payload_backend = payload
# # #             headers = {"Accept": "text/event-stream"}
# # #             synthetic_data = []
# # #             last_progress = 0

# # #             with requests.post(url, json=payload_backend, headers=headers, stream=True) as response:
# # #                 if response.status_code != 200:
# # #                     st.error(f"❌ LLM backend error {response.status_code}: {response.text}")
# # #                     return []

# # #                 for line in response.iter_lines():
# # #                     if line:
# # #                         decoded_line = line.decode("utf-8")
# # #                         if decoded_line.startswith("data: "):
# # #                             event_data = json.loads(decoded_line[6:])
# # #                             if event_data.get("type") == "progress_update":
# # #                                 progress = event_data["progress"]
# # #                                 if abs(progress - last_progress) > 0.01:
# # #                                     last_progress = progress
# # #                                     render_custom_progress_bar(progress, progress_placeholder)
# # #                                     time.sleep(0.05)
# # #                             elif event_data.get("type") == "final_result":
# # #                                 synthetic_data = event_data.get("synthetic_data", [])

# # #             return synthetic_data

# # #     except Exception as e:
# # #         st.error(f"❌ Error during data generation: {e}")
# # #         return []


# # # # # UI Trigger for generation
# # # # if "synthetic_data" not in st.session_state:
# # # #     st.session_state.synthetic_data = None
# # # # user_message = st.sidebar.text_area("💬 Use this Prompt for batch property", placeholder="E.g., Generate realistic company names...")

# # # # # Progress placeholder for UI updates
# # # # progress_placeholder = st.empty()

# # # # def extract_local_name(iri: str):
# # # #     if "#" in iri:
# # # #         return iri.split("#")[-1]
# # # #     return iri.split("/")[-1]

# # # # if st.sidebar.button("⚉ Generate Synthetic Data (Batch Request)"):
# # # #     st.markdown("""
# # # #         <h2 style="text-align: left; font-family: 'Roboto', sans-serif; font-size: 14px; font-weight: bold; color: #2c3e50; margin-top: 20px;">
# # # #             ⚇ Generating Synthetic RDF Data (Batch Mode)...
# # # #         </h2>
# # # #     """, unsafe_allow_html=True)

# # # #     # Loop over selected properties
# # # #     all_generated_data = []

# # # #     # for path, prop_info in property_model_map.items():
# # # #     #     model_name = prop_info.get("name", "string_vae")
# # # #     #     model_type = prop_info.get("type", "LLM")

# # # #     #     payload = {
# # # #     #         "num_samples": num_samples,
# # # #     #         "distribution_type": property_distribution_map.get(path, {}).get("type", "normal").lower(),
# # # #     #         "parameters": property_distribution_map.get(path, {}).get("parameters", {}),
# # # #     #         "property_model_map": property_model_map,
# # # #     #         "user_message": user_message,
# # # #     #         "model_name": [model_name],
# # # #     #         "subject_input": path.split('/')[-1],       # or prop['shape'] if available
# # # #     #         "predicate_input": path.split('/')[-1]
# # # #     #     }

# # # #     for shape in shape_map:
# # # #         target_class_iri = shape.get("target_classes", [None])[0]
# # # #         subject_local = extract_local_name(target_class_iri)

# # # #         for prop in shape.get("properties", []):
# # # #             path = prop["path"]
# # # #             predicate_local = extract_local_name(path)

# # # #             # Use FULL IRI as key for UI
# # # #             property_model_map[path] = {
# # # #                 "type": "LLM",
# # # #                 "name": "LLM_default",
# # # #                 "subject": subject_local,
# # # #                 "predicate": predicate_local
# # # #             }

# # # #         generated_objects = generate_synthetic_data(progress_placeholder, num_samples, payload)
# # # #         all_generated_data.append({
# # # #             "property": path,
# # # #             "generated_objects": generated_objects
# # # #         })

# # # #     # Save in session state
# # # #     st.session_state.synthetic_data = all_generated_data


# # # # UI Trigger for generation
# # # if "synthetic_data" not in st.session_state:
# # #     st.session_state.synthetic_data = None

# # # user_message = st.sidebar.text_area(
# # #     "Use this Prompt for batch property",
# # #     placeholder="E.g., Generate realistic company names..."
# # # )

# # # progress_placeholder = st.empty()

# # # def extract_local_name(iri: str):
# # #     if "#" in iri:
# # #         return iri.split("#")[-1]
# # #     return iri.split("/")[-1]


# # # if st.sidebar.button("⚉ Generate Synthetic Data (Batch Request)"):

# # #     st.markdown("""
# # #         <h2 style="text-align: left; font-size: 14px; font-weight: bold; margin-top: 20px;">
# # #             ⚇ Generating Synthetic RDF Data (Batch Mode)...
# # #         </h2>
# # #     """, unsafe_allow_html=True)

# # #     all_generated_data = []

# # #     for shape in shape_map:

# # #         # 🔹 Get subject from SHACL targetClass
# # #         target_class_iri = shape.get("target_classes", [None])[0]
# # #         subject_local = extract_local_name(target_class_iri)

# # #         for prop in shape.get("properties", []):

# # #             path = prop["path"]
# # #             predicate_local = extract_local_name(path)

# # #             # 🔹 Get selected model from sidebar (DO NOT overwrite it)
# # #             prop_info = property_model_map.get(path, {})
# # #             model_name = prop_info.get("name", "LLM_default")

# # #             payload = {
# # #                 "num_samples": num_samples,
# # #                 "distribution_type": property_distribution_map.get(path, {}).get("type", "normal").lower(),
# # #                 "parameters": property_distribution_map.get(path, {}).get("parameters", {}),
# # #                 "property_model_map": property_model_map,
# # #                 "user_message": user_message,
# # #                 "model_name": [model_name],
# # #                 "subject_input": subject_local,
# # #                 "predicate_input": predicate_local
# # #             }

# # #             generated_objects = generate_synthetic_data(
# # #                 progress_placeholder,
# # #                 num_samples,
# # #                 payload
# # #             )

# # #             all_generated_data.append({
# # #                 "property": path,
# # #                 "generated_objects": generated_objects
# # #             })

# # #     st.session_state.synthetic_data = all_generated_data

# # # # Display generated data
# # # synthetic_data = st.session_state.get("synthetic_data")
# # # if synthetic_data:
# # #     st.success("✅ Synthetic data generated successfully!")
# # #     st.subheader("Generated Data")
# # #     st.json(synthetic_data)

# # #     # Generate download buttons
# # #     json_data = get_serialized_json_bytes(synthetic_data)
# # #     jsonld_data = get_serialized_jsonld_bytes(synthetic_data)
# # #     ttl_data = get_serialized_ttl_bytes(synthetic_data)

# # #     st.download_button("📥 Download Synthetic Data (JSON)", data=json_data, file_name="synthetic_data_batch.json", mime="application/json")
# # #     st.download_button("📥 Download Synthetic Data (JSON-LD)", data=jsonld_data, file_name="synthetic_data_batch.jsonld", mime="application/ld+json")
# # #     st.download_button("📥 Download Synthetic Data (TTL)", data=ttl_data, file_name="synthetic_data_batch.ttl", mime="text/turtle")



# # import streamlit as st
# # import requests
# # import json
# # import base64
# # import rdflib
# # from io import BytesIO
# # import os

# # # ----------------------------
# # # --- STYLING: Background + GIF ---
# # # ----------------------------
# # def get_base64_of_file(file_path):
# #     with open(file_path, "rb") as f:
# #         return base64.b64encode(f.read()).decode()

# # # Paths to your assets
# # background_path = "./background.png"  # background image
# # gif_path = "./srdfgen.gif"            # top hero GIF

# # # Encode
# # base64_background = get_base64_of_file(background_path)
# # encoded_gif = get_base64_of_file(gif_path)

# # st.markdown(f"""
# # <style>
# # @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

# # /* ---------------- GLOBAL FONT ---------------- */
# # html, body, [class*="css"] {{
# #     font-family: 'Inter', sans-serif;
# #     font-size: 15px;
# #     letter-spacing: 0.2px;
# # }}

# # /* ---------------- BACKGROUND ---------------- */
# # .stApp {{
# #     background: url("data:image/png;base64,{base64_background}");
# #     background-size: 1000px;
# #     background-position: center;
# #     background-attachment: fixed;
# # }}

# # /* ---------------- TOP CENTER HERO ---------------- */
# # .top-wrapper {{
# #     display: flex;
# #     flex-direction: column;
# #     align-items: center;
# #     justify-content: center;
# #     margin-top: 30px;
# #     margin-bottom: 40px;
# #     text-align: center;
# # }}

# # .top-wrapper img {{
# #     width: 280px;
# #     max-width: 90%;
# # }}

# # .top-wrapper h1 {{
# #     font-family: 'Poppins', sans-serif !important;
# #     font-size: 32px !important;
# #     font-weight: 700 !important;
# #     margin-top: 15px;
# #     color: #1f2937;
# # }}

# # /* ---------------- BUTTONS ---------------- */
# # .stButton>button {{
# #     font-family: 'Poppins', sans-serif;
# #     font-weight: 600;
# #     border-radius: 12px;
# #     font-size: 15px;
# #     padding: 10px 16px;
# #     background: linear-gradient(135deg, #ff4b4b, #ff6b6b);
# #     color: white;
# #     border: none;
# #     transition: 0.2s ease-in-out;
# # }}

# # .stButton>button:hover {{
# #     transform: translateY(-2px);
# #     box-shadow: 0 6px 14px rgba(0,0,0,0.2);
# # }}

# # /* ---------------- CODE / RDF PATHS ---------------- */
# # code {{
# #     font-family: 'JetBrains Mono', monospace !important;
# #     font-size: 13px;
# #     background-color: #f3f4f6;
# #     padding: 3px 6px;
# #     border-radius: 6px;
# # }}
# # </style>

# # <div class="top-wrapper">
# #     <img src="data:image/gif;base64,{encoded_gif}" />
# #     <h1>SHACL → RDF Generator</h1>
# # </div>
# # """, unsafe_allow_html=True)

# # # ----------------------------
# # # --- SIDEBAR: Model Selection ---
# # # ----------------------------
# # model_type = st.sidebar.selectbox("Model Type", ["LLM", "VAE", "GAN"])
# # model_name = None
# # if model_type in ["VAE", "GAN"]:
# #     model_name = st.sidebar.text_input("Saved Model Name (for VAE/GAN)")

# # # ----------------------------
# # # --- Upload SHACL file ---
# # # ----------------------------
# # uploaded_file = st.file_uploader("Upload SHACL file (.ttl/.shacl)", type=["ttl", "shacl"])
# # json_schema = []

# # if uploaded_file is not None:
# #     files = {"file": uploaded_file}
# #     response = requests.post("http://localhost:8000/upload_shacl_and_generate_schema", files=files)
# #     if response.status_code == 200:
# #         data = response.json()
# #         json_schema = data.get("json_schema", [])
# #         st.success("SHACL file uploaded and schema extracted!")
# #     else:
# #         st.error(f"Error: {response.text}")

# # # ----------------------------
# # # --- Editable Distributions ---
# # # ----------------------------
# # if json_schema:
# #     st.subheader("Edit Distributions (optional)")
# #     for i, prop in enumerate(json_schema):
# #         st.markdown(f"**{prop['path']}** ({prop['datatype']})")
# #         if prop["distribution_type"] == "categorical":
# #             allowed_list = st.text_input(f"Allowed list for {prop['path']}", ",".join(prop["distribution_params"].get("allowed_list", [])), key=f"cat_{i}")
# #             probabilities = st.text_input(f"Probabilities for {prop['path']}", ",".join([str(p) for p in prop["distribution_params"].get("probabilities", [])]), key=f"prob_{i}")
# #             json_schema[i]["distribution_params"]["allowed_list"] = [x.strip() for x in allowed_list.split(",")]
# #             json_schema[i]["distribution_params"]["probabilities"] = [float(x.strip()) for x in probabilities.split(",")]
# #         elif prop["distribution_type"] == "numeric":
# #             mean = st.number_input(f"Mean for {prop['path']}", value=prop["distribution_params"].get("mean", 10.0), key=f"mean_{i}")
# #             std = st.number_input(f"Std for {prop['path']}", value=prop["distribution_params"].get("std", 2.0), key=f"std_{i}")
# #             min_val = st.number_input(f"Min for {prop['path']}", value=prop["distribution_params"].get("min", 0.0), key=f"min_{i}")
# #             max_val = st.number_input(f"Max for {prop['path']}", value=prop["distribution_params"].get("max", 20.0), key=f"max_{i}")
# #             json_schema[i]["distribution_params"].update({"mean": mean, "std": std, "min": min_val, "max": max_val})

# # # # ----------------------------
# # # # --- Generate RDF Button ---
# # # # ----------------------------
# # # if st.button("Generate RDF Data"):
# # #     if not json_schema:
# # #         st.warning("Upload a SHACL file first!")
# # #     else:
# # #         payload = {
# # #             "model_type": model_type,
# # #             "model_name": model_name,
# # #             "json_schema": json_schema
# # #         }
# # #         gen_response = requests.post("http://localhost:8000/generate_from_shacl", json=payload)
# # #         if gen_response.status_code == 200:
# # #             gen_data = gen_response.json()
# # #             ttl_data = gen_data["rdf_turtle"]

# # #             st.subheader("Generated RDF (Turtle)")
# # #             st.code(ttl_data, language="turtle")

# # #             # Convert to JSON-LD
# # #             g = rdflib.Graph()
# # #             g.parse(data=ttl_data, format="turtle")
# # #             jsonld_data = g.serialize(format="json-ld", indent=2)

# # #             st.subheader("Generated RDF (JSON-LD)")
# # #             st.code(jsonld_data, language="json")

# # #             # Download buttons
# # #             ttl_bytes = BytesIO(ttl_data.encode("utf-8"))
# # #             jsonld_bytes = BytesIO(jsonld_data.encode("utf-8"))

# # #             st.download_button("Download RDF (Turtle)", data=ttl_bytes, file_name="generated_data.ttl", mime="text/turtle")
# # #             st.download_button("Download RDF (JSON-LD)", data=jsonld_bytes, file_name="generated_data.jsonld", mime="application/ld+json")
# # #         else:
# # #             st.error(f"Generation failed: {gen_response.text}")


# # # # ----------------------------
# # # # --- Generate RDF Button + Preview ---
# # # # ----------------------------
# # # if st.button("Generate RDF Data"):
# # #     if not json_schema:
# # #         st.warning("Upload a SHACL file first!")
# # #     else:
# # #         payload = {
# # #             "model_type": model_type,
# # #             "model_name": model_name,
# # #             "json_schema": json_schema
# # #         }
# # #         gen_response = requests.post("http://localhost:8000/generate_from_shacl", json=payload)
# # #         if gen_response.status_code == 200:
# # #             gen_data = gen_response.json()
# # #             ttl_data = gen_data["rdf_turtle"]

# # #             st.subheader("Generated RDF (Turtle)")
# # #             st.code(ttl_data, language="turtle")

# # #             # Convert to JSON-LD
# # #             g = rdflib.Graph()
# # #             g.parse(data=ttl_data, format="turtle")
# # #             jsonld_data = g.serialize(format="json-ld", indent=2)

# # #             st.subheader("Generated RDF (JSON-LD)")
# # #             st.code(jsonld_data, language="json")

# # #             # -------------------
# # #             # Preview Table
# # #             # -------------------
# # #         #     st.subheader("Preview Triples")
# # #         #     triples_preview = []
# # #         #     for s, p, o in g:
# # #         #         triples_preview.append({
# # #         #             "Subject": str(s),
# # #         #             "Predicate": str(p),
# # #         #             "Object": str(o)
# # #         #         })
# # #         #     st.table(triples_preview)

# # #         #     # -------------------
# # #         #     # Download buttons
# # #         #     # -------------------
# # #         #     from io import BytesIO
# # #         #     ttl_bytes = BytesIO(ttl_data.encode("utf-8"))
# # #         #     jsonld_bytes = BytesIO(jsonld_data.encode("utf-8"))

# # #         #     st.download_button("Download RDF (Turtle)", data=ttl_bytes, file_name="generated_data.ttl", mime="text/turtle")
# # #         #     st.download_button("Download RDF (JSON-LD)", data=jsonld_bytes, file_name="generated_data.jsonld", mime="application/ld+json")
# # #         # else:
# # #         #     st.error(f"Generation failed: {gen_response.text}")

# # #         from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
# # #         import pandas as pd
# # #         from io import BytesIO

# # #         # -------------------
# # #         # Interactive Preview Table
# # #         # -------------------
# # #         st.subheader("Preview Triples (Interactive)")
# # #         triples_preview = []
# # #         for s, p, o in g:
# # #             triples_preview.append({
# # #                 "Subject": str(s),
# # #                 "Predicate": str(p),
# # #                 "Object": str(o)
# # #             })

# # #         df_triples = pd.DataFrame(triples_preview)

# # #         # Build AgGrid options
# # #         gb = GridOptionsBuilder.from_dataframe(df_triples)
# # #         gb.configure_default_column(filterable=True, sortable=True, editable=False, resizable=True)
# # #         gb.configure_selection(selection_mode="multiple", use_checkbox=True)
# # #         grid_options = gb.build()

# # #         # Render AgGrid table and capture selected rows
# # #         grid_response = AgGrid(
# # #             df_triples,
# # #             gridOptions=grid_options,
# # #             height=300,
# # #             fit_columns_on_grid_load=True,
# # #             update_mode=GridUpdateMode.MODEL_CHANGED,
# # #         )

# # #         selected_rows = grid_response["selected_rows"]
# # #         selected_df = pd.DataFrame(selected_rows)

# # #         # -------------------
# # #         # Download Selected Triples
# # #         # -------------------
# # #         if not selected_df.empty:
# # #             # Convert selected triples to Turtle
# # #             g_selected = rdflib.Graph()
# # #             for _, row in selected_df.iterrows():
# # #                 g_selected.add((rdflib.URIRef(row["Subject"]), rdflib.URIRef(row["Predicate"]), rdflib.URIRef(row["Object"])))
# # #             ttl_selected = g_selected.serialize(format="turtle")
# # #             jsonld_selected = g_selected.serialize(format="json-ld", indent=2)

# # #             ttl_bytes = BytesIO(ttl_selected.encode("utf-8"))
# # #             jsonld_bytes = BytesIO(jsonld_selected.encode("utf-8"))

# # #             st.download_button(
# # #                 "Download Selected Triples (Turtle)",
# # #                 data=ttl_bytes,
# # #                 file_name="selected_triples.ttl",
# # #                 mime="text/turtle"
# # #             )
# # #             st.download_button(
# # #                 "Download Selected Triples (JSON-LD)",
# # #                 data=jsonld_bytes,
# # #                 file_name="selected_triples.jsonld",
# # #                 mime="application/ld+json"
# # #             )


# # import streamlit as st
# # import requests
# # import pandas as pd
# # import rdflib
# # from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
# # from io import BytesIO

# # # ----------------------------
# # # --- Generate RDF Data ---
# # # ----------------------------
# # st.subheader("Generate RDF Data from SHACL")

# # if st.button("Generate RDF"):
# #     if not json_schema:
# #         st.warning("Upload a SHACL file first!")
# #     else:
# #         payload = {
# #             "model_type": model_type,  # LLM / GAN / VAE
# #             "model_name": model_name,  # Required if GAN or VAE
# #             "json_schema": json_schema
# #         }
# #         gen_response = requests.post("http://localhost:8000/generate_from_shacl", json=payload)
# #         if gen_response.status_code == 200:
# #             gen_data = gen_response.json()
# #             ttl_data = gen_data["rdf_turtle"]

# #             # -------------------
# #             # Full RDF Preview
# #             # -------------------
# #             st.subheader("Generated RDF (Turtle)")
# #             st.code(ttl_data, language="turtle")

# #             g = rdflib.Graph()
# #             g.parse(data=ttl_data, format="turtle")
# #             jsonld_data = g.serialize(format="json-ld", indent=2)

# #             st.subheader("Generated RDF (JSON-LD)")
# #             st.code(jsonld_data, language="json")

# #             # -------------------
# #             # Interactive Triple Table
# #             # -------------------
# #             st.subheader("Preview Triples (Interactive)")
# #             triples_preview = [{"Subject": str(s), "Predicate": str(p), "Object": str(o)} for s, p, o in g]
# #             df_triples = pd.DataFrame(triples_preview)

# #             gb = GridOptionsBuilder.from_dataframe(df_triples)
# #             gb.configure_default_column(filterable=True, sortable=True, resizable=True)
# #             gb.configure_selection(selection_mode="multiple", use_checkbox=True)
# #             grid_options = gb.build()

# #             grid_response = AgGrid(
# #                 df_triples,
# #                 gridOptions=grid_options,
# #                 height=300,
# #                 fit_columns_on_grid_load=True,
# #                 update_mode=GridUpdateMode.MODEL_CHANGED,
# #             )

# #             selected_rows = grid_response["selected_rows"]
# #             selected_df = pd.DataFrame(selected_rows)

# #             # -------------------
# #             # Download Buttons
# #             # -------------------
# #             st.subheader("Download RDF Data")

# #             # Full RDF download
# #             ttl_bytes = BytesIO(ttl_data.encode("utf-8"))
# #             jsonld_bytes = BytesIO(jsonld_data.encode("utf-8"))
# #             st.download_button("Download Full RDF (Turtle)", data=ttl_bytes, file_name="full_rdf.ttl", mime="text/turtle")
# #             st.download_button("Download Full RDF (JSON-LD)", data=jsonld_bytes, file_name="full_rdf.jsonld", mime="application/ld+json")

# #             # Selected triples download
# #             if not selected_df.empty:
# #                 g_selected = rdflib.Graph()
# #                 for _, row in selected_df.iterrows():
# #                     g_selected.add((rdflib.URIRef(row["Subject"]), rdflib.URIRef(row["Predicate"]), rdflib.URIRef(row["Object"])))
# #                 ttl_selected = g_selected.serialize(format="turtle")
# #                 jsonld_selected = g_selected.serialize(format="json-ld", indent=2)

# #                 ttl_selected_bytes = BytesIO(ttl_selected.encode("utf-8"))
# #                 jsonld_selected_bytes = BytesIO(jsonld_selected.encode("utf-8"))

# #                 st.download_button("Download Selected Triples (Turtle)", data=ttl_selected_bytes, file_name="selected_triples.ttl", mime="text/turtle")
# #                 st.download_button("Download Selected Triples (JSON-LD)", data=jsonld_selected_bytes, file_name="selected_triples.jsonld", mime="application/ld+json")

# #         else:
# #             st.error(f"RDF generation failed: {gen_response.text}")



# # import streamlit as st
# # import requests
# # import json
# # import base64
# # import pandas as pd
# # # ----------------------------
# # # --- STYLING: Background + GIF ---
# # # ----------------------------
# # def get_base64_of_file(file_path):
# #     with open(file_path, "rb") as f:
# #         return base64.b64encode(f.read()).decode()

# # background_path = "./background.png"
# # gif_path = "./srdfgen.gif"

# # base64_background = get_base64_of_file(background_path)
# # encoded_gif = get_base64_of_file(gif_path)

# # st.markdown(f"""
# # <style>
# # html, body, [class*="css"] {{
# #     font-family: 'Inter', sans-serif;
# # }}
# # .stApp {{
# #     background: url("data:image/png;base64,{base64_background}");
# #     background-size: 1000px;
# #     background-position: center;
# #     background-attachment: fixed;
# # }}
# # .top-wrapper {{
# #     display: flex;
# #     flex-direction: column;
# #     align-items: center;
# #     justify-content: center;
# #     margin-top: 30px;
# #     margin-bottom: 40px;
# #     text-align: center;
# # }}
# # .top-wrapper img {{ width: 280px; max-width: 90%; }}
# # .top-wrapper h1 {{
# #     font-family: 'Poppins', sans-serif !important;
# #     font-size: 32px !important;
# #     font-weight: 700 !important;
# #     margin-top: 15px;
# #     color: #1f2937;
# # }}
# # .stButton>button {{
# #     font-family: 'Poppins', sans-serif;
# #     font-weight: 600;
# #     border-radius: 12px;
# #     font-size: 15px;
# #     padding: 10px 16px;
# #     background: linear-gradient(135deg, #ff4b4b, #ff6b6b);
# #     color: white;
# #     border: none;
# # }}
# # </style>

# # <div class="top-wrapper">
# #     <img src="data:image/gif;base64,{encoded_gif}" />
# #     <h1>SHACL → RDF Generator</h1>
# # </div>
# # """, unsafe_allow_html=True)


# # st.title("SHACL-Based Synthetic Data Generator")

# # uploaded_file = st.file_uploader("Upload your SHACL file (.ttl/.shacl)", type=["ttl", "shacl"])

# # if uploaded_file:
# #     st.success(f"Uploaded: {uploaded_file.name}")
# #     files = {"file": (uploaded_file.name, uploaded_file.getvalue())}

# #     try:
# #         schema_resp = requests.post(
# #             "http://localhost:8000/upload_shacl_and_extract_schema",
# #             files=files
# #         )
# #         schema_resp.raise_for_status()
# #         schema = schema_resp.json()
# #     except Exception as e:
# #         st.error(f"Failed to extract schema: {e}")
# #         st.stop()

# #     st.subheader("Schema Tree & Property Configuration")

# #     property_widgets = {}

# #     # Recursive function to display tree
# #     def render_property_tree(prop, idx_prefix=""):
# #         """Displays property in expander, recursively if nested"""
# #         idx = idx_prefix + str(prop.get("index", "0"))
# #         with st.expander(f"{prop['path']}", expanded=True):
# #             model_type = st.selectbox(f"Model Type ({prop['path']})", ["LLM", "VAE", "GAN"], key=f"model_type_{idx}")
# #             model_name = st.text_input(f"Model Name ({prop['path']})", value=prop.get("model_name", ""), key=f"model_name_{idx}")
# #             dist_type = st.selectbox(f"Distribution Type ({prop['path']})", ["categorical", "numeric"], 
# #                                      index=0 if prop.get("distribution_type")=="categorical" else 1, key=f"dist_type_{idx}")
# #             dist_params = {}
# #             if dist_type == "categorical":
# #                 allowed_list = st.text_input(f"Allowed List (comma-separated)", 
# #                                              value=",".join(prop.get("distribution_params", {}).get("allowed_list", [])), key=f"allowed_{idx}")
# #                 probabilities = st.text_input(f"Probabilities (comma-separated)", 
# #                                              value=",".join([str(p) for p in prop.get("distribution_params", {}).get("probabilities", [])]), key=f"probs_{idx}")
# #                 dist_params["allowed_list"] = [x.strip() for x in allowed_list.split(",") if x.strip()]
# #                 dist_params["probabilities"] = [float(x) for x in probabilities.split(",") if x.strip()]
# #             else:
# #                 min_val = st.number_input(f"Min", value=prop.get("distribution_params", {}).get("min", 0), key=f"min_{idx}")
# #                 max_val = st.number_input(f"Max", value=prop.get("distribution_params", {}).get("max", 10), key=f"max_{idx}")
# #                 mean_val = st.number_input(f"Mean", value=prop.get("distribution_params", {}).get("mean", 5), key=f"mean_{idx}")
# #                 std_val = st.number_input(f"Std", value=prop.get("distribution_params", {}).get("std", 1), key=f"std_{idx}")
# #                 dist_params = {"min": min_val, "max": max_val, "mean": mean_val, "std": std_val}

# #             # Save config
# #             property_widgets[idx] = {
# #                 "path": prop["path"],
# #                 "datatype": prop.get("datatype", "http://www.w3.org/2001/XMLSchema#string"),
# #                 "min_count": prop.get("min_count", 1),
# #                 "max_count": prop.get("max_count", 1),
# #                 "model_type": model_type,
# #                 "model_name": model_name,
# #                 "distribution_type": dist_type,
# #                 "distribution_params": dist_params
# #             }

# #             # Recursively render nested properties if any
# #             for child_idx, child in enumerate(prop.get("children", [])):
# #                 child["index"] = f"{idx}_{child_idx}"
# #                 render_property_tree(child, idx_prefix=idx+"_")

# #     # Render top-level properties
# #     for i, prop in enumerate(schema):
# #         prop["index"] = str(i)
# #         render_property_tree(prop)

# #     num_samples = st.number_input("Number of samples per property", min_value=1, max_value=100, value=3)

# #     if st.button("Generate Data"):
# #         payload = [widget.copy() for idx, widget in property_widgets.items()]
# #         try:
# #             gen_resp = requests.post("http://localhost:8000/generate_from_shacl", json=payload)
# #             gen_resp.raise_for_status()
# #             st.success("Data generated successfully!")
# #             st.json(gen_resp.json())
# #         except Exception as e:
# #             st.error(f"Failed to generate data: {e}")


# # import streamlit as st
# # import requests
# # import json

# # st.title("SHACL-Based Synthetic Data Generator")

# # # ------------------------------
# # # Upload SHACL file
# # # ------------------------------
# # uploaded_file = st.file_uploader("Upload your SHACL file (.ttl/.shacl)", type=["ttl", "shacl"])

# # if uploaded_file:
# #     st.success(f"Uploaded: {uploaded_file.name}")
# #     files = {"file": (uploaded_file.name, uploaded_file.getvalue())}

# #     try:
# #         schema_resp = requests.post(
# #             "http://localhost:8000/upload_shacl_and_extract_schema",
# #             files=files
# #         )
# #         schema_resp.raise_for_status()
# #         # Access the actual list of properties
# #         schema = schema_resp.json().get("json_schema", [])
# #     except Exception as e:
# #         st.error(f"Failed to extract schema: {e}")
# #         st.stop()

# #     # ------------------------------
# #     # Convert numeric strings to floats
# #     # ------------------------------
# #     for prop in schema:
# #         dist_params = prop.get("distribution_params", {})
# #         if prop.get("distribution_type") == "categorical":
# #             probs = dist_params.get("probabilities", [])
# #             dist_params["probabilities"] = [float(p) for p in probs]
# #         elif prop.get("distribution_type") == "numeric":
# #             for key in ["min", "max", "mean", "std"]:
# #                 if key in dist_params:
# #                     dist_params[key] = float(dist_params[key])
# #         prop["distribution_params"] = dist_params

# #     st.subheader("Schema Tree & Property Configuration")

# #     property_widgets = {}

# #     # ------------------------------
# #     # Recursive function to render tree
# #     # ------------------------------
# #     def render_property_tree(prop, idx_prefix=""):
# #         """Displays property in expander, recursively if nested"""
# #         idx = idx_prefix + str(prop.get("index", "0"))
# #         with st.expander(f"{prop['path']}", expanded=True):
# #             # Model type & name
# #             model_type = st.selectbox(f"Model Type ({prop['path']})", ["LLM", "VAE", "GAN"], 
# #                                       index=["LLM","VAE","GAN"].index(prop.get("model_type","LLM")),
# #                                       key=f"model_type_{idx}")
# #             model_name = st.text_input(f"Model Name ({prop['path']})", 
# #                                        value=prop.get("model_name", ""), key=f"model_name_{idx}")

# #             # Distribution type
# #             dist_type = st.selectbox(f"Distribution Type ({prop['path']})", ["categorical", "numeric"], 
# #                                      index=0 if prop.get("distribution_type")=="categorical" else 1, key=f"dist_type_{idx}")
            
# #             # Distribution parameters
# #             dist_params = {}
# #             if dist_type == "categorical":
# #                 allowed_list = st.text_input(f"Allowed List (comma-separated)", 
# #                                              value=",".join(prop.get("distribution_params", {}).get("allowed_list", [])), key=f"allowed_{idx}")
# #                 probabilities = st.text_input(f"Probabilities (comma-separated)", 
# #                                              value=",".join([str(p) for p in prop.get("distribution_params", {}).get("probabilities", [])]), key=f"probs_{idx}")
# #                 dist_params["allowed_list"] = [x.strip() for x in allowed_list.split(",") if x.strip()]
# #                 dist_params["probabilities"] = [float(x) for x in probabilities.split(",") if x.strip()]
# #             else:
# #                 min_val = st.number_input(f"Min", value=prop.get("distribution_params", {}).get("min", 0), key=f"min_{idx}")
# #                 max_val = st.number_input(f"Max", value=prop.get("distribution_params", {}).get("max", 10), key=f"max_{idx}")
# #                 mean_val = st.number_input(f"Mean", value=prop.get("distribution_params", {}).get("mean", 5), key=f"mean_{idx}")
# #                 std_val = st.number_input(f"Std", value=prop.get("distribution_params", {}).get("std", 1), key=f"std_{idx}")
# #                 dist_params = {"min": min_val, "max": max_val, "mean": mean_val, "std": std_val}

# #             # Save config
# #             property_widgets[idx] = {
# #                 "path": prop["path"],
# #                 "datatype": prop.get("datatype", "http://www.w3.org/2001/XMLSchema#string"),
# #                 "min_count": prop.get("min_count", 1),
# #                 "max_count": prop.get("max_count", 1),
# #                 "model_type": model_type,
# #                 "model_name": model_name,
# #                 "distribution_type": dist_type,
# #                 "distribution_params": dist_params
# #             }

# #             # Recursively render children if any
# #             for child_idx, child in enumerate(prop.get("children", [])):
# #                 child["index"] = f"{idx}_{child_idx}"
# #                 render_property_tree(child, idx_prefix=idx+"_")

# #     # ------------------------------
# #     # Render top-level properties
# #     # ------------------------------
# #     for i, prop in enumerate(schema):
# #         prop["index"] = str(i)
# #         render_property_tree(prop)

# #     # ------------------------------
# #     # Number of samples input
# #     # ------------------------------
# #     num_samples = st.number_input("Number of samples per property", min_value=1, max_value=100, value=3)

# #     # ------------------------------
# #     # Generate button
# #     # ------------------------------
# #     if st.button("Generate Data"):
# #         # Add num_samples to each property payload
# #         payload = []
# #         for idx, widget in property_widgets.items():
# #             prop_copy = widget.copy()
# #             prop_copy["num_samples"] = num_samples
# #             payload.append(prop_copy)
# #         try:
# #             gen_resp = requests.post("http://localhost:8000/generate_from_shacl", json=payload)
# #             gen_resp.raise_for_status()
# #             st.success("Data generated successfully!")
# #             st.json(gen_resp.json())
# #         except Exception as e:
# #             st.error(f"Failed to generate data: {e}")


# # import streamlit as st
# # import requests
# # import json

# # st.title("SHACL-Based Synthetic Data Generator")

# # # ------------------------------
# # # Upload SHACL file
# # # ------------------------------
# # uploaded_file = st.file_uploader("Upload your SHACL file (.ttl/.shacl)", type=["ttl", "shacl"])

# # if uploaded_file:
# #     st.success(f"Uploaded: {uploaded_file.name}")
# #     files = {"file": (uploaded_file.name, uploaded_file.getvalue())}

# #     try:
# #         schema_resp = requests.post(
# #             "http://localhost:8000/upload_shacl_and_extract_schema",
# #             files=files
# #         )
# #         schema_resp.raise_for_status()
# #         schema = schema_resp.json().get("json_schema", [])
# #     except Exception as e:
# #         st.error(f"Failed to extract schema: {e}")
# #         st.stop()

# #     # Convert numeric strings to floats
# #     for prop in schema:
# #         dist_params = prop.get("distribution_params", {})
# #         if prop.get("distribution_type") == "categorical":
# #             probs = dist_params.get("probabilities", [])
# #             dist_params["probabilities"] = [float(p) for p in probs]
# #         elif prop.get("distribution_type") == "numeric":
# #             for key in ["min", "max", "mean", "std"]:
# #                 if key in dist_params:
# #                     dist_params[key] = float(dist_params[key])
# #         prop["distribution_params"] = dist_params

# #     st.subheader("Schema Tree & Property Configuration")

# #     property_widgets = {}

# #     # ------------------------------
# #     # Recursive tree renderer
# #     # ------------------------------
# #     def render_property_tree(prop, idx_prefix=""):
# #         idx = idx_prefix + str(prop.get("index", "0"))
# #         with st.expander(f"{prop['path']}", expanded=True):
# #             # Model Type selection
# #             model_type = st.selectbox(
# #                 f"Model Type ({prop['path']})",
# #                 ["LLM", "VAE", "GAN"],
# #                 index=["LLM", "VAE", "GAN"].index(prop.get("model_type", "LLM")),
# #                 key=f"model_type_{idx}"
# #             )

# #             # Model Name selection: fetch dynamically for VAE/GAN, skip for LLM
# #             model_name = ""
# #             if model_type in ["VAE", "GAN"]:
# #                 try:
# #                     resp = requests.get(f"http://localhost:8000/models", params={"model_type": model_type})
# #                     resp.raise_for_status()
# #                     available_models = resp.json()
# #                 except Exception as e:
# #                     st.error(f"Failed to fetch {model_type} models: {e}")
# #                     available_models = []

# #                 model_name = st.selectbox(
# #                     f"Model Name ({prop['path']})",
# #                     options=available_models,
# #                     index=0 if available_models else -1,
# #                     key=f"model_name_{idx}"
# #                 )

# #             # Distribution type
# #             dist_type = st.selectbox(
# #                 f"Distribution Type ({prop['path']})",
# #                 ["categorical", "numeric"],
# #                 index=0 if prop.get("distribution_type")=="categorical" else 1,
# #                 key=f"dist_type_{idx}"
# #             )

# #             # Distribution params
# #             dist_params = {}
# #             if dist_type == "categorical":
# #                 allowed_list = st.text_input(
# #                     f"Allowed List (comma-separated)",
# #                     value=",".join(prop.get("distribution_params", {}).get("allowed_list", [])),
# #                     key=f"allowed_{idx}"
# #                 )
# #                 probabilities = st.text_input(
# #                     f"Probabilities (comma-separated)",
# #                     value=",".join([str(p) for p in prop.get("distribution_params", {}).get("probabilities", [])]),
# #                     key=f"probs_{idx}"
# #                 )
# #                 dist_params["allowed_list"] = [x.strip() for x in allowed_list.split(",") if x.strip()]
# #                 dist_params["probabilities"] = [float(x) for x in probabilities.split(",") if x.strip()]
# #             else:
# #                 min_val = st.number_input(
# #                     f"Min", value=prop.get("distribution_params", {}).get("min", 0), key=f"min_{idx}"
# #                 )
# #                 max_val = st.number_input(
# #                     f"Max", value=prop.get("distribution_params", {}).get("max", 10), key=f"max_{idx}"
# #                 )
# #                 mean_val = st.number_input(
# #                     f"Mean", value=prop.get("distribution_params", {}).get("mean", 5), key=f"mean_{idx}"
# #                 )
# #                 std_val = st.number_input(
# #                     f"Std", value=prop.get("distribution_params", {}).get("std", 1), key=f"std_{idx}"
# #                 )
# #                 dist_params = {"min": min_val, "max": max_val, "mean": mean_val, "std": std_val}

# #             # Save widget config
# #             property_widgets[idx] = {
# #                 "path": prop["path"],
# #                 "datatype": prop.get("datatype", "http://www.w3.org/2001/XMLSchema#string"),
# #                 "min_count": prop.get("min_count", 1),
# #                 "max_count": prop.get("max_count", 1),
# #                 "model_type": model_type,
# #                 "model_name": model_name,
# #                 "distribution_type": dist_type,
# #                 "distribution_params": dist_params
# #             }

# #             # Recursively render children
# #             for child_idx, child in enumerate(prop.get("children", [])):
# #                 child["index"] = f"{idx}_{child_idx}"
# #                 render_property_tree(child, idx_prefix=idx+"_")

# #     # ------------------------------
# #     # Render top-level properties
# #     # ------------------------------
# #     for i, prop in enumerate(schema):
# #         prop["index"] = str(i)
# #         render_property_tree(prop)

# #     # ------------------------------
# #     # Number of samples input
# #     # ------------------------------
# #     num_samples = st.number_input("Number of samples per property", min_value=1, max_value=100, value=3)

# #     # ------------------------------
# #     # Generate button
# #     # ------------------------------
# #     if st.button("Generate Data"):
# #         payload = []
# #         for idx, widget in property_widgets.items():
# #             prop_copy = widget.copy()
# #             prop_copy["num_samples"] = num_samples
# #             payload.append(prop_copy)
# #         try:
# #             gen_resp = requests.post("http://localhost:8000/generate_from_shacl", json=payload)
# #             gen_resp.raise_for_status()
# #             st.success("Data generated successfully!")
# #             st.json(gen_resp.json())
# #         except Exception as e:
# #             st.error(f"Failed to generate data: {e}")


# # import streamlit as st
# # import requests
# # import json

# # st.title("SHACL-Based Synthetic Data Generator")

# # # ---------------------------
# # # Upload SHACL file
# # # ---------------------------
# # uploaded_file = st.file_uploader("Upload your SHACL file (.ttl/.shacl)", type=["ttl", "shacl"])

# # if uploaded_file:
# #     st.success(f"Uploaded: {uploaded_file.name}")
# #     files = {"file": (uploaded_file.name, uploaded_file.getvalue())}

# #     try:
# #         schema_resp = requests.post(
# #             "http://localhost:8000/upload_shacl_and_extract_schema",
# #             files=files
# #         )
# #         schema_resp.raise_for_status()
# #         schema = schema_resp.json().get("json_schema", [])
# #     except Exception as e:
# #         st.error(f"Failed to extract schema: {e}")
# #         st.stop()

# #     st.subheader("Schema Tree & Property Configuration")

# #     property_widgets = {}

# #     # ---------------------------
# #     # Recursive function to render properties
# #     # ---------------------------
# #     def render_property_tree(prop, idx_prefix=""):
# #         idx = idx_prefix + str(prop.get("index", "0"))
# #         with st.expander(f"{prop['path']}", expanded=True):

# #             # Model type selection
# #             model_type = st.selectbox(
# #                 f"Model Type ({prop['path']})",
# #                 ["LLM", "VAE", "GAN"],
# #                 key=f"model_type_{idx}"
# #             )

# #             # Fetch available models for VAE/GAN
# #             model_name = ""
# #             if model_type in ["VAE", "GAN"]:
# #                 try:
# #                     models_resp = requests.get(f"http://localhost:8000/models?model_type={model_type}")
# #                     models_resp.raise_for_status()
# #                     available_models = models_resp.json()
# #                 except Exception as e:
# #                     available_models = []
# #                     st.warning(f"Failed to fetch models: {e}")

# #                 if available_models:
# #                     model_name = st.selectbox(
# #                         f"Select {model_type} model for {prop['path']}",
# #                         available_models,
# #                         key=f"model_name_{idx}"
# #                     )
# #                 else:
# #                     model_name = st.text_input(
# #                         f"Enter {model_type} model name manually",
# #                         value="",
# #                         key=f"model_name_{idx}"
# #                     )

# #             # Distribution type
# #             dist_type = st.selectbox(
# #                 f"Distribution Type ({prop['path']})",
# #                 ["categorical", "numeric"],
# #                 index=0 if prop.get("distribution_type")=="categorical" else 1,
# #                 key=f"dist_type_{idx}"
# #             )

# #             # Distribution parameters
# #             dist_params = {}
# #             if dist_type == "categorical":
# #                 allowed_list = st.text_input(
# #                     "Allowed List (comma-separated)",
# #                     value=",".join(prop.get("distribution_params", {}).get("allowed_list", [])),
# #                     key=f"allowed_{idx}"
# #                 )
# #                 probabilities = st.text_input(
# #                     "Probabilities (comma-separated)",
# #                     value=",".join([str(p) for p in prop.get("distribution_params", {}).get("probabilities", [])]),
# #                     key=f"probs_{idx}"
# #                 )
# #                 dist_params["allowed_list"] = [x.strip() for x in allowed_list.split(",") if x.strip()]
# #                 dist_params["probabilities"] = [float(x) for x in probabilities.split(",") if x.strip()]
# #             else:
# #                 min_val = st.number_input(
# #                     "Min",
# #                     value=float(prop.get("distribution_params", {}).get("min", 0)),
# #                     key=f"min_{idx}"
# #                 )
# #                 max_val = st.number_input(
# #                     "Max",
# #                     value=float(prop.get("distribution_params", {}).get("max", 10)),
# #                     key=f"max_{idx}"
# #                 )
# #                 mean_val = st.number_input(
# #                     "Mean",
# #                     value=float(prop.get("distribution_params", {}).get("mean", 5)),
# #                     key=f"mean_{idx}"
# #                 )
# #                 std_val = st.number_input(
# #                     "Std",
# #                     value=float(prop.get("distribution_params", {}).get("std", 1)),
# #                     key=f"std_{idx}"
# #                 )
# #                 dist_params = {"min": min_val, "max": max_val, "mean": mean_val, "std": std_val}

# #             # Save property configuration
# #             property_widgets[idx] = {
# #                 "path": prop["path"],
# #                 "datatype": prop.get("datatype", "http://www.w3.org/2001/XMLSchema#string"),
# #                 "min_count": prop.get("min_count", 1),
# #                 "max_count": prop.get("max_count", 1),
# #                 "model_type": model_type,
# #                 "model_name": model_name,
# #                 "distribution_type": dist_type,
# #                 "distribution_params": dist_params
# #             }

# #             # Render children recursively if any
# #             for child_idx, child in enumerate(prop.get("children", [])):
# #                 child["index"] = f"{idx}_{child_idx}"
# #                 render_property_tree(child, idx_prefix=idx+"_")

# #     # Assign index to top-level properties and render tree
# #     for i, prop in enumerate(schema):
# #         prop["index"] = str(i)
# #         render_property_tree(prop)

# #     # ---------------------------
# #     # Number of samples input
# #     # ---------------------------
# #     num_samples = st.number_input(
# #         "Number of samples per property",
# #         min_value=1,
# #         max_value=100,
# #         value=3
# #     )

# #     # ---------------------------
# #     # Generate button
# #     # ---------------------------
# #     if st.button("Generate Data"):
# #         payload = [widget.copy() for idx, widget in property_widgets.items()]

# #         # Update min_count/max_count based on user input for number of samples
# #         for p in payload:
# #             p["min_count"] = num_samples
# #             p["max_count"] = num_samples

# #         try:
# #             gen_resp = requests.post(
# #                 "http://localhost:8000/generate_from_shacl",
# #                 json=payload
# #             )
# #             gen_resp.raise_for_status()
# #             st.success("Data generated successfully!")
# #             st.json(gen_resp.json())
# #         except Exception as e:
# #             st.error(f"Failed to generate data: {e}")


# # import streamlit as st
# # import requests
# # from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode

# # st.title("SHACL-Based Synthetic Data Generator – Tree View")

# # # ---------------------------
# # # Upload SHACL file
# # # ---------------------------
# # uploaded_file = st.file_uploader("Upload your SHACL file (.ttl/.shacl)", type=["ttl","shacl"])

# # if uploaded_file:
# #     st.success(f"Uploaded: {uploaded_file.name}")
# #     files = {"file": (uploaded_file.name, uploaded_file.getvalue())}

# #     try:
# #         schema_resp = requests.post(
# #             "http://localhost:8000/upload_shacl_and_extract_schema",
# #             files=files
# #         )
# #         schema_resp.raise_for_status()
# #         schema = schema_resp.json().get("json_schema", [])
# #     except Exception as e:
# #         st.error(f"Failed to extract schema: {e}")
# #         st.stop()

# #     # ---------------------------
# #     # Flatten schema for tree table
# #     # ---------------------------
# #     def flatten_schema(schema, parent_path=""):
# #         flat = []
# #         for i, prop in enumerate(schema):
# #             path_key = f"{parent_path}/{i}" if parent_path else str(i)
# #             flat.append({
# #                 "index": path_key,
# #                 "path": prop["path"],
# #                 "datatype": prop.get("datatype", "http://www.w3.org/2001/XMLSchema#string"),
# #                 "min_count": prop.get("min_count", 1),
# #                 "max_count": prop.get("max_count", 1),
# #                 "model_type": prop.get("model_type", "LLM"),
# #                 "model_name": prop.get("model_name", ""),
# #                 "distribution_type": prop.get("distribution_type", "categorical"),
# #                 "distribution_params": prop.get("distribution_params", {})
# #             })
# #             children = prop.get("children", [])
# #             if children:
# #                 flat.extend(flatten_schema(children, path_key))
# #         return flat

# #     # flat_schema = flatten_schema(schema)

# #     # # ---------------------------
# #     # # Fetch available models
# #     # # ---------------------------
# #     # try:
# #     #     vae_models = requests.get("http://localhost:8000/models?model_type=VAE").json()
# #     #     gan_models = requests.get("http://localhost:8000/models?model_type=GAN").json()
# #     # except Exception:
# #     #     vae_models = []
# #     #     gan_models = []

# #     # # ---------------------------
# #     # # Configure AgGrid
# #     # # ---------------------------
# #     # gb = GridOptionsBuilder.from_dataframe(pd.DataFrame(flat_schema))
# #     # gb.configure_column("index", editable=False)
# #     # gb.configure_column("path", editable=False)
# #     # gb.configure_column("datatype", editable=False)
# #     # gb.configure_column("model_type", editable=True, cellEditor="agSelectCellEditor",
# #     #                     cellEditorParams={"values":["LLM","VAE","GAN"]})
# #     # gb.configure_column("model_name", editable=True)
# #     # gb.configure_column("distribution_type", editable=True, cellEditor="agSelectCellEditor",
# #     #                     cellEditorParams={"values":["categorical","numeric"]})
# #     # gb.configure_column("distribution_params", editable=True)
# #     # gb.configure_column("min_count", editable=True)
# #     # gb.configure_column("max_count", editable=True)
# #     # gb.configure_selection(selection_mode="multiple", use_checkbox=True)
# #     # grid_options = gb.build()

# #     # grid_response = AgGrid(
# #     #     pd.DataFrame(flat_schema),
# #     #     gridOptions=grid_options,
# #     #     height=400,
# #     #     data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
# #     #     update_mode=GridUpdateMode.VALUE_CHANGED,
# #     #     fit_columns_on_grid_load=True
# #     # )

# #     # # ---------------------------
# #     # # Number of samples
# #     # # ---------------------------
# #     # num_samples = st.number_input("Number of samples per property", min_value=1, max_value=100, value=3)

# #     # if st.button("Generate Data"):
# #     #     # Prepare payload from AgGrid
# #     #     payload = grid_response["data"].to_dict("records")
# #     #     for p in payload:
# #     #         p["min_count"] = num_samples
# #     #         p["max_count"] = num_samples

# #     #         # Auto-select model_name for VAE/GAN if empty
# #     #         if p["model_type"] == "VAE" and not p["model_name"] and vae_models:
# #     #             p["model_name"] = vae_models[0]
# #     #         if p["model_type"] == "GAN" and not p["model_name"] and gan_models:
# #     #             p["model_name"] = gan_models[0]

# #     #     try:
# #     #         gen_resp = requests.post("http://localhost:8000/generate_from_shacl", json=payload)
# #     #         gen_resp.raise_for_status()
# #     #         st.success("Data generated successfully!")
# #     #         st.json(gen_resp.json())
# #     #     except Exception as e:
# #     #         st.error(f"Failed to generate data: {e}")

# #     import pandas as pd
# #     from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# #     # Flatten schema for table display
# #     flat_schema = []
# #     for i, prop in enumerate(schema):
# #         flat_prop = {
# #             "index": i,
# #             "path": prop["path"],
# #             "datatype": prop.get("datatype", "http://www.w3.org/2001/XMLSchema#string"),
# #             "model_type": prop.get("model_type", "LLM"),
# #             "model_name": prop.get("model_name", ""),
# #             "distribution_type": prop.get("distribution_type", "categorical"),
# #             "min_count": prop.get("min_count", 1),
# #             "max_count": prop.get("max_count", 1),
# #         }

# #         # Unpack distribution parameters
# #         dist_params = prop.get("distribution_params", {})
# #         if flat_prop["distribution_type"] == "categorical":
# #             flat_prop["allowed_list"] = ",".join(dist_params.get("allowed_list", []))
# #             flat_prop["probabilities"] = ",".join([str(p) for p in dist_params.get("probabilities", [])])
# #             flat_prop["min"] = flat_prop["max"] = flat_prop["mean"] = flat_prop["std"] = None
# #         else:  # numeric
# #             flat_prop["min"] = dist_params.get("min", 0)
# #             flat_prop["max"] = dist_params.get("max", 10)
# #             flat_prop["mean"] = dist_params.get("mean", 5)
# #             flat_prop["std"] = dist_params.get("std", 1)
# #             flat_prop["allowed_list"] = flat_prop["probabilities"] = None

# #         flat_schema.append(flat_prop)

# #     # Build AgGrid
# #     df = pd.DataFrame(flat_schema)
# #     gb = GridOptionsBuilder.from_dataframe(df)

# #     # Configure columns
# #     gb.configure_column("index", editable=False, width=50)
# #     gb.configure_column("path", editable=False, width=300)
# #     gb.configure_column("datatype", editable=False, width=180)

# #     gb.configure_column("model_type", editable=True, cellEditor="agSelectCellEditor",
# #                         cellEditorParams={"values":["LLM","VAE","GAN"]}, width=100)
# #     gb.configure_column("model_name", editable=True, width=150)

# #     gb.configure_column("distribution_type", editable=True, cellEditor="agSelectCellEditor",
# #                         cellEditorParams={"values":["categorical","numeric"]}, width=120)

# #     # Categorical columns
# #     gb.configure_column("allowed_list", editable=True, width=200)
# #     gb.configure_column("probabilities", editable=True, width=150)

# #     # Numeric columns
# #     gb.configure_column("min", editable=True, width=80)
# #     gb.configure_column("max", editable=True, width=80)
# #     gb.configure_column("mean", editable=True, width=80)
# #     gb.configure_column("std", editable=True, width=80)

# #     gb.configure_column("min_count", editable=True, width=90)
# #     gb.configure_column("max_count", editable=True, width=90)

# #     gb.configure_selection(selection_mode="multiple", use_checkbox=True)
# #     grid_options = gb.build()

# #     # Render grid
# #     grid_response = AgGrid(
# #         df,
# #         gridOptions=grid_options,
# #         height=600,
# #         width='100%',
# #         fit_columns_on_grid_load=False,
# #         update_mode=GridUpdateMode.VALUE_CHANGED,
# #         data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
# #         allow_unsafe_jscode=True,
# #         theme="alpine"
# #     )

# #     # After edits, reconstruct payload for /generate_from_shacl
# #     updated_df = grid_response["data"]
# #     payload = []

# #     for _, row in updated_df.iterrows():
# #         dist_type = row["distribution_type"]
# #         dist_params = {}
# #         if dist_type == "categorical":
# #             dist_params["allowed_list"] = [x.strip() for x in row["allowed_list"].split(",") if x.strip()]
# #             dist_params["probabilities"] = [float(x) for x in row["probabilities"].split(",") if x.strip()]
# #         else:
# #             dist_params = {
# #                 "min": row["min"],
# #                 "max": row["max"],
# #                 "mean": row["mean"],
# #                 "std": row["std"]
# #             }

# #         payload.append({
# #             "path": row["path"],
# #             "datatype": row["datatype"],
# #             "min_count": row["min_count"],
# #             "max_count": row["max_count"],
# #             "model_type": row["model_type"],
# #             "model_name": row["model_name"],
# #             "distribution_type": dist_type,
# #             "distribution_params": dist_params
# #         })

# #     # Send payload to /generate_from_shacl
# #     if st.button("Generate Data"):
# #         try:
# #             gen_resp = requests.post("http://localhost:8000/generate_from_shacl", json=payload)
# #             gen_resp.raise_for_status()
# #             st.success("Data generated successfully!")
# #             st.json(gen_resp.json())
# #         except Exception as e:
# #             st.error(f"Failed to generate data: {e}")


# # import streamlit as st
# # import requests
# # from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
# # import pandas as pd

# # st.set_page_config(page_title="SHACL Data Generator", layout="wide")
# # st.title("SHACL-Based Synthetic Data Generator")

# # # ---------------------------
# # # Upload SHACL File
# # # ---------------------------
# # uploaded_file = st.file_uploader("Upload SHACL file (.ttl/.shacl)", type=["ttl", "shacl"])

# # if uploaded_file:
# #     st.success(f"Uploaded: {uploaded_file.name}")
# #     files = {"file": (uploaded_file.name, uploaded_file.getvalue())}

# #     try:
# #         schema_resp = requests.post(
# #             "http://localhost:8000/upload_shacl_and_extract_schema",
# #             files=files
# #         )
# #         schema_resp.raise_for_status()
# #         schema = schema_resp.json()["json_schema"]
# #     except Exception as e:
# #         st.error(f"Failed to extract schema: {e}")
# #         st.stop()

# #     # ---------------------------
# #     # Convert schema to DataFrame for AgGrid
# #     # ---------------------------
# #     df = pd.DataFrame(schema)

# #     # Add placeholder columns for model_type and model_name
# #     if "model_type" not in df.columns:
# #         df["model_type"] = "LLM"
# #     if "model_name" not in df.columns:
# #         df["model_name"] = ""

# #     # ---------------------------
# #     # AgGrid Options
# #     # ---------------------------
# #     gb = GridOptionsBuilder.from_dataframe(df)

# #     # model_type: dropdown
# #     gb.configure_column(
# #         "model_type",
# #         editable=True,
# #         cellEditor="agSelectCellEditor",
# #         cellEditorParams={"values": ["LLM", "VAE", "GAN"]},
# #     )

# #     # model_name: will populate dynamically
# #     gb.configure_column(
# #         "model_name",
# #         editable=True,
# #         cellEditor="agSelectCellEditor",
# #         cellEditorParams={"values": []},  # updated dynamically
# #     )

# #     # Enable row selection and editing
# #     gb.configure_selection("single")
# #     gb.configure_default_column(editable=True)
# #     grid_options = gb.build()

# #     grid_response = AgGrid(
# #         df,
# #         gridOptions=grid_options,
# #         update_mode=GridUpdateMode.VALUE_CHANGED,
# #         data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
# #         fit_columns_on_grid_load=True,
# #         allow_unsafe_jscode=True,
# #     )

# #     updated_df = grid_response["data"]

# #     # ---------------------------
# #     # Dynamic model_name update
# #     # ---------------------------
# #     for i, row in updated_df.iterrows():
# #         model_type = row["model_type"]
# #         if model_type in ["VAE", "GAN"]:
# #             try:
# #                 resp = requests.get(f"http://localhost:8000/models?model_type={model_type}")
# #                 resp.raise_for_status()
# #                 models = resp.json()
# #             except:
# #                 models = []

# #             updated_df.at[i, "model_name"] = models[0] if models else ""
# #         else:
# #             updated_df.at[i, "model_name"] = ""

# #     st.subheader("Adjusted Schema & Configuration")
# #     st.dataframe(updated_df)

# #     # ---------------------------
# #     # Generate Data Button
# #     # ---------------------------
# #     num_samples = st.number_input("Number of samples per property", min_value=1, max_value=100, value=3)

# #     if st.button("Generate Data"):
# #         # Prepare payload for backend
# #         payload = []
# #         for _, row in updated_df.iterrows():
# #             dist_params = row.get("distribution_params", {})
# #             payload.append({
# #                 "path": row["path"],
# #                 "datatype": row.get("datatype", "http://www.w3.org/2001/XMLSchema#string"),
# #                 "min_count": row.get("min_count", 1),
# #                 "max_count": row.get("max_count", 1),
# #                 "model_type": row["model_type"],
# #                 "model_name": row["model_name"],
# #                 "distribution_type": row.get("distribution_type", "categorical"),
# #                 "distribution_params": dist_params
# #             })

# #         try:
# #             gen_resp = requests.post("http://localhost:8000/generate_from_shacl", json=payload)
# #             gen_resp.raise_for_status()
# #             st.success("Data generated successfully!")
# #             st.json(gen_resp.json())
# #         except Exception as e:
# #             st.error(f"Failed to generate data: {e}")


# # import streamlit as st
# # import requests
# # from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
# # import pandas as pd

# # st.set_page_config(page_title="SHACL Data Generator", layout="wide")
# # st.title("SHACL-Based Synthetic Data Generator")

# # # ---------------------------
# # # Upload SHACL File
# # # ---------------------------
# # uploaded_file = st.file_uploader("Upload SHACL file (.ttl/.shacl)", type=["ttl", "shacl"])

# # if uploaded_file:
# #     st.success(f"Uploaded: {uploaded_file.name}")
# #     files = {"file": (uploaded_file.name, uploaded_file.getvalue())}

# #     try:
# #         schema_resp = requests.post(
# #             "http://localhost:8000/upload_shacl_and_extract_schema",
# #             files=files
# #         )
# #         schema_resp.raise_for_status()
# #         schema = schema_resp.json()["json_schema"]
# #     except Exception as e:
# #         st.error(f"Failed to extract schema: {e}")
# #         st.stop()

# #     # ---------------------------
# #     # Convert schema to DataFrame for AgGrid
# #     # ---------------------------
# #     df = pd.DataFrame(schema)

# #     # Add placeholder columns
# #     if "model_type" not in df.columns:
# #         df["model_type"] = "LLM"
# #     if "model_name" not in df.columns:
# #         df["model_name"] = ""

# #     # ---------------------------
# #     # Fetch available models
# #     # ---------------------------
# #     try:
# #         vae_models = requests.get("http://localhost:8000/models?model_type=VAE").json()
# #     except:
# #         vae_models = []

# #     try:
# #         gan_models = requests.get("http://localhost:8000/models?model_type=GAN").json()
# #     except:
# #         gan_models = []

# #     # Combine all models for dropdown
# #     all_models = vae_models + gan_models

# #     # ---------------------------
# #     # AgGrid Options
# #     # ---------------------------
# #     gb = GridOptionsBuilder.from_dataframe(df)

# #     # model_type dropdown
# #     gb.configure_column(
# #         "model_type",
# #         editable=True,
# #         cellEditor="agSelectCellEditor",
# #         cellEditorParams={"values": ["LLM", "VAE", "GAN"]},
# #     )

# #     # model_name dropdown
# #     # IMPORTANT: pass the Python list directly, not a string
# #     gb.configure_column(
# #         "model_name",
# #         editable=True,
# #         cellEditor="agSelectCellEditor",
# #         cellEditorParams={"values": all_models},  # must be a Python list
# #     )

# #     gb.configure_selection("single")
# #     gb.configure_default_column(editable=True)
# #     grid_options = gb.build()

# #     # ---------------------------
# #     # Render AgGrid
# #     # ---------------------------
# #     grid_response = AgGrid(
# #         df,
# #         gridOptions=grid_options,
# #         update_mode=GridUpdateMode.VALUE_CHANGED,
# #         data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
# #         fit_columns_on_grid_load=True,
# #         allow_unsafe_jscode=True,
# #     )

# #     updated_df = grid_response["data"]

# #     st.subheader("Adjusted Schema & Configuration")
# #     st.dataframe(updated_df)

# #     # ---------------------------
# #     # Generate Data
# #     # ---------------------------
# #     num_samples = st.number_input("Number of samples per property", min_value=1, max_value=100, value=3)

# #     if st.button("Generate Data"):
# #         payload = []
# #         for _, row in updated_df.iterrows():
# #             dist_params = row.get("distribution_params", {})
# #             payload.append({
# #                 "path": row["path"],
# #                 "datatype": row.get("datatype", "http://www.w3.org/2001/XMLSchema#string"),
# #                 "min_count": row.get("min_count", 1),
# #                 "max_count": row.get("max_count", 1),
# #                 "model_type": row["model_type"],
# #                 "model_name": row["model_name"],
# #                 "distribution_type": row.get("distribution_type", "categorical"),
# #                 "distribution_params": dist_params
# #             })

# #         try:
# #             gen_resp = requests.post("http://localhost:8000/generate_from_shacl", json=payload)
# #             gen_resp.raise_for_status()
# #             st.success("Data generated successfully!")
# #             st.json(gen_resp.json())
# #         except Exception as e:
# #             st.error(f"Failed to generate data: {e}")



# import streamlit as st
# import requests
# import base64
# from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
# import pandas as pd

# def get_base64_of_file(file_path):
#     with open(file_path, "rb") as f:
#         return base64.b64encode(f.read()).decode()

# background_path = "./background.png"
# gif_path = "./srdfgen.gif"

# base64_background = get_base64_of_file(background_path)
# encoded_gif = get_base64_of_file(gif_path)

# st.markdown("""
# <link href="https://fonts.googleapis.com/css2?family=Lato:wght@400;700&family=Raleway:wght@600;700&display=swap" rel="stylesheet">
# """, unsafe_allow_html=True)

# st.markdown(f"""
# <style>
# html, body, [class*="css"] {{
#     font-family: 'Lato', sans-serif;
# }}
# .stApp {{
#     background: url("data:image/png;base64,{base64_background}");
#     background-size: 1000px;
#     background-position: center;
#     background-attachment: fixed;
# }}
# .top-wrapper {{
#     display: flex;
#     flex-direction: column;
#     align-items: center;
#     justify-content: center;
#     margin-top: 30px;
#     margin-bottom: 40px;
#     text-align: center;
# }}
# .top-wrapper img {{ width: 280px; max-width: 90%; }}
# .top-wrapper h1 {{
#     font-family: 'Raleway', sans-serif !important;
#     font-size: 32px !important;
#     font-weight: 700 !important;
#     margin-top: 15px;
#     color: #1f2937;
# }}
# .stButton>button {{
#     font-family: 'Raleway', sans-serif;
#     font-weight: 600;
#     border-radius: 12px;
#     font-size: 15px;
#     padding: 10px 16px;
#     background: linear-gradient(135deg, #ff4b4b, #ff6b6b);
#     color: white;
#     border: none;
# }}
# </style>

# <div class="top-wrapper">
#     <img src="data:image/gif;base64,{encoded_gif}" />
# </div>
# """, unsafe_allow_html=True)


# import streamlit as st
# import requests
# import pandas as pd
# from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode



# FASTAPI_URL = "http://localhost:8000"
# st.set_page_config(page_title="SHACL Data Generator", layout="wide")
# # st.title("SHACL-Based Synthetic Data Generator")

# st.markdown("""
# <h1 style="font-family: 'Raleway', sans-serif; font-size: 35px; font-weight: 700; color:#1f2937;">
# SHACL-Based Synthetic Data Generator
# </h1>
# """, unsafe_allow_html=True)


# uploaded_file = st.file_uploader("Upload SHACL file (.ttl/.shacl)", type=["ttl", "shacl"])

# if uploaded_file:
#     st.success(f"Uploaded: {uploaded_file.name}")
#     files = {"file": (uploaded_file.name, uploaded_file.getvalue())}

#     # Extract schema from backend
#     try:
#         schema_resp = requests.post(
#             "http://localhost:8000/upload_shacl_and_extract_schema",
#             files=files
#         )
#         schema_resp.raise_for_status()
#         schema = schema_resp.json()["json_schema"]
#     except Exception as e:
#         st.error(f"Failed to extract schema: {e}")
#         st.stop()

#     # Convert schema to DataFrame
#     df = pd.DataFrame(schema)
#     df["model_type"] = "LLM"
#     df["model_name"] = ""

#     # -----------------------------
#     # JavaScript for dynamic dropdown
#     # -----------------------------
#     # When model_type changes, fetch model_name options from backend
#     js_code = JsCode("""
#     function(params) {
#         if (params.colDef.field === "model_type") {
#             let model_type = params.newValue;
#             if (model_type === "LLM") {
#                 params.data.model_name = "";
#                 return;
#             }
#             // Fetch model names dynamically
#             fetch(`http://localhost:8000/models?model_type=${model_type}`)
#                 .then(response => response.json())
#                 .then(data => {
#                     if (data.length > 0) {
#                         params.data.model_name = data[0];
#                         params.api.refreshCells({rowNodes:[params.node], columns:["model_name"]});
#                     } else {
#                         params.data.model_name = "";
#                         params.api.refreshCells({rowNodes:[params.node], columns:["model_name"]});
#                     }
#                 });
#         }
#     }
#     """)

#     # # -----------------------------
#     # # Configure AgGrid
#     # # -----------------------------
#     # gb = GridOptionsBuilder.from_dataframe(df)
#     # gb.configure_default_column(editable=True)

#     # gb.configure_column(
#     #     "model_type",
#     #     cellEditor="agSelectCellEditor",
#     #     cellEditorParams={"values": ["LLM", "VAE", "GAN"]},
#     # )

#     # gb.configure_column(
#     #     "model_name",
#     #     editable=True,
#     #     cellEditor="agSelectCellEditor",
#     #     cellEditorParams={"values": []},  # options updated dynamically
#     # )

#     # grid_options = gb.build()
#     # grid_options["onCellValueChanged"] = js_code

#     # grid_response = AgGrid(
#     #     df,
#     #     gridOptions=grid_options,
#     #     update_mode=GridUpdateMode.VALUE_CHANGED,
#     #     fit_columns_on_grid_load=True,
#     #     allow_unsafe_jscode=True
#     # )

#     # updated_df = grid_response["data"]

#     # -----------------------------
#     # Configure AgGrid
#     # -----------------------------
#     gb = GridOptionsBuilder.from_dataframe(df)
#     gb.configure_default_column(editable=True)

#     # model_type remains a dropdown
#     gb.configure_column(
#         "model_type",
#         cellEditor="agSelectCellEditor",
#         cellEditorParams={"values": ["LLM", "VAE", "GAN"]},
#     )

#     # model_name becomes a simple editable text input
#     gb.configure_column(
#         "model_name",
#         editable=True,  # user can type directly
#         cellEditor="agTextCellEditor"
#     )

#     grid_options = gb.build()

#     grid_response = AgGrid(
#         df,
#         gridOptions=grid_options,
#         update_mode=GridUpdateMode.VALUE_CHANGED,
#         fit_columns_on_grid_load=True,
#         allow_unsafe_jscode=True
#     )

#     updated_df = grid_response["data"]

#     # st.subheader("Adjusted Schema & Configuration")
#     st.markdown("""
#     <h2 style="font-family: 'Raleway', sans-serif; font-size: 28px; font-weight: 600; color:#1f2937;">
#     Adjusted Schema & Configuration
#     </h2>
#     """, unsafe_allow_html=True)
#     st.dataframe(updated_df)

#     num_samples = st.number_input("Number of samples per property", min_value=1, max_value=100, value=3)

#     if st.button("Generate Data"):
#         payload = []
#         for _, row in updated_df.iterrows():
#             payload.append({
#                 "path": row["path"],
#                 "datatype": row.get("datatype", "http://www.w3.org/2001/XMLSchema#string"),
#                 "min_count": row.get("min_count", 1),
#                 "max_count": row.get("max_count", 1),
#                 "model_type": row["model_type"],
#                 "model_name": row["model_name"],
#                 "distribution_type": row.get("distribution_type", "categorical"),
#                 "distribution_params": row.get("distribution_params", {})
#             })
#         st.subheader("Request Payload")
#         st.json(payload)  # or st.write(payload)
#         try:
#             # gen_resp = requests.post("http://localhost:8000/generate_from_shacl", json=payload)
#             gen_resp = requests.post("http://localhost:8000/generate_from_shacl", json=payload)
#             gen_resp.raise_for_status()
#             st.success("Data generated successfully!")
#             st.json(gen_resp.json())
#         except Exception as e:
#             st.error(f"Failed to generate data: {e}")





# import streamlit as st
# import requests
# import json

# API_BASE = "http://localhost:8000"

# # -----------------------------
# # Custom progress bar
# # -----------------------------
# def render_custom_progress_bar(progress):
#     percentage = int(progress)
#     bar = f"""
#     <div style="background-color: #e0e0e0; border-radius: 8px; height: 24px; width: 100%; margin-top: 20px;">
#         <div style="
#             background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
#             width: {percentage}%;
#             height: 100%;
#             border-radius: 8px;
#             text-align: center;
#             color: white;
#             font-weight: bold;
#             line-height: 24px;">
#             {percentage}%
#         </div>
#     </div>
#     """
#     st.markdown(bar, unsafe_allow_html=True)

# # -----------------------------
# # Upload SHACL
# # -----------------------------
# st.title("SHACL to RDF Generator")

# uploaded_file = st.file_uploader("Upload SHACL (.ttl)", type=["ttl"])
# parsed_schema = None

# if uploaded_file:
#     with st.spinner("Uploading and parsing SHACL..."):
#         files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
#         response = requests.post(f"{API_BASE}/upload_shacl_and_extract_schema", files=files)
#         if response.status_code == 200:
#             parsed_schema = response.json()["json_schema"]
#             st.success("SHACL parsed successfully!")

# # -----------------------------
# # Configure properties
# # -----------------------------
# if parsed_schema:
#     st.header("Configure Generation Parameters")

#     num_samples = st.number_input("Number of samples", min_value=1, value=3)

#     for idx, prop in enumerate(parsed_schema):
#         st.subheader(f"Property: {prop['path'].split('#')[-1]}")
#         col1, col2 = st.columns(2)

#         with col1:
#             prop["model_type"] = st.selectbox(
#                 "Model Type", ["LLM", "VAE", "GAN"], key=f"model_type_{idx}"
#             )
#         with col2:
#             default_name = "" if prop["model_type"] == "LLM" else prop.get("model_name", "")
#             prop["model_name"] = st.text_input(
#                 "Model Name", value=default_name, key=f"model_name_{idx}"
#             )

#         # Adjust distribution parameters dynamically
#         if prop["distribution_type"] == "categorical":
#             allowed_list = prop["distribution_params"].get("allowed_list", [])
#             probabilities = prop["distribution_params"].get("probabilities", [])
#             st.text_area(f"Allowed Values", value=", ".join(allowed_list), key=f"allowed_{idx}")
#             st.text_area(f"Probabilities", value=", ".join(probabilities), key=f"prob_{idx}")
#         elif prop["distribution_type"] == "numeric":
#             mean = prop["distribution_params"].get("mean", 0)
#             std = prop["distribution_params"].get("std", 1)
#             min_val = prop["distribution_params"].get("min", 0)
#             max_val = prop["distribution_params"].get("max", 10)
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 prop["distribution_params"]["mean"] = st.number_input(f"Mean", value=float(mean), key=f"mean_{idx}")
#             with col2:
#                 prop["distribution_params"]["std"] = st.number_input(f"Std", value=float(std), key=f"std_{idx}")
#             with col3:
#                 prop["distribution_params"]["min"] = st.number_input(f"Min", value=float(min_val), key=f"min_{idx}")
#             with col4:
#                 prop["distribution_params"]["max"] = st.number_input(f"Max", value=float(max_val), key=f"max_{idx}")

# # -----------------------------
# # Generate RDF
# # -----------------------------
# if parsed_schema and st.button("Generate RDF Samples"):
#     st.subheader("Generating RDF...")
#     progress_placeholder = st.empty()

#     # Update parsed_schema with user edits for allowed/probabilities
#     for idx, prop in enumerate(parsed_schema):
#         if prop["distribution_type"] == "categorical":
#             allowed_str = st.session_state.get(f"allowed_{idx}", "")
#             prob_str = st.session_state.get(f"prob_{idx}", "")
#             prop["distribution_params"]["allowed_list"] = [v.strip() for v in allowed_str.split(",") if v.strip()]
#             if prob_str:
#                 prop["distribution_params"]["probabilities"] = [v.strip() for v in prob_str.split(",") if v.strip()]

#     # Simulate progress updates
#     import time
#     total = num_samples
#     generated_data = None

#     for i in range(total):
#         render_custom_progress_bar((i / total) * 100)
#         time.sleep(0.3)  # simulate progress

#     # Make API request
#     payload = json.dumps(parsed_schema)
#     response = requests.post(f"{API_BASE}/generate_from_shacl?num_samples={num_samples}", data=payload, headers={"Content-Type": "application/json"})
#     if response.status_code == 200:
#         result = response.json()
#         st.success("RDF generation completed!")
#         st.subheader("Generated RDF Samples (Turtle)")
#         for idx, rdf in enumerate(result["rdf_turtle_samples"]):
#             st.text_area(f"Sample {idx+1}", rdf, height=150)

#         st.subheader("Generated Data Samples (JSON)")
#         st.json(result["generated_data_samples"])
#     else:
#         st.error(f"Error: {response.text}")




# import streamlit as st
# import requests
# import json
# from sseclient import SSEClient  # make sure to install via `pip install sseclient-py`

# API_BASE = "http://localhost:8000"

# # -----------------------------
# # Custom progress bar
# # -----------------------------
# def render_custom_progress_bar(progress):
#     percentage = int(progress)
#     bar = f"""
#     <div style="background-color: #e0e0e0; border-radius: 8px; height: 24px; width: 100%; margin-top: 20px;">
#         <div style="
#             background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
#             width: {percentage}%;
#             height: 100%;
#             border-radius: 8px;
#             text-align: center;
#             color: white;
#             font-weight: bold;
#             line-height: 24px;">
#             {percentage}%
#         </div>
#     </div>
#     """
#     st.markdown(bar, unsafe_allow_html=True)

# # -----------------------------
# # Upload SHACL
# # -----------------------------
# st.title("SHACL to RDF Generator")

# uploaded_file = st.file_uploader("Upload SHACL (.ttl)", type=["ttl"])
# parsed_schema = None

# if uploaded_file:
#     with st.spinner("Uploading and parsing SHACL..."):
#         files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
#         response = requests.post(f"{API_BASE}/upload_shacl_and_extract_schema", files=files)
#         if response.status_code == 200:
#             parsed_schema = response.json()["json_schema"]
#             st.success("SHACL parsed successfully!")

# # -----------------------------
# # Configure properties
# # -----------------------------
# if parsed_schema:
#     st.header("Configure Generation Parameters")

#     num_samples = st.number_input("Number of samples", min_value=1, value=3)

#     for idx, prop in enumerate(parsed_schema):
#         st.subheader(f"Property: {prop['path'].split('#')[-1]}")
#         col1, col2 = st.columns(2)

#         with col1:
#             prop["model_type"] = st.selectbox(
#                 "Model Type", ["LLM", "VAE", "GAN"], key=f"model_type_{idx}"
#             )
#         with col2:
#             default_name = "" if prop["model_type"] == "LLM" else prop.get("model_name", "")
#             prop["model_name"] = st.text_input(
#                 "Model Name", value=default_name, key=f"model_name_{idx}"
#             )

#         if prop["distribution_type"] == "categorical":
#             allowed_list = prop["distribution_params"].get("allowed_list", [])
#             probabilities = prop["distribution_params"].get("probabilities", [])
#             st.text_area(f"Allowed Values", value=", ".join(allowed_list), key=f"allowed_{idx}")
#             st.text_area(f"Probabilities", value=", ".join(probabilities), key=f"prob_{idx}")
#         elif prop["distribution_type"] == "numeric":
#             mean = prop["distribution_params"].get("mean", 0)
#             std = prop["distribution_params"].get("std", 1)
#             min_val = prop["distribution_params"].get("min", 0)
#             max_val = prop["distribution_params"].get("max", 10)
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 prop["distribution_params"]["mean"] = st.number_input(f"Mean", value=float(mean), key=f"mean_{idx}")
#             with col2:
#                 prop["distribution_params"]["std"] = st.number_input(f"Std", value=float(std), key=f"std_{idx}")
#             with col3:
#                 prop["distribution_params"]["min"] = st.number_input(f"Min", value=float(min_val), key=f"min_{idx}")
#             with col4:
#                 prop["distribution_params"]["max"] = st.number_input(f"Max", value=float(max_val), key=f"max_{idx}")

# # -----------------------------
# # Generate RDF using SSE
# # -----------------------------
# if parsed_schema and st.button("Generate RDF Samples (Real-Time)"):
#     st.subheader("Generating RDF...")

#     # Update parsed_schema with user edits for allowed/probabilities
#     for idx, prop in enumerate(parsed_schema):
#         if prop["distribution_type"] == "categorical":
#             allowed_str = st.session_state.get(f"allowed_{idx}", "")
#             prob_str = st.session_state.get(f"prob_{idx}", "")
#             prop["distribution_params"]["allowed_list"] = [v.strip() for v in allowed_str.split(",") if v.strip()]
#             if prob_str:
#                 prop["distribution_params"]["probabilities"] = [v.strip() for v in prob_str.split(",") if v.strip()]

#     # SSE request
#     url = f"{API_BASE}/generate_from_shacl_stream?num_samples={num_samples}"
    
#     response = requests.post(url, json=parsed_schema, stream=True)
#     client = SSEClient(response)
#     progress_bar = st.empty()
#     generated_rdf = None
#     generated_data = None

#     for event in client.events():
#         data = json.loads(event.data)
#         progress = data.get("progress", 0)
#         render_custom_progress_bar(progress)

#         if "rdf_turtle_samples" in data:
#             generated_rdf = data["rdf_turtle_samples"]
#             generated_data = data["generated_data_samples"]

#     if generated_rdf and generated_data:
#         st.success("RDF generation completed!")
#         st.subheader("Generated RDF Samples (Turtle)")
#         for idx, rdf in enumerate(generated_rdf):
#             st.text_area(f"Sample {idx+1}", rdf, height=150)

#         st.subheader("Generated Data Samples (JSON)")
#         st.json(generated_data)




# import streamlit as st
# import requests
# import json
# from sseclient import SSEClient  # pip install sseclient-py

# API_BASE = "http://fastapi-backend:8000"

# # -----------------------------
# # Custom progress bar
# # -----------------------------
# def render_custom_progress_bar(progress):
#     percentage = int(progress)
#     bar = f"""
#     <div style="background-color: #e0e0e0; border-radius: 8px; height: 24px; width: 100%; margin-top: 20px;">
#         <div style="
#             background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
#             width: {percentage}%;
#             height: 100%;
#             border-radius: 8px;
#             text-align: center;
#             color: white;
#             font-weight: bold;
#             line-height: 24px;">
#             {percentage}%
#         </div>
#     </div>
#     """
#     st.markdown(bar, unsafe_allow_html=True)

# # -----------------------------
# # HTTP request logger
# # -----------------------------
# def log_request(r, *args, **kwargs):
#     st.subheader("Debug: HTTP Request Sent")
#     st.text(f"Request URL: {r.request.url}")
#     st.text(f"Request Method: {r.request.method}")
#     st.text(f"Request Headers: {r.request.headers}")
    
#     body = r.request.body
#     if isinstance(body, bytes):
#         body = body.decode('utf-8')
#     st.text(f"Request Body (truncated): {body[:1000]}{'...' if len(body) > 1000 else ''}")

# # -----------------------------
# # Upload SHACL
# # -----------------------------
# st.title("SHACL to RDF Generator")

# uploaded_file = st.file_uploader("Upload SHACL (.ttl)", type=["ttl"])
# parsed_schema = None

# if uploaded_file:
#     with st.spinner("Uploading and parsing SHACL..."):
#         files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
#         response = requests.post(f"{API_BASE}/upload_shacl_and_extract_schema", files=files)
#         if response.status_code == 200:
#             parsed_schema = response.json()["json_schema"]
#             st.success("SHACL parsed successfully!")

# # -----------------------------
# # Configure properties
# # -----------------------------
# if parsed_schema:
#     st.header("Configure Generation Parameters")
#     num_samples = st.number_input("Number of samples", min_value=1, value=3)

#     for idx, prop in enumerate(parsed_schema):
#         st.subheader(f"Property: {prop['path'].split('#')[-1]}")
#         col1, col2 = st.columns(2)

#         with col1:
#             prop["model_type"] = st.selectbox(
#                 "Model Type", ["LLM", "VAE", "GAN"], key=f"model_type_{idx}"
#             )
#         with col2:
#             default_name = "" if prop["model_type"] == "LLM" else prop.get("model_name", "")
#             prop["model_name"] = st.text_input(
#                 "Model Name", value=default_name, key=f"model_name_{idx}"
#             )

#         # Distribution parameters
#         if prop["distribution_type"] == "categorical":
#             allowed_list = prop["distribution_params"].get("allowed_list", [])
#             probabilities = prop["distribution_params"].get("probabilities", [])
#             st.text_area(f"Allowed Values", value=", ".join(allowed_list), key=f"allowed_{idx}")
#             st.text_area(f"Probabilities", value=", ".join(probabilities), key=f"prob_{idx}")
#         elif prop["distribution_type"] == "numeric":
#             mean = prop["distribution_params"].get("mean", 0)
#             std = prop["distribution_params"].get("std", 1)
#             min_val = prop["distribution_params"].get("min", 0)
#             max_val = prop["distribution_params"].get("max", 10)
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 prop["distribution_params"]["mean"] = st.number_input(f"Mean", value=float(mean), key=f"mean_{idx}")
#             with col2:
#                 prop["distribution_params"]["std"] = st.number_input(f"Std", value=float(std), key=f"std_{idx}")
#             with col3:
#                 prop["distribution_params"]["min"] = st.number_input(f"Min", value=float(min_val), key=f"min_{idx}")
#             with col4:
#                 prop["distribution_params"]["max"] = st.number_input(f"Max", value=float(max_val), key=f"max_{idx}")

# # -----------------------------
# # Generate RDF using SSE
# # -----------------------------
# if parsed_schema and st.button("Generate RDF Samples (Real-Time)"):
#     st.subheader("Generating RDF...")

#     # Update categorical distribution values
#     for idx, prop in enumerate(parsed_schema):
#         if prop["distribution_type"] == "categorical":
#             allowed_str = st.session_state.get(f"allowed_{idx}", "")
#             prob_str = st.session_state.get(f"prob_{idx}", "")
#             prop["distribution_params"]["allowed_list"] = [v.strip() for v in allowed_str.split(",") if v.strip()]
#             if prob_str:
#                 prop["distribution_params"]["probabilities"] = [v.strip() for v in prob_str.split(",") if v.strip()]

#     # SSE request
#     url = f"{API_BASE}/generate_from_shacl_stream?num_samples={num_samples}"
#     response = requests.post(url, json=parsed_schema, stream=True, hooks={'response': log_request})
#     client = SSEClient(response)

#     progress_bar = st.empty()
#     generated_rdf = None
#     generated_data = None

#     for event in client.events():
#         data = json.loads(event.data)
#         progress = data.get("progress", 0)
#         render_custom_progress_bar(progress)

#         if "rdf_turtle_samples" in data:
#             generated_rdf = data["rdf_turtle_samples"]
#             generated_data = data["generated_data_samples"]

#     if generated_rdf and generated_data:
#         st.success("RDF generation completed!")

#         st.subheader("Generated RDF Samples (Turtle)")
#         for idx, rdf in enumerate(generated_rdf):
#             st.text_area(f"Sample {idx+1}", rdf, height=150)

#         st.subheader("Generated Data Samples (JSON)")
#         st.json(generated_data)



# import streamlit as st
# import requests
# import json
# import base64
# import pandas as pd
# # ----------------------------
# # --- STYLING: Background + GIF ---
# # ----------------------------
# def get_base64_of_file(file_path):
#     with open(file_path, "rb") as f:
#         return base64.b64encode(f.read()).decode()

# background_path = "./background.png"
# gif_path = "./srdfgen.gif"

# base64_background = get_base64_of_file(background_path)
# encoded_gif = get_base64_of_file(gif_path)

# st.markdown(f"""
# <style>
# html, body, [class*="css"] {{
#     font-family: 'Inter', sans-serif;
# }}
# .stApp {{
#     background: url("data:image/png;base64,{base64_background}");
#     background-size: 1000px;
#     background-position: center;
#     background-attachment: fixed;
# }}
# .top-wrapper {{
#     display: flex;
#     flex-direction: column;
#     align-items: center;
#     justify-content: center;
#     margin-top: 30px;
#     margin-bottom: 40px;
#     text-align: center;
# }}
# .top-wrapper img {{ width: 280px; max-width: 90%; }}
# .top-wrapper h1 {{
#     font-family: 'Poppins', sans-serif !important;
#     font-size: 32px !important;
#     font-weight: 700 !important;
#     margin-top: 15px;
#     color: #1f2937;
# }}
# .stButton>button {{
#     font-family: 'Poppins', sans-serif;
#     font-weight: 600;
#     border-radius: 12px;
#     font-size: 15px;
#     padding: 10px 16px;
#     background: linear-gradient(135deg, #ff4b4b, #ff6b6b);
#     color: white;
#     border: none;
# }}
# </style>

# <div class="top-wrapper">
#     <img src="data:image/gif;base64,{encoded_gif}" />
# </div>
# """, unsafe_allow_html=True)


# import streamlit as st
# import requests
# import json
# from sseclient import SSEClient  # pip install sseclient-py

# API_BASE = "http://fastapi-backend:8000"

# # -----------------------------
# # Custom progress bar
# # -----------------------------
# def render_custom_progress_bar(progress):
#     percentage = int(progress)
#     bar = f"""
#     <div style="background-color: #e0e0e0; border-radius: 8px; height: 24px; width: 100%; margin-top: 20px;">
#         <div style="
#             background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
#             width: {percentage}%;
#             height: 100%;
#             border-radius: 8px;
#             text-align: center;
#             color: white;
#             font-weight: bold;
#             line-height: 24px;">
#             {percentage}%
#         </div>
#     </div>
#     """
#     st.markdown(bar, unsafe_allow_html=True)

# # # -----------------------------
# # # HTTP request logger for debugging
# # # -----------------------------
# # def log_request(r, *args, **kwargs):
# #     st.subheader("Debug: HTTP Request Sent")
# #     st.text(f"Request URL: {r.request.url}")
# #     st.text(f"Request Method: {r.request.method}")
# #     st.text(f"Request Headers: {r.request.headers}")
    
# #     body = r.request.body
# #     if isinstance(body, bytes):
# #         body = body.decode('utf-8')
# #     st.text(f"Request Body (truncated): {body[:1000]}{'...' if len(body) > 1000 else ''}")

# # -----------------------------
# # App title
# # -----------------------------
# st.title("SHACL to RDF Generator")

# # -----------------------------
# # Upload SHACL
# # -----------------------------
# uploaded_file = st.file_uploader("Upload SHACL (.ttl)", type=["ttl"])
# parsed_schema = None

# if uploaded_file:
#     with st.spinner("Uploading and parsing SHACL..."):
#         files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
#         response = requests.post(f"{API_BASE}/upload_shacl_and_extract_schema", files=files)
#         if response.status_code == 200:
#             parsed_schema = response.json()["json_schema"]
#             st.success("SHACL parsed successfully!")

# # -----------------------------
# # Configure properties
# # -----------------------------
# # if parsed_schema:
# #     st.header("Configure Generation Parameters")
# #     num_samples = st.number_input("Number of samples", min_value=1, value=3)

# #     for idx, prop in enumerate(parsed_schema):
# #         st.subheader(f"Property: {prop['path'].split('#')[-1]}")
# #         col1, col2 = st.columns(2)

# #         with col1:
# #             prop["model_type"] = st.selectbox(
# #                 "Model Type", ["LLM", "VAE", "GAN"], key=f"model_type_{idx}"
# #             )
# #         with col2:
# #             default_name = "" if prop["model_type"] == "LLM" else prop.get("model_name", "")
# #             prop["model_name"] = st.text_input(
# #                 "Model Name", value=default_name, key=f"model_name_{idx}"
# #             )

# #         # Distribution parameters
# #         if prop["distribution_type"] == "categorical":
# #             allowed_list = prop["distribution_params"].get("allowed_list", [])
# #             probabilities = prop["distribution_params"].get("probabilities", [])
# #             st.text_area(f"Allowed Values", value=", ".join(allowed_list), key=f"allowed_{idx}")
# #             st.text_area(f"Probabilities", value=", ".join(probabilities), key=f"prob_{idx}")
# #         elif prop["distribution_type"] == "numeric":
# #             mean = prop["distribution_params"].get("mean", 0)
# #             std = prop["distribution_params"].get("std", 1)
# #             min_val = prop["distribution_params"].get("min", 0)
# #             max_val = prop["distribution_params"].get("max", 10)
# #             col1, col2, col3, col4 = st.columns(4)
# #             with col1:
# #                 prop["distribution_params"]["mean"] = st.number_input(f"Mean", value=float(mean), key=f"mean_{idx}")
# #             with col2:
# #                 prop["distribution_params"]["std"] = st.number_input(f"Std", value=float(std), key=f"std_{idx}")
# #             with col3:
# #                 prop["distribution_params"]["min"] = st.number_input(f"Min", value=float(min_val), key=f"min_{idx}")
# #             with col4:
# #                 prop["distribution_params"]["max"] = st.number_input(f"Max", value=float(max_val), key=f"max_{idx}")

# # -----------------------------
# # Configure properties
# # -----------------------------
# if parsed_schema:
#     st.header("Configure Generation Parameters")
#     num_samples = st.number_input("Number of samples", min_value=1, value=3)

#     for idx, prop in enumerate(parsed_schema):
#         # Use expander for each property
#         with st.expander(f"Property: {prop['path'].split('#')[-1]}", expanded=False):
#             col1, col2 = st.columns(2)

#             with col1:
#                 prop["model_type"] = st.selectbox(
#                     "Model Type", ["LLM", "VAE", "GAN"], key=f"model_type_{idx}"
#                 )
#             with col2:
#                 default_name = "" if prop["model_type"] == "LLM" else prop.get("model_name", "")
#                 prop["model_name"] = st.text_input(
#                     "Model Name", value=default_name, key=f"model_name_{idx}"
#                 )

#             # Distribution parameters
#             if prop["distribution_type"] == "categorical":
#                 allowed_list = prop["distribution_params"].get("allowed_list", [])
#                 probabilities = prop["distribution_params"].get("probabilities", [])
#                 st.text_area(f"Allowed Values", value=", ".join(allowed_list), key=f"allowed_{idx}")
#                 st.text_area(f"Probabilities", value=", ".join(probabilities), key=f"prob_{idx}")
#             elif prop["distribution_type"] == "numeric":
#                 mean = prop["distribution_params"].get("mean", 0)
#                 std = prop["distribution_params"].get("std", 1)
#                 min_val = prop["distribution_params"].get("min", 0)
#                 max_val = prop["distribution_params"].get("max", 10)
#                 col1, col2, col3, col4 = st.columns(4)
#                 with col1:
#                     prop["distribution_params"]["mean"] = st.number_input(
#                         f"Mean", value=float(mean), key=f"mean_{idx}"
#                     )
#                 with col2:
#                     prop["distribution_params"]["std"] = st.number_input(
#                         f"Std", value=float(std), key=f"std_{idx}"
#                     )
#                 with col3:
#                     prop["distribution_params"]["min"] = st.number_input(
#                         f"Min", value=float(min_val), key=f"min_{idx}"
#                     )
#                 with col4:
#                     prop["distribution_params"]["max"] = st.number_input(
#                         f"Max", value=float(max_val), key=f"max_{idx}"
#                     )

# # # -----------------------------
# # # Generate RDF using SSE
# # # -----------------------------
# # if parsed_schema and st.button("Generate RDF Samples (Real-Time)"):
# #     st.subheader("Generating RDF...")

# #     # Update categorical values from session_state
# #     for idx, prop in enumerate(parsed_schema):
# #         if prop["distribution_type"] == "categorical":
# #             allowed_str = st.session_state.get(f"allowed_{idx}", "")
# #             prob_str = st.session_state.get(f"prob_{idx}", "")
# #             prop["distribution_params"]["allowed_list"] = [v.strip() for v in allowed_str.split(",") if v.strip()]
# #             if prob_str:
# #                 prop["distribution_params"]["probabilities"] = [v.strip() for v in prob_str.split(",") if v.strip()]

# #     # SSE streaming request
# #     url = f"{API_BASE}/generate_from_shacl_stream?num_samples={num_samples}"
# #     response = requests.post(url, json=parsed_schema, stream=True, hooks={'response': log_request})
# #     client = SSEClient(response)

# #     progress_bar = st.empty()
# #     generated_rdf = []
# #     generated_data = []

# #     for event in client.events():
# #         data = json.loads(event.data)
# #         progress = data.get("progress", 0)
# #         render_custom_progress_bar(progress)

# #         # Capture final data
# #         if "rdf_turtle_samples" in data:
# #             generated_rdf = data["rdf_turtle_samples"]
# #             generated_data = data["generated_data_samples"]

# #     # Save results in session_state to persist across reruns
# #     if generated_rdf and generated_data:
# #         st.session_state['generated_rdf'] = generated_rdf
# #         st.session_state['generated_data'] = generated_data
# #         st.success("RDF generation completed!")

# # # -----------------------------
# # # Display results and download
# # # -----------------------------
# # if 'generated_rdf' in st.session_state and 'generated_data' in st.session_state:
# #     rdf_list = st.session_state['generated_rdf']
# #     data_list = st.session_state['generated_data']

# #     st.subheader("Generated RDF Samples (Turtle)")
# #     for idx, rdf in enumerate(rdf_list):
# #         st.text_area(f"Sample {idx+1}", rdf, height=150)

# #     st.subheader("Generated Data Samples (JSON-LD)")
# #     st.json(data_list)

# #     # Download buttons
# #     ttl_bytes = "\n\n".join(rdf_list).encode("utf-8")
# #     st.download_button(
# #         label="Download TTL File",
# #         data=ttl_bytes,
# #         file_name="generated_samples.ttl",
# #         mime="text/turtle"
# #     )

# #     jsonld_bytes = json.dumps(data_list, indent=2).encode("utf-8")
# #     st.download_button(
# #         label="Download JSON-LD File",
# #         data=jsonld_bytes,
# #         file_name="generated_samples.jsonld",
# #         mime="application/ld+json"
# #     )


# # -----------------------------
# # Generate RDF using SSE
# # -----------------------------
# # if parsed_schema and st.button("Generate RDF Samples (Real-Time)"):
# #     st.subheader("Generating RDF...")

# #     # Update categorical values from session_state
# #     for idx, prop in enumerate(parsed_schema):
# #         if prop["distribution_type"] == "categorical":
# #             allowed_str = st.session_state.get(f"allowed_{idx}", "")
# #             prob_str = st.session_state.get(f"prob_{idx}", "")
# #             prop["distribution_params"]["allowed_list"] = [
# #                 v.strip() for v in allowed_str.split(",") if v.strip()
# #             ]
# #             if prob_str:
# #                 prop["distribution_params"]["probabilities"] = [
# #                     v.strip() for v in prob_str.split(",") if v.strip()
# #                 ]

# #     # SSE streaming request (debug logging removed)
# #     url = f"{API_BASE}/generate_from_shacl_stream?num_samples={num_samples}"
# #     response = requests.post(url, json=parsed_schema, stream=True)
# #     client = SSEClient(response)

# #     progress_bar = st.empty()
# #     generated_rdf = []
# #     generated_data = []

# #     for event in client.events():
# #         data = json.loads(event.data)
# #         progress = data.get("progress", 0)
# #         render_custom_progress_bar(progress)

# #         # Capture final data
# #         if "rdf_turtle_samples" in data:
# #             generated_rdf = data["rdf_turtle_samples"]
# #             generated_data = data["generated_data_samples"]

# #     # Save results in session_state to persist across reruns
# #     if generated_rdf and generated_data:
# #         st.session_state['generated_rdf'] = generated_rdf
# #         st.session_state['generated_data'] = generated_data
# #         st.success("RDF generation completed!")


# # -----------------------------
# # Generate RDF using SSE
# # -----------------------------
# if parsed_schema and st.button("Generate RDF Samples (Real-Time)"):
#     st.subheader("Generating RDF...")

#     # Update categorical values from session_state
#     for idx, prop in enumerate(parsed_schema):
#         if prop["distribution_type"] == "categorical":
#             allowed_str = st.session_state.get(f"allowed_{idx}", "")
#             prob_str = st.session_state.get(f"prob_{idx}", "")
#             prop["distribution_params"]["allowed_list"] = [
#                 v.strip() for v in allowed_str.split(",") if v.strip()
#             ]
#             if prob_str:
#                 prop["distribution_params"]["probabilities"] = [
#                     v.strip() for v in prob_str.split(",") if v.strip()
#                 ]

#     # SSE streaming request
#     url = f"{API_BASE}/generate_from_shacl_stream?num_samples={num_samples}"
#     response = requests.post(url, json=parsed_schema, stream=True)
#     client = SSEClient(response)

#     # Streamlit progress bar
#     progress_bar = st.progress(0)
#     generated_rdf = []
#     generated_data = []

#     for event in client.events():
#         data = json.loads(event.data)
#         progress = data.get("progress", 0)
#         progress_bar.progress(progress)  # Update progress bar

#         # Capture final data
#         if "rdf_turtle_samples" in data:
#             generated_rdf = data["rdf_turtle_samples"]
#             generated_data = data["generated_data_samples"]

#     # Save results in session_state to persist across reruns
#     if generated_rdf and generated_data:
#         st.session_state['generated_rdf'] = generated_rdf
#         st.session_state['generated_data'] = generated_data
#         st.success("RDF generation completed!")

# # -----------------------------
# # Display results and download
# # -----------------------------
# if 'generated_rdf' in st.session_state and 'generated_data' in st.session_state:
#     rdf_list = st.session_state['generated_rdf']
#     data_list = st.session_state['generated_data']

#     st.subheader("Generated RDF Samples (Turtle)")
#     for idx, rdf in enumerate(rdf_list):
#         st.text_area(f"Sample {idx+1}", rdf, height=150)

#     st.subheader("Generated Data Samples (JSON-LD)")
#     st.json(data_list)

#     # Download buttons
#     ttl_bytes = "\n\n".join(rdf_list).encode("utf-8")
#     st.download_button(
#         label="Download TTL File",
#         data=ttl_bytes,
#         file_name="generated_samples.ttl",
#         mime="text/turtle"
#     )

#     jsonld_bytes = json.dumps(data_list, indent=2).encode("utf-8")
#     st.download_button(
#         label="Download JSON-LD File",
#         data=jsonld_bytes,
#         file_name="generated_samples.jsonld",
#         mime="application/ld+json"
#     )












import streamlit as st
import requests
import json
import base64
import pandas as pd
# ----------------------------
# --- STYLING: Background + GIF ---
# ----------------------------
def get_base64_of_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

background_path = "./background.png"
gif_path = "./srdfgen.gif"

base64_background = get_base64_of_file(background_path)
encoded_gif = get_base64_of_file(gif_path)

# st.markdown(f"""
# <style>
# html, body, [class*="css"] {{
#     font-family: 'Inter', sans-serif;
# }}
# .stApp {{
#     background: url("data:image/png;base64,{base64_background}");
#     background-size: 1000px;
#     background-position: center;
#     background-attachment: fixed;
# }}
# .top-wrapper {{
#     display: flex;
#     flex-direction: column;
#     align-items: center;
#     justify-content: center;
#     margin-top: 30px;
#     margin-bottom: 40px;
#     text-align: center;
# }}
# .top-wrapper img {{ width: 280px; max-width: 90%; }}
# .top-wrapper h1 {{
#     font-family: 'Poppins', sans-serif !important;
#     font-size: 32px !important;
#     font-weight: 700 !important;
#     margin-top: 15px;
#     color: #1f2937;
# }}
# .stButton>button {{
#     font-family: 'Poppins', sans-serif;
#     font-weight: 600;
#     border-radius: 12px;
#     font-size: 15px;
#     padding: 10px 16px;
#     background: linear-gradient(135deg, #ff4b4b, #ff6b6b);
#     color: white;
#     border: none;
# }}
# </style>

# <div class="top-wrapper">
#     <img src="data:image/gif;base64,{encoded_gif}" />
# </div>
# """, unsafe_allow_html=True)


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

# -----------------------------
# Custom progress bar
# -----------------------------
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

# -----------------------------
# App title
# -----------------------------
# st.title("SHACL to RDF Generator")
st.markdown("""
<h1 style="font-family: 'Poppins', sans-serif; font-size: 28px; font-weight: 700; 
            background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;">
    SHACL to RDF Generator
</h1>
""", unsafe_allow_html=True)

# -----------------------------
# Upload SHACL
# -----------------------------
uploaded_file = st.file_uploader("Upload SHACL (.ttl)", type=["ttl"])
parsed_schema = None

if uploaded_file:
    with st.spinner("Uploading and parsing SHACL..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        response = requests.post(f"{API_BASE}/upload_shacl_and_extract_schema", files=files)
        if response.status_code == 200:
            parsed_schema = response.json()["json_schema"]
            st.success("SHACL parsed successfully!")

# -----------------------------
# Configure properties
# -----------------------------
if parsed_schema:
    # st.header("Configure Generation Parameters")


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
        # Use expander for each property
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

            # Distribution parameters
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


# -----------------------------
# Generate RDF using SSE
# -----------------------------
if parsed_schema and st.button("Generate RDF Samples (Real-Time)"):
    # st.subheader("Generating RDF...")

    st.markdown("""
    <h3 style="font-family: 'Roboto', sans-serif; font-size: 18px; font-weight: 500; 
                background: linear-gradient(90deg, #ff7f7f, #ff4b4b);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;">
        Generating RDF...)
    </h3>
    """, unsafe_allow_html=True)

    # Update categorical values from session_state
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

    # SSE streaming request
    url = f"{API_BASE}/generate_from_shacl_stream?num_samples={num_samples}"
    response = requests.post(url, json=parsed_schema, stream=True)
    client = SSEClient(response)

    # Streamlit progress bar
    progress_bar = st.progress(0)
    generated_rdf = []
    generated_data = []

    for event in client.events():
        data = json.loads(event.data)
        progress = data.get("progress", 0)
        progress_bar.progress(progress)  # Update progress bar

        # Capture final data
        if "rdf_turtle_samples" in data:
            generated_rdf = data["rdf_turtle_samples"]
            generated_data = data["generated_data_samples"]

    # Save results in session_state to persist across reruns
    if generated_rdf and generated_data:
        st.session_state['generated_rdf'] = generated_rdf
        st.session_state['generated_data'] = generated_data
        st.success("RDF generation completed!")

# -----------------------------
# Display results and download
# -----------------------------
if 'generated_rdf' in st.session_state and 'generated_data' in st.session_state:
    rdf_list = st.session_state['generated_rdf']
    data_list = st.session_state['generated_data']

    # st.subheader("Generated RDF Samples (Turtle)")
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

    # st.subheader("Generated Data Samples (JSON-LD)")

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