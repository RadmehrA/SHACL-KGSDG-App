import os
import shutil
from typing import Dict, Any, List, Tuple, Union, Optional
from fastapi import FastAPI, UploadFile, File, APIRouter, HTTPException, Form, Query, Body
from pydantic import BaseModel
from rdflib import Graph, Namespace, RDF, URIRef, BNode
from openai import OpenAI
import random
from fastapi.responses import StreamingResponse, JSONResponse
import numpy as np
from models.gan_model import generate_synthetic_data
from models.llm_generator import generate_llm_data, simplify_key
#from models.vae_generator import generate_synthetic_data_vae  # Import the function
import asyncio
import json
import subprocess
from pathlib import Path
import rdflib
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import torch
from models.gan_model import factorize_and_initialize_gan, generate_synthetic_data, save_model, load_model, train_gan,  Generator, Discriminator, factorize_and_initialize_gans
from pydantic import BaseModel
from typing import Dict, Any, Union, List
from models.distribution_helpers import generate_normal_distribution, generate_uniform_distribution, generate_skewed_distribution, extract_distribution_info
from models.vae_generator import generate_data_vae_model, factorize_and_initialize_vae, train_vae, save_vae_model, load_vae_model, load_and_generate_vae_data
import tempfile
import pickle




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:8501"] for Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Store the parsed shape map globally
shape_map_storage = []

from rdflib import Graph, Namespace, RDF, URIRef, BNode
from typing import List, Dict, Tuple

DIST_NS = "http://example.org/distribution#"
SH = Namespace("http://www.w3.org/ns/shacl#")

def extract_distribution_info(constraints: List[Dict[str, str]]) -> Dict[str, any]:
    dist_info = {}
    for c in constraints:
        for key, val in c.items():
            if key.startswith(DIST_NS):
                short_key = key[len(DIST_NS):]  # e.g., "distribution", "categories", "mean", etc.
                dist_info[short_key] = val
    return dist_info

def parse_shacl(file_path: str) -> List[Dict]:
    g = Graph()
    g.parse(file_path, format="turtle")

    shapes = []

    for s in g.subjects(RDF.type, SH.NodeShape):
        shape_map_entry = {
            "shape": str(s),
            "target_classes": [],
            "properties": []
        }

        # Extract target classes
        for target_class in g.objects(s, SH.targetClass):
            shape_map_entry["target_classes"].append(str(target_class))

        # Extract properties and their constraints
        for property in g.objects(s, SH.property):
            property_entry = {"property": str(property), "constraints": []}
            for predicate, value in g.predicate_objects(property):
                if isinstance(predicate, URIRef):
                    property_entry["constraints"].append({str(predicate): str(value)})
                elif isinstance(predicate, BNode):
                    property_entry["constraints"].append({"BlankNode": str(predicate)})

            # Extract distribution info from constraints
            property_entry["distribution"] = extract_distribution_info(property_entry["constraints"])

            shape_map_entry["properties"].append(property_entry)

        shapes.append(shape_map_entry)

    return shapes


@app.post("/upload_shacl")
async def upload_shacl(file: UploadFile = File(...)):
    file_location = f"shacl_files/{file.filename}"
    os.makedirs("shacl_files", exist_ok=True)

    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    global shape_map_storage
    shape_map_storage = parse_shacl(file_location)

    return {"message": f"SHACL file uploaded successfully: {file_location}", "shape_map": shape_map_storage}


from typing import List, Dict, Tuple


from typing import List, Dict, Tuple

def extract_path_and_datatype(constraints: List[Dict[str, str]]) -> Tuple[str, str]:
    path = None
    datatype = "http://www.w3.org/2001/XMLSchema#string"
    
    for c in constraints:
        if "http://www.w3.org/ns/shacl#path" in c:
            path = c["http://www.w3.org/ns/shacl#path"]
        if "http://www.w3.org/ns/shacl#datatype" in c:
            datatype = c["http://www.w3.org/ns/shacl#datatype"]
        elif "http://www.w3.org/ns/shacl#nodeKind" in c and c["http://www.w3.org/ns/shacl#nodeKind"] == "http://www.w3.org/ns/shacl#IRI":
            datatype = "IRI"
    
    return path, datatype


def get_cardinality(constraints: List[Dict[str, str]]) -> Tuple[int, int]:
    min_count = 1
    max_count = 1
    for c in constraints:
        if "http://www.w3.org/ns/shacl#minCount" in c:
            min_count = int(c["http://www.w3.org/ns/shacl#minCount"])
        if "http://www.w3.org/ns/shacl#maxCount" in c:
            max_count = int(c["http://www.w3.org/ns/shacl#maxCount"])
    return min_count, max_count

def generate_synthetic_sample_with_distribution(constraints: List[Dict[str, str]], user_interactive_message: str, distribution_type: str, distribution_parameters: Dict[str, Any]) -> Dict[str, Any]:
    generated_sample = {}

    # Extract path and datatype (assume extract_path_and_datatype is defined elsewhere)
    path, datatype = extract_path_and_datatype(constraints)

    # Get cardinality for the number of values (assume get_cardinality is defined elsewhere)
    min_count, max_count = get_cardinality(constraints)
    num_values = random.randint(min_count, max_count)

    if path:
        # Handle the distribution-based generation
        if distribution_type == "Normal":
            mean = distribution_parameters.get("mean", 0.0)
            stddev = distribution_parameters.get("stddev", 1.0)
            values = generate_normal_distribution(mean, stddev, num_values)
        elif distribution_type == "Uniform":
            low = distribution_parameters.get("low", 0.0)
            high = distribution_parameters.get("high", 1.0)
            values = generate_uniform_distribution(low, high, num_values)
        elif distribution_type == "Skewed":
            low = distribution_parameters.get("low", 0.0)
            high = distribution_parameters.get("high", 1.0)
            custom_param = distribution_parameters.get("custom_param", "")
            values = generate_skewed_distribution(low, high, num_values, custom_param)
        else:
            # Default case for LLM-based generation (can be extended if needed)
            values = [generate_llm_data(path, datatype, user_interactive_message) for _ in range(num_values)]

        # Store the generated values
        generated_sample[simplify_key(path)] = values[0] if len(values) == 1 else values

    return generated_sample


def get_all_shacl_property_paths() -> List[Dict[str, str]]:
    properties = []
    for shape in shape_map_storage:
        for prop in shape.get("properties", []):
            for constraint in prop.get("constraints", []):
                path = constraint.get("http://www.w3.org/ns/shacl#path")
                if path:
                    properties.append({
                        "shape": shape["shape"],
                        "path": path
                    })
    return properties


DATATYPE_MAP = {
    "http://www.w3.org/2001/XMLSchema#string": "text",
    "http://www.w3.org/2001/XMLSchema#integer": "integer",
    "http://www.w3.org/2001/XMLSchema#decimal": "decimal number",
    "http://www.w3.org/2001/XMLSchema#boolean": "true/false value",
    "http://www.w3.org/2001/XMLSchema#date": "date (YYYY-MM-DD)",
    "http://www.w3.org/2001/XMLSchema#dateTime": "datetime (YYYY-MM-DDTHH:MM:SS)",
}

# Function to simplify the path into a field name
def simplify_key(path: str) -> str:
    return path.split("/")[-1]



class DistributionRequest(BaseModel):
    num_samples: int
    distribution_type: str
    parameters: Dict[str, Any]
    property_model_map: Dict[str, Union[str, Dict[str, str]]]
    user_message: str = ""
    model_name: List[str]  # Add model_name as a List of strings to hold model selections

def build_distribution_parameters(distribution_type: str) -> Dict[str, Any]:
    if distribution_type == "Normal":
        return {"mean": 0.0, "stddev": 1.0}
    elif distribution_type == "Uniform":
        return {"low": 0.0, "high": 1.0}
    elif distribution_type == "Skewed":
        return {"low": 0.0, "high": 1.0, "custom_param": "right"}
    else:
        return {}


def generate_synthetic_sample_with_distribution_gan(
    constraints: List[Dict[str, str]],
    subject_input: str,
    model_config: Union[Dict[str, str], None],
    distribution_type: str,
    distribution_parameters: Dict[str, Any]
) -> Dict[str, Any]:
    generated_sample = {}

    # Extract predicate path and datatype from SHACL constraints
    path, datatype = extract_path_and_datatype(constraints)
    if not path:
        return generated_sample  # skip if no valid path

    simplified = simplify_key(path)

    # Determine cardinality
    min_count, max_count = get_cardinality(constraints)
    num_values = random.randint(min_count, max_count)

    # Choose models
    if isinstance(model_config, dict) and "modelname" in model_config:
        selected_models = [model_config["modelname"], "all"]
    else:
        selected_models = list_saved_models_gan()["saved_models"]
        if "all" not in selected_models:
            selected_models.append("all")

    # Generate values
    generated_values = []
    for model_name in selected_models:
        try:
            values = generate_synthetic_data(
                model_name=model_name,
                subject_input=subject_input,
                predicate_input=simplified,
                num_samples=num_values,
                distribution=distribution_type
            )
            generated_values.extend(values)
        except ValueError as e:
            generated_values.append(f"Error: {str(e)}")

    # Deduplicate and limit to num_values
    generated_values = list(dict.fromkeys(generated_values))[:num_values]

    # Store in final sample
    generated_sample[simplified] = generated_values[0] if len(generated_values) == 1 else generated_values
    return generated_sample

def generate_synthetic_sample_with_distribution_vae(
    model,
    factorized_data,
    subject_input,
    predicate_input,
    constraints,
    num_samples=None,
    distribution="normal"
):
    # Extract path and datatype from SHACL constraints
    path, datatype = extract_path_and_datatype(constraints)
    if not path:
        raise ValueError("No valid path found in SHACL constraints.")

    # Get cardinality (used if num_samples not explicitly set)
    min_count, max_count = get_cardinality(constraints)
    if num_samples is None:
        num_samples = random.randint(min_count, max_count)

    # Subject and predicate matching
    subject_dim = factorized_data["subject_dim"]
    predicate_dim = factorized_data["predicate_dim"]

    subject_input_lower = subject_input.lower()
    predicate_input_lower = predicate_input.lower()

    subject_matches = [s for s in factorized_data["subject_uniques"] if subject_input_lower in s.lower()]
    predicate_matches = [p for p in factorized_data["predicate_uniques"] if predicate_input_lower in p.lower()]

    if len(subject_matches) == 0:
        raise ValueError(f"Subject '{subject_input}' not found.")
    if len(predicate_matches) == 0:
        raise ValueError(f"Predicate '{predicate_input}' not found.")

    subject_input = subject_matches[0]
    predicate_input = predicate_matches[0]

    subject_idx = np.where(factorized_data["subject_uniques"] == subject_input)[0][0]
    predicate_idx = np.where(factorized_data["predicate_uniques"] == predicate_input)[0][0]

    s = torch.tensor([subject_idx], dtype=torch.long)
    p = torch.tensor([predicate_idx], dtype=torch.long)

    s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
    p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()

    x_cond = torch.cat((s_oh, p_oh), dim=1).repeat(num_samples, 1)

    # Generate samples using the selected distribution
    with torch.no_grad():
        if distribution == "normal":
            mu, logvar = model.encode(x_cond)
            z = model.reparameterize(mu, logvar)
            generated = model.decode(z, x_cond)
        elif distribution == "uniform":
            z = torch.rand_like(torch.randn_like(x_cond))  # Uniform distribution
            generated = model.decode(z, x_cond)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    generated_idx = torch.argmax(generated, dim=1).numpy()
    decoded_objects = [factorized_data["object_inverse_map"].get(idx, "UNKNOWN") for idx in generated_idx]

    return decoded_objects



@app.post("/generate_data")
async def generate_data(request: DistributionRequest):
    async def data_generator():
        shape_map = shape_map_storage
        distribution_type = request.distribution_type
        parameters = request.parameters
        synthetic_data = []

        total_samples = request.num_samples

        for idx in range(total_samples):
            sample = {}

            for shape in shape_map:
                subject = shape.get("target_classes", ["Unknown"])[0]
                simplified_subject = simplify_key(subject)

                for property_data in shape["properties"]:
                    constraints = property_data["constraints"]
                    path, _ = extract_path_and_datatype(constraints)
                    simplified = simplify_key(path)

                    model_config = request.property_model_map.get(path, "LLM")
                    model_type = "LLM"
                    if isinstance(model_config, str):
                        model_type = model_config
                    elif isinstance(model_config, dict):
                        model_type = model_config.get("type", "LLM")

                    try:
                        
                        if model_type == "GAN":
                            generated = generate_synthetic_sample_with_distribution_gan(
                                constraints=constraints,
                                subject_input=simplified_subject,
                                model_config=model_config,
                                distribution_type=distribution_type,
                                distribution_parameters = build_distribution_parameters(distribution_type)

                            )
                            sample.update(generated)

                        elif model_type == "VAE":
                            if isinstance(model_config, dict) and "modelname" in model_config:
                                selected_models = [model_config["modelname"], "string_vae"]
                            else:
                                selected_models = list_saved_models_vae()["saved_models"]
                                if "string_vae" not in selected_models:
                                    selected_models.append("string_vae")

                            for model_name in selected_models:
                                if model_name not in loaded_models:
                                    model_path = f"/app/models/saved_models/vae/{model_name}/vae.pth"
                                    if not os.path.exists(model_path):
                                        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found in storage.")
                                    print(f"Model '{model_name}' not loaded. Loading model from storage...")
                                    ttl_path = f"/app/uploaded/vae/{model_name}.ttl"
                                    factorized_data, vae_model, vae_optimizer = load_vae_model(model_name=model_name, ttl_path=ttl_path)

                                    loaded_models[model_name] = {
                                        "vae_model": vae_model,
                                        "factorized_data": factorized_data,
                                        "optimizer": vae_optimizer
                                    }

                                vae_model_info = loaded_models[model_name]
                                vae_model = vae_model_info["vae_model"]
                                factorized_data = vae_model_info["factorized_data"]

                                # 🧠 Extract cardinality and compute num_samples here
                                min_count, max_count = get_cardinality(constraints)
                                num_samples = max(random.randint(min_count, max_count), 1)  

                                # 🎯 Now pass num_samples to your generation function
                                generated_samples = generate_synthetic_sample_with_distribution_vae(
                                    constraints=constraints,
                                    model=vae_model,
                                    factorized_data=factorized_data,
                                    subject_input=simplified_subject,
                                    predicate_input=simplified,
                                    num_samples=num_samples,
                                    distribution=distribution_type
                                )
                                sample[simplified] = generated_samples

                        else:  # LLM
                            user_interactive_message = request.user_message
                            generated_sample = generate_synthetic_sample_with_distribution(
                                constraints,
                                user_interactive_message,
                                distribution_type,
                                parameters
                            )
                            sample.update(generated_sample)

                    except Exception as e:
                        sample[simplified] = f"Error: {str(e)}"

            synthetic_data.append(sample)

            progress_info = {
                "type": "progress_update",
                "progress": round(((idx + 1) / total_samples) * 100, 2),
                "current_sample_idx": idx + 1,
                "total_samples": total_samples
            }
            yield f"data: {json.dumps(progress_info)}\n\n"
            await asyncio.sleep(0)

        final_info = {
            "type": "final_result",
            "synthetic_data": synthetic_data,
            "statistics": {}
        }
        yield f"data: {json.dumps(final_info)}\n\n"

    return StreamingResponse(data_generator(), media_type="text/event-stream")




@app.get("/test_generate_llm_data")
async def test_generate_llm_data(path: str, datatype: str, user_interactive_message: str = ""):
    # Log the received parameters for debugging
    print(f"[DEBUG] Received Path: {path}, Datatype: {datatype}, User Interactive Message: {user_interactive_message}")

    # Pass the user interactive message to the generate_llm_data function
    value = generate_llm_data(path, datatype, user_interactive_message)

    # Log the generated value for debugging
    print(f"[DEBUG] Generated Value: {value}")

    return {"generated_value": value}

@app.get("/stream_generate_data")
async def stream_generate_data(num_samples: int = 5, user_interactive_message: str = ""):
    shape_map = shape_map_storage

    async def event_stream():
        try:
            for i in range(num_samples):
                sample = {}
                for shape in shape_map:
                    for property_data in shape["properties"]:
                        constraints = property_data["constraints"]
                        # Pass the user interactive message to generate synthetic sample
                        generated_sample = generate_synthetic_sample_with_distribution(constraints, user_interactive_message)
                        sample.update(generated_sample)
                yield f"data: {json.dumps(sample)}\n\n"
                await asyncio.sleep(0.1)
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Endpoint to generate synthetic data from GAN model
@app.post("/generate_gan_data")
async def generate_gan_data(num_samples: int = 5):
    """
    Endpoint to generate synthetic data using the GAN model.
    The `num_samples` parameter specifies how many samples to generate.
    """
    generated_data = generate_synthetic_data(num_samples)

    # Return the generated data in a structured format
    return {"generated_data": generated_data.tolist()}


class VAERequest(BaseModel):
    num_samples: int
    constraints: List[Dict[str, str]]

@app.get("/get_shacl_properties")
async def list_shacl_properties():
    """
    Return all properties (paths) found in the parsed SHACL file.
    """
    if not shape_map_storage:
        return {"error": "No SHACL file uploaded or parsed yet."}
    
    properties = get_all_shacl_property_paths()
    return {"properties": properties}

@app.get("/fetch-latest-dbpedia")
async def fetch_latest_dbpedia():
    script_path = Path("triples_resource/download_dbpedia_core.sh").resolve()
    try:
        result = subprocess.run(
            ["bash", str(script_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/app"  # Working directory inside container
        )
        print("Standard Output:", result.stdout)
        return {"message": "DBpedia files downloaded successfully!"}
    except subprocess.CalledProcessError as e:
        print("Standard Error:", e.stderr)
        return {"error": f"Failed to download DBpedia files: {e.stderr}"}
    
    
class SyntheticRequest(BaseModel):
    subject_index: int
    predicate_index: int
    num_samples: int = 1

class SyntheticResponse(BaseModel):
    synthetic_objects: List[List[float]]

@app.post("/generate-synthetic-data/")
def generate_synthetic_data_endpoint(subject_input: str, predicate_input: str, num_samples: int = 1):
    try:
        # Ensure that the generator and discriminator models are loaded
        load_model("generator.pth", "discriminator.pth")  # Add the correct model paths

        # Call the synthetic data generation function
        generated_data = generate_synthetic_data(subject_input, predicate_input, num_samples)
        return {"generated_data": generated_data}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


def factorize_rdf_data(file_path: str):
    """
    Factorizes RDF data into subjects, predicates, and objects, then returns factorized data.
    """
    # Load the RDF data into a Graph
    graph = Graph()
    graph.parse(file_path, format="turtle")

    # Extract subjects, predicates, and objects from the RDF graph
    subjects = []
    predicates = []
    objects = []
    for subj, pred, obj in graph:
        subjects.append(str(subj))
        predicates.append(str(pred))
        objects.append(str(obj))

    # Factorize the subjects, predicates, and objects
    subject_encoded, subject_unique = pd.factorize(subjects)
    predicate_encoded, predicate_unique = pd.factorize(predicates)
    object_encoded, object_unique = pd.factorize(objects)

    # Return factorized data
    factorized_data = {
        "subjects": subject_encoded,
        "predicates": predicate_encoded,
        "objects": object_encoded,
        "subject_dim": len(subject_unique),
        "predicate_dim": len(predicate_unique),
        "object_dim": len(object_unique),
        "df": pd.DataFrame({
            'subject': subject_encoded,
            'predicate': predicate_encoded,
            'object': object_encoded
        }),
        "subject_uniques": subject_unique,
        "predicate_uniques": predicate_unique
    }

    return factorized_data

@app.post("/upload-train/")
def upload_and_train(file: UploadFile = File(...), num_epochs: int = 1000):
    try:
        file_path = Path(f"./uploads/{file.filename}")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # Step 1: Factorize the RDF data
        factorized_data = factorize_rdf_data(file_path)

        # You can now access the factorized data, e.g., factorized_data['subject_uniques']

        # Step 2: Train your GAN with the factorized data
        train_gan(factorized_data, num_epochs=num_epochs)

        # Step 3: Save the model
        save_model()

        # Clean up the uploaded file after training
        file_path.unlink()

        return {"detail": f"GAN trained on {file.filename} and model saved."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
    
@app.post("/upload-ttl/")
async def upload_ttl(file: UploadFile = File(...), model_name: str = Form(...)):
    try:
        os.makedirs("uploaded", exist_ok=True)
        os.makedirs(f"models/{model_name}", exist_ok=True)

        contents = await file.read()
        path = f"uploaded/{model_name}.ttl"
        with open(path, "wb") as f:
            f.write(contents)

        print("✅ TTL file saved")
        factorize_and_initialize_gan(path)
        print("✅ GAN initialized")
        train_gan(num_epochs=100, batch_size=32)  # This will now use dynamic batch size
        print("✅ GAN trained")

        save_model(model_name)
        print("✅ Model saved")
        
        return {"message": f"Model '{model_name}' trained and saved."}
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return {"error": str(e)}



@app.get("/generate/")
def generate(modelname: str, subject: str, predicate: str, num_samples: int = 1):
    try:
        generated_objects = generate_synthetic_data(modelname, subject, predicate, num_samples)
        return {"generated_objects": generated_objects}
    except Exception as e:
        return {"error": str(e)}


class GANBatchRequest(BaseModel):
    requests: List[dict]  # Each dict should contain: subject, predicate, num_samples

@app.post("/generate_batch_gan/")
def generate_batch_gan(request: GANBatchRequest):
    results = []
    for req in request.requests:
        subject = req.get("subject")
        predicate = req.get("predicate")
        num_samples = req.get("num_samples", 1)
        try:
            generated_objects = generate_synthetic_data(subject, predicate, num_samples)
            results.append({
                "subject": subject,
                "predicate": predicate,
                "generated_objects": generated_objects
            })
        except Exception as e:
            results.append({
                "subject": subject,
                "predicate": predicate,
                "error": str(e)
            })
    return {"results": results}


# Function to list saved models
def list_saved_models_gan(models_root="/app/models/saved_models/gan"):
    if not os.path.exists(models_root):
        return {"message": f"Directory {models_root} does not exist."}

    # List the subdirectories (which are our models)
    model_names = [d for d in os.listdir(models_root) if os.path.isdir(os.path.join(models_root, d))]
    
    if not model_names:
        return {"message": "No models found."}
    
    return {"saved_models": model_names}

# Function to list saved models
def list_saved_models_vae(models_root="/app/models/saved_models/vae"):
    if not os.path.exists(models_root):
        return {"message": f"Directory {models_root} does not exist."}

    # List the subdirectories (which are our models)
    model_names = [d for d in os.listdir(models_root) if os.path.isdir(os.path.join(models_root, d))]
    
    if not model_names:
        return {"message": "No models found."}
    
    return {"saved_models": model_names}



@app.get("/models/saved/gan")
async def get_saved_models():
    return list_saved_models_gan(models_root="/app/models/saved_models/gan")

@app.get("/models/saved/vae")
async def get_saved_models():
    return list_saved_models_vae(models_root="/app/models/saved_models/vae")


def load_model_by_name(model_name, models_root="/app/models/saved_models/gan"):
    global generator, discriminator
    path = f"uploaded/{model_name}.ttl"
    factorize_and_initialize_gan(path)
    try:
        model_path = os.path.join(models_root, model_name)
        gen_path = os.path.join(model_path, "generator.pth")
        disc_path = os.path.join(model_path, "discriminator.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory {model_path} does not exist.")

        if os.path.exists(gen_path) and os.path.exists(disc_path):
            print(f"Loading model: {model_name}")

            # Re-initialize the models before loading
            generator = Generator(
                factorized_data["subject_dim"],
                factorized_data["predicate_dim"],
                factorized_data["object_dim"],
                z_dim
            )
            discriminator = Discriminator(
                factorized_data["subject_dim"],
                factorized_data["predicate_dim"],
                factorized_data["object_dim"]
            )

            # Load the state dicts into the models
            generator.load_state_dict(torch.load(gen_path, map_location=torch.device('cpu')))
            discriminator.load_state_dict(torch.load(disc_path, map_location=torch.device('cpu')))

            # Optionally, set the models to evaluation mode
            generator.eval()
            discriminator.eval()

            return f"Model {model_name} loaded successfully."
        else:
            raise FileNotFoundError(f"Missing generator/discriminator files for model {model_name}.")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return str(e)


@app.post("/load-model/")
async def load_model_endpoint(model_name: str):
    try:
        ttl_path = f"uploaded/{model_name}.ttl"
        if not os.path.exists(ttl_path):
            return {"error": f"TTL file for model '{model_name}' not found at {ttl_path}"}

        load_model(model_name)
        return {"message": f"Model '{model_name}' loaded successfully."}
    except Exception as e:
        return {"error": str(e)}



def load_all_models_on_startup():
    uploaded_dir = "uploaded"
    if not os.path.exists(uploaded_dir):
        print("📂 No 'uploaded' directory found. Skipping model preload.")
        return

    ttl_files = [f for f in os.listdir(uploaded_dir) if f.endswith(".ttl")]
    for ttl_file in ttl_files:
        model_name = ttl_file.replace(".ttl", "")
        ttl_path = os.path.join(uploaded_dir, ttl_file)
        try:
            print(f"🔄 Loading model: {model_name}")
            factorize_and_initialize_gan(ttl_path)
            load_model(model_name)
            print(f"✅ Model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load model '{model_name}': {e}")


@app.on_event("startup")
def startup_event():
    load_all_models_on_startup()


class GenerateDataRequest(BaseModel):
    model_name: str
    subject: str
    predicate: str
    num_samples: int = 1
    distribution: str = "normal"

@app.post("/generate")
def generate_synthetic_data_endpoint(request: GenerateDataRequest):
    try:
        results = generate_synthetic_data(
            model_name=request.model_name,
            subject_input=request.subject,
            predicate_input=request.predicate,
            num_samples=request.num_samples,
            distribution=request.distribution
        )
        return {"generated_objects": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.on_event("startup")
def load_all_saved_models():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        return

    for model_name in os.listdir(MODEL_DIR):
        model_path = os.path.join(MODEL_DIR, model_name, "vae.pth")
        if os.path.exists(model_path):
            try:
                factorized_data, vae_model, vae_optimizer = load_vae_model(model_name)
                loaded_models[model_name] = {
                    "vae_model": vae_model,
                    "factorized_data": factorized_data,
                    "optimizer": vae_optimizer
                }
                print(f"Loaded model '{model_name}' at startup.")
            except Exception as e:
                print(f"Failed to load model '{model_name}': {e}")



@app.post("/upload_and_train_vae")
async def upload_and_train_vae(
    file: UploadFile = File(...),
    epochs: int = Form(100),
    model_name: str = Form(...)
):
    

    try:
        if not model_name.endswith("vae"):
            return {"message": f"Model '{model_name}' is not a VAE model. Skipping load."}

        model_dir = os.path.join(MODEL_DIR, model_name)
        model_path = os.path.join(model_dir, "vae.pth")
        ttl_path = os.path.join(ttl_dir, f"{model_name}.ttl")
        factorized_data_path = os.path.join(model_dir, f"{model_name}_factorized_data.pkl")

        # If already in memory
        if model_name in loaded_models:
            return {"message": f"Model '{model_name}' already loaded in memory."}

        # If saved on disk
        if os.path.exists(model_path):
            factorized_data, vae_model, vae_optimizer = load_vae_model(model_name, ttl_path)
            loaded_models[model_name] = {
                "vae_model": vae_model,
                "factorized_data": factorized_data,
                "optimizer": vae_optimizer
            }
            return {"message": f"Model '{model_name}' loaded from disk."}


        # Train new model
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ttl") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        factorized_data, vae_model, vae_optimizer = factorize_and_initialize_vae(tmp_path)
        train_vae(vae_model, vae_optimizer, factorized_data, num_epochs=epochs)

        os.makedirs(model_dir, exist_ok=True)
        torch.save(vae_model.state_dict(), model_path)

        with open(factorized_data_path, "wb") as f:
            pickle.dump(factorized_data, f)
        with open(ttl_path, "wb") as f:
            f.write(content)

        loaded_models[model_name] = {
            "vae_model": vae_model,
            "factorized_data": factorized_data,
            "optimizer": vae_optimizer
        }

        return {"message": f"Model '{model_name}' trained, saved, and loaded."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


vae_model = None
vae_optimizer = None
factorized_data = None
loaded_models = {}
MODEL_DIR = "/app/models/saved_models/vae"
ttl_dir = "/app/uploaded/vae"

from typing import Optional, Dict, Any

class VAEGenerationRequest(BaseModel):
    model_name: str
    subject: str
    predicate: str  # Accept only the part after the last slash
    num_samples: int = 1
    distribution: str = "normal"  # For legacy support
    dist_params: Optional[Dict[str, Any]] = None  # parsed distribution params from SHACL


@app.post("/generate_vae")
async def generate_vae(request: VAEGenerationRequest):
    try:
        model_name = request.model_name
        subject = request.subject
        predicate = request.predicate  # Expecting just the part after last /
        num_samples = request.num_samples
        distribution = request.distribution  # If used

        print(f"Received Predicate: {predicate}")  # Log to check if it's correct

        if model_name not in loaded_models:
            model_path = f"/app/models/saved_models/vae/{model_name}/vae.pth"
            if not os.path.exists(model_path):
                raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found in storage.")

            print(f"Model '{model_name}' not loaded. Loading model from storage...")
            ttl_path = f"/app/uploaded/{model_name}.ttl"
            factorized_data, vae_model, vae_optimizer = load_vae_model(model_name=model_name, ttl_path=ttl_path)

            loaded_models[model_name] = {
                "vae_model": vae_model,
                "factorized_data": factorized_data,
                "optimizer": vae_optimizer
            }

        vae_model_info = loaded_models[model_name]
        vae_model = vae_model_info["vae_model"]
        factorized_data = vae_model_info["factorized_data"]

        generated_objects = generate_data_vae_model(vae_model, factorized_data, subject, predicate, num_samples)
        
        return JSONResponse(content={"generated_objects": generated_objects})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_and_train_gan/")
async def upload_ttl(file: UploadFile = File(...), model_name: str = Form(...)):
    try:
        os.makedirs("uploaded", exist_ok=True)
        os.makedirs(f"models/{model_name}", exist_ok=True)

        contents = await file.read()
        path = f"uploaded/{model_name}.ttl"
        with open(path, "wb") as f:
            f.write(contents)

        print("✅ TTL file saved")
        factorize_and_initialize_gans(path)
        print("✅ GAN initialized")
        train_gan(num_epochs=100, batch_size=32)  # This will now use dynamic batch size
        print("✅ GAN trained")

        save_model(model_name)
        print("✅ Model saved")
        
        return {"message": f"Model '{model_name}' trained and saved."}
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return {"error": str(e)}

# class GenerateRequest(BaseModel):
#     model_name: str
#     subject: str
#     predicate: str
#     num_samples: int = 1
#     distribution: Optional[str] = "normal"  # normal, uniform, skewed, categorical

from typing import Optional, Dict, Any

class GenerateRequest(BaseModel):
    model_name: str
    subject: str
    predicate: str
    num_samples: int = 1
    distribution: Optional[str] = "normal"  # normal, uniform, skewed, categorical
    dist_params: Optional[Dict[str, Any]] = None

@app.post("/gan/load-and-generate")
def load_and_generate_gan_data(request: GenerateRequest):
    model_name = request.model_name

    # Load the model (and .ttl file) if not already loaded
    try:
        if model_name not in loaded_models:
            load_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    # Generate synthetic data
    try:
        results = generate_synthetic_data(
            model_name=model_name,
            subject_input=request.subject,
            predicate_input=request.predicate,
            num_samples=request.num_samples,
            distribution=request.distribution,
            dist_params=request.dist_params  # <-- pass dist_params here
        )
        return {"generated_objects": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating data: {str(e)}")
