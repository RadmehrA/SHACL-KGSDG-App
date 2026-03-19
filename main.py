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
from models.llm_generator import generate_llm_data, simplify_key
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
from models.vae_generator import generate_data_vae_model, factorize_and_initialize_vae, train_vae, save_vae_model, load_vae_model, load_and_generate_vae_data, find_uploaded_file, save_vae_to_mongo, get_vae_model, upload_train_save_vae
from models.gan_model import generate_synthetic_data, save_gan_to_mongo, load_gan_from_mongo, load_model_gan, load_gan_from_mongo_async

import tempfile
import pickle




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
                short_key = key[len(DIST_NS):]
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

        
        for target_class in g.objects(s, SH.targetClass):
            shape_map_entry["target_classes"].append(str(target_class))

        
        for property in g.objects(s, SH.property):
            property_entry = {"property": str(property), "constraints": []}
            for predicate, value in g.predicate_objects(property):
                if isinstance(predicate, URIRef):
                    property_entry["constraints"].append({str(predicate): str(value)})
                elif isinstance(predicate, BNode):
                    property_entry["constraints"].append({"BlankNode": str(predicate)})

            
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

    
    path, datatype = extract_path_and_datatype(constraints)

    
    min_count, max_count = get_cardinality(constraints)
    num_values = random.randint(min_count, max_count)

    if path:
        
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
            
            values = [generate_llm_data(path, datatype, user_interactive_message) for _ in range(num_values)]

        
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

def simplify_key(path: str) -> str:
    return path.split("/")[-1]



class DistributionRequest(BaseModel):
    num_samples: int
    distribution_type: str
    parameters: Dict[str, Any]
    property_model_map: Dict[str, Union[str, Dict[str, str]]]
    user_message: str = ""
    model_name: List[str]

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

    
    path, datatype = extract_path_and_datatype(constraints)
    if not path:
        return generated_sample

    simplified = simplify_key(path)

    
    min_count, max_count = get_cardinality(constraints)
    num_values = random.randint(min_count, max_count)

    
    if isinstance(model_config, dict) and "modelname" in model_config:
        selected_models = [model_config["modelname"], "all"]
    else:
        selected_models = list_saved_models_gan()["saved_models"]
        if "all" not in selected_models:
            selected_models.append("all")

    
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

    
    generated_values = list(dict.fromkeys(generated_values))[:num_values]

    
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
    
    path, datatype = extract_path_and_datatype(constraints)
    if not path:
        raise ValueError("No valid path found in SHACL constraints.")

    
    min_count, max_count = get_cardinality(constraints)
    if num_samples is None:
        num_samples = random.randint(min_count, max_count)

    
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

    
    with torch.no_grad():
        if distribution == "normal":
            mu, logvar = model.encode(x_cond)
            z = model.reparameterize(mu, logvar)
            generated = model.decode(z, x_cond)
        elif distribution == "uniform":
            z = torch.rand_like(torch.randn_like(x_cond))
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

                                min_count, max_count = get_cardinality(constraints)
                                num_samples = max(random.randint(min_count, max_count), 1)  

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

                        else:
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
    print(f"[DEBUG] Received Path: {path}, Datatype: {datatype}, User Interactive Message: {user_interactive_message}")

    value = generate_llm_data(path, datatype, user_interactive_message)

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
                        
                        generated_sample = generate_synthetic_sample_with_distribution(constraints, user_interactive_message)
                        sample.update(generated_sample)
                yield f"data: {json.dumps(sample)}\n\n"
                await asyncio.sleep(0.1)
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/generate_gan_data")
async def generate_gan_data(num_samples: int = 5):
    """
    Endpoint to generate synthetic data using the GAN model.
    The `num_samples` parameter specifies how many samples to generate.
    """
    generated_data = generate_synthetic_data(num_samples)

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
            cwd="/app"
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
        
        load_model("generator.pth", "discriminator.pth")

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
    
    graph = Graph()
    graph.parse(file_path, format="turtle")

    
    subjects = []
    predicates = []
    objects = []
    for subj, pred, obj in graph:
        subjects.append(str(subj))
        predicates.append(str(pred))
        objects.append(str(obj))

    subject_encoded, subject_unique = pd.factorize(subjects)
    predicate_encoded, predicate_unique = pd.factorize(predicates)
    object_encoded, object_unique = pd.factorize(objects)

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

        
        factorized_data = factorize_rdf_data(file_path)

        
        train_gan(factorized_data, num_epochs=num_epochs)

        
        save_model()

        
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
        train_gan(num_epochs=100, batch_size=32)
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
    requests: List[dict]

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


def list_saved_models_gan(models_root="/app/models/saved_models/gan"):
    if not os.path.exists(models_root):
        return {"message": f"Directory {models_root} does not exist."}

    model_names = [d for d in os.listdir(models_root) if os.path.isdir(os.path.join(models_root, d))]
    
    if not model_names:
        return {"message": "No models found."}
    
    return {"saved_models": model_names}

def list_saved_models_vae(models_root="/app/models/saved_models/vae"):
    if not os.path.exists(models_root):
        return {"message": f"Directory {models_root} does not exist."}

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

            
            generator.load_state_dict(torch.load(gen_path, map_location=torch.device('cpu')))
            discriminator.load_state_dict(torch.load(disc_path, map_location=torch.device('cpu')))

            
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

MODEL_DIR = "/app/models/saved_models/vae"


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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import os
import tempfile


loaded_models = {}

class VAEGenerationRequest(BaseModel):
    model_name: str
    subject: str
    predicate: str
    num_samples: int = 1
    distribution: str = "normal"
    dist_params: dict = None

@app.post("/upload_and_train_vae")
async def upload_and_train_vae(file: UploadFile = File(...), model_name: str = Form(...), epochs: int = Form(50)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".ttl", ".owl", ".rdf", ".xml"]:
        raise HTTPException(status_code=400, detail="Unsupported RDF format")

    try:
        
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        
        models, vae_model_names = await upload_train_save_vae(tmp_path, model_name, num_epochs=epochs)

        
        for name in vae_model_names:
            loaded_models[name] = models[name]

        return {"message": f"VAE model(s) {vae_model_names} trained and saved to MongoDB."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/generate_vae")
async def generate_vae(request: VAEGenerationRequest):
    try:
        
        model_data = await get_vae_model(request.model_name)
        vae_model = model_data["vae_model"]
        factorized_data = model_data["factorized_data"]

        
        generated_objects = generate_data_vae_model(
            vae_model, factorized_data, request.subject, request.predicate, request.num_samples
        )

        return {"generated_objects": generated_objects}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/loaded_vae_models")
async def list_loaded_vae_models():
    return {"loaded_models": list(loaded_models.keys())}


@app.post("/upload_and_train_gan/")
async def upload_ttl(file: UploadFile = File(...), model_name: str = Form(...)):
    try:
        os.makedirs("uploaded", exist_ok=True)
        contents = await file.read()
        ext = os.path.splitext(file.filename)[1].lower()
        path = f"uploaded/{model_name}{ext}"
        with open(path, "wb") as f:
            f.write(contents)

        factorize_and_initialize_gans(path)
        train_gan(num_epochs=100, batch_size=32)

        save_gan_to_mongo(model_name)
        return {"message": f"Model '{model_name}' trained and saved in MongoDB."}

    except Exception as e:
        return {"error": str(e)}

from typing import Optional, Dict, Any

class GenerateRequest(BaseModel):
    model_name: str
    subject: str
    predicate: str
    num_samples: int = 1
    distribution: Optional[str] = "normal"
    dist_params: Optional[Dict[str, Any]] = None

@app.post("/gan/load-and-generate")
async def load_and_generate_gan_data(request: GenerateRequest):
    model_name = request.model_name

    try:
        
        if model_name not in loaded_models:
            model = await load_gan_from_mongo_async(model_name)
        else:
            model = loaded_models[model_name]

        factorized_data = model["factorized_data"]
        generator = model["generator"]

        
        print("Factorized predicates:", factorized_data["predicate_uniques"])

        
        def map_local_name(factorized_data, local_name, key="predicate"):
            return factorized_data.get(f"{key}_map", {}).get(local_name, local_name)

        subject_mapped = map_local_name(factorized_data, request.subject, "subject")
        predicate_mapped = map_local_name(factorized_data, request.predicate, "predicate")

        results = generate_synthetic_data(
            model_name=model_name,
            subject_input=subject_mapped,
            predicate_input=predicate_mapped,
            num_samples=request.num_samples,
            distribution=request.distribution,
            dist_params=request.dist_params
        )

        return {"generated_objects": results}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating data: {str(e)}")
    
from db.mongo import gan_collection, vae_collection


from fastapi import FastAPI, HTTPException, Query
from enum import Enum
from typing import List
from motor.motor_asyncio import AsyncIOMotorClient


class ModelType(str, Enum):
    GAN = "GAN"
    VAE = "VAE"

@app.get("/models", response_model=List[str])
async def get_available_models(model_type: ModelType = Query(...)):
    """
    Returns available model names based on model type (GAN or VAE).
    """

    if model_type == ModelType.GAN:
        collection = gan_collection
    elif model_type == ModelType.VAE:
        collection = vae_collection
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

    
    models = await collection.find({}, {"_id": 0, "model_name": 1}).to_list(length=None)

    
    model_list = [model["model_name"] for model in models]

    return model_list


from fastapi import FastAPI, HTTPException
from typing import List

async def fetch_models_from_collection(collection):
    models = await collection.find({}, {"_id": 0, "model_name": 1}).to_list(length=None)
    return [m["model_name"] for m in models]

@app.get("/models/gan", response_model=List[str])
async def get_gan_models():
    try:
        return await fetch_models_from_collection(gan_collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch GAN models: {e}")

@app.get("/models/vae", response_model=List[str])
async def get_vae_models():
    try:
        return await fetch_models_from_collection(vae_collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch VAE models: {e}")

@app.get("/gan_models")
async def get_gan_models():
    models = await gan_collection.find({}, {"model_name": 1, "_id": 0}).to_list(length=None)
    return {"saved_models": [model["model_name"] for model in models]}

@app.get("/vae_models")
async def get_vae_models():
    models = await vae_collection.find({}, {"model_name": 1, "_id": 0}).to_list(length=None)
    return {"saved_models": [model["model_name"] for model in models]}


import os
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from rdflib import Graph, RDF, RDFS, OWL, URIRef
from rdflib.collection import Collection

from models.graph_vae import GraphVAE
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")

mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client["graphvae_db"]
vae_collection = db["vae_models"]

def extract_explicit_triples(owl_path: str):
    g = Graph()
    g.parse(owl_path)
    triples = set()
    EXCLUDED = {str(RDF.type), str(RDFS.subClassOf)}

    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(o, URIRef) and str(p) not in EXCLUDED:
            triples.add((str(s), str(p), str(o)))

    for cls in g.subjects(RDF.type, OWL.Class):
        for restriction in g.objects(cls, RDFS.subClassOf):
            if (restriction, RDF.type, OWL.Restriction) in g:
                prop = g.value(restriction, OWL.onProperty)
                if not prop:
                    continue
                some = g.value(restriction, OWL.someValuesFrom)
                if some:
                    triples.add((str(cls), str(prop), str(some)))
                allv = g.value(restriction, OWL.allValuesFrom)
                if allv:
                    if isinstance(allv, URIRef):
                        triples.add((str(cls), str(prop), str(allv)))
                    for union_list in g.objects(allv, OWL.unionOf):
                        collection = Collection(g, union_list)
                        for item in collection:
                            triples.add((str(cls), str(prop), str(item)))
                
                hasv = g.value(restriction, OWL.hasValue)
                if hasv:
                    triples.add((str(cls), str(prop), str(hasv)))

    return list(triples)


def factorize_triples(triples):
    subjects = sorted(set(t[0] for t in triples))
    predicates = sorted(set(t[1] for t in triples))
    objects = sorted(set(t[2] for t in triples))

    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    predicate_to_idx = {p: i for i, p in enumerate(predicates)}
    object_to_idx = {o: i for i, o in enumerate(objects)}

    sp_to_obj = {}
    for s, p, o in triples:
        key = (s, p)
        if key not in sp_to_obj:
            sp_to_obj[key] = []
        sp_to_obj[key].append(object_to_idx[o])

    return {
        "subjects": subjects,
        "predicates": predicates,
        "objects": objects,
        "subject_to_idx": subject_to_idx,
        "predicate_to_idx": predicate_to_idx,
        "object_to_idx": object_to_idx,
        "sp_to_obj": sp_to_obj
    }


def vae_loss(recon_logits, target, mu, logvar):
    recon_loss = nn.CrossEntropyLoss()(recon_logits, target)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def train_model(X, y, factorized, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphVAE(
        len(factorized["subjects"]),
        len(factorized["predicates"]),
        len(factorized["objects"])
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.LongTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        s_idx = X_tensor[:, 0]
        p_idx = X_tensor[:, 1]
        output, mu, logvar = model(s_idx, p_idx)
        loss = vae_loss(output, y_tensor, mu, logvar)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    return model


@app.post("/graphvae/upload_and_train")
async def upload_and_train(model_name: str, file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    triples = extract_explicit_triples(temp_path)
    if not triples:
        raise HTTPException(status_code=400, detail="No valid triples extracted.")

    factorized = factorize_triples(triples)
    X = np.array([[factorized["subject_to_idx"][s], factorized["predicate_to_idx"][p]] for s, p, o in triples])
    y = np.array([factorized["object_to_idx"][o] for _, _, o in triples])

    model = train_model(X, y, factorized, epochs=100)

    await vae_collection.replace_one(
        {"model_name": model_name},
        {
            "model_name": model_name,
            "model_state": pickle.dumps(model.state_dict()),
            "factorized_data": pickle.dumps(factorized)
        },
        upsert=True
    )

    return {"message": f"Model '{model_name}' trained successfully."}



from typing import Optional


from typing import Optional

class GenerateRequest(BaseModel):
    model_name: str
    subject: str
    predicate: str
    num_samples: int = 3
    distribution_type: Optional[str] = None
    distribution_params: Optional[dict] = None
    encode_categorical: bool = False


import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

def filter_by_distribution(samples, dist_type, params, encode_categorical=False):
    filtered = []

    
    if dist_type == "categorical":
        mode = params.get("mode", "allowed_list")

        if mode not in ["allowed_list", "regex", "top_k"]:
            raise ValueError(f"Unsupported categorical mode: {mode}")

        if mode == "allowed_list":
            allowed = set(params.get("allowed", []))
            filtered = [s for s in samples if s in allowed]

        elif mode == "regex":
            pattern = params.get("pattern")
            if not pattern:
                raise ValueError("Regex pattern must be provided for regex mode")
            filtered = [s for s in samples if re.match(pattern, s)]

        elif mode == "top_k":
            top_k = params.get("top_k", len(samples))
            filtered = samples[:top_k]

    
    elif dist_type == "numeric":
        
        if encode_categorical:
            le = LabelEncoder()
            numeric_samples = le.fit_transform(samples)
        else:
            numeric_samples = []
            for s in samples:
                try:
                    numeric_samples.append(float(s))
                except ValueError:
                    continue

        numeric_samples = np.array(numeric_samples)
        mask = np.ones(len(numeric_samples), dtype=bool)

        mode = params.get("mode", "gaussian")

        if mode == "gaussian":
            mean = params["mean"]
            std = params["std"]
            mask &= np.abs(numeric_samples - mean) <= 3*std
            mask &= numeric_samples >= params.get("truncate_min", -np.inf)
            mask &= numeric_samples <= params.get("truncate_max", np.inf)

        elif mode == "uniform":
            min_val = params["min"]
            max_val = params["max"]
            mask &= (numeric_samples >= min_val) & (numeric_samples <= max_val)

        elif mode == "exponential":
            scale = params["scale"]
            mask &= numeric_samples >= 0

        elif mode == "poisson":
            lam = params["lambda"]
            mask &= np.round(numeric_samples) == numeric_samples

        elif mode == "beta":
            alpha = params["alpha"]
            beta = params["beta"]
            min_val = params.get("min", 0)
            max_val = params.get("max", 1)
            norm_samples = (numeric_samples - min_val) / (max_val - min_val)
            mask &= (norm_samples >= 0) & (norm_samples <= 1)

        else:
            raise ValueError(f"Unsupported numeric mode: {mode}")

        filtered = [samples[i] for i, keep in enumerate(mask) if keep]

    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

    return filtered


from pydantic import BaseModel

class DistributionRequest(BaseModel):
    generated_objects: list[str]
    distribution_type: str
    params: dict
    encode_categorical: bool = False

@app.post("/graphvae/filter_distribution")
async def filter_distribution(req: DistributionRequest):
    try:
        filtered = filter_by_distribution(
            req.generated_objects,
            req.distribution_type,
            req.params,
            encode_categorical=req.encode_categorical
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"filtered_objects": filtered}



@app.post("/graphvae/generate")
async def generate_vae(req: GenerateRequest):
    
    doc = await vae_collection.find_one({"model_name": req.model_name})
    if not doc:
        raise HTTPException(status_code=404, detail="Model not found")

    factorized = pickle.loads(doc["factorized_data"])
    model = GraphVAE(
        len(factorized["subjects"]),
        len(factorized["predicates"]),
        len(factorized["objects"])
    )
    model.load_state_dict(pickle.loads(doc["model_state"]))
    model.eval()

    if req.subject not in factorized["subject_to_idx"] or req.predicate not in factorized["predicate_to_idx"]:
        raise HTTPException(status_code=400, detail="Unknown subject or predicate")
    if (req.subject, req.predicate) not in factorized["sp_to_obj"]:
        raise HTTPException(status_code=400, detail="Invalid subject–predicate combination")

    s_idx = torch.LongTensor([factorized["subject_to_idx"][req.subject]])
    p_idx = torch.LongTensor([factorized["predicate_to_idx"][req.predicate]])
    valid_objects = factorized["sp_to_obj"][(req.subject, req.predicate)]

    
    results = []
    max_attempts = req.num_samples * 5
    attempts = 0

    with torch.no_grad():
        while len(results) < req.num_samples and attempts < max_attempts:
            logits, _, _ = model(s_idx, p_idx)
            probs = torch.softmax(logits, dim=1)
            mask = torch.zeros_like(probs)
            mask[:, valid_objects] = 1
            probs = probs * mask
            if probs.sum() == 0:
                attempts += 1
                continue
            probs = probs / probs.sum()
            obj_idx = torch.multinomial(probs, 1).item()
            results.append(factorized["objects"][obj_idx])
            attempts += 1

    
    if req.distribution_type and req.distribution_params:
        results = filter_by_distribution(
            results,
            req.distribution_type,
            req.distribution_params,
            encode_categorical=req.encode_categorical
        )

    
    if not results:
        raise HTTPException(
            status_code=500,
            detail="No samples matched the distribution. Try relaxing the filter."
        )

    return {"generated_objects": results[:req.num_samples]}

@app.get("/graphvae/subject_predicates/{model_name}")
async def list_subject_predicates(model_name: str):
    doc = await vae_collection.find_one({"model_name": model_name})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    try:
        factorized = pickle.loads(doc["factorized_data"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load factorized data: {str(e)}")

    sp_map = factorized.get("sp_to_obj")
    if sp_map is None:
        raise HTTPException(status_code=500, detail="factorized_data missing 'sp_to_obj'")

    subject_predicates = {}
    for (subject, predicate) in sp_map.keys():
        subject_predicates.setdefault(subject, []).append(predicate)

    return {
        "model_name": model_name,
        "subject_predicates": subject_predicates
    }





class DistributionRequest(BaseModel):
    generated_objects: list[str]
    distribution_type: str
    params: dict

import numpy as np
import numpy as np
from sklearn.preprocessing import LabelEncoder



import os
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from rdflib import Graph, RDF, RDFS, OWL, URIRef
from rdflib.collection import Collection

from models.graph_gan import GraphGenerator, GraphDiscriminator
from motor.motor_asyncio import AsyncIOMotorClient


gan_collection = db["gan_models"]

def extract_explicit_triples(owl_path: str):
    g = Graph()
    g.parse(owl_path)
    triples = set()
    EXCLUDED = {str(RDF.type), str(RDFS.subClassOf)}

    
    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(o, URIRef) and str(p) not in EXCLUDED:
            triples.add((str(s), str(p), str(o)))

    
    for cls in g.subjects(RDF.type, OWL.Class):
        for restriction in g.objects(cls, RDFS.subClassOf):
            if (restriction, RDF.type, OWL.Restriction) in g:
                prop = g.value(restriction, OWL.onProperty)
                if not prop:
                    continue
                some = g.value(restriction, OWL.someValuesFrom)
                if some:
                    triples.add((str(cls), str(prop), str(some)))
                allv = g.value(restriction, OWL.allValuesFrom)
                if allv:
                    if isinstance(allv, URIRef):
                        triples.add((str(cls), str(prop), str(allv)))
                    for union_list in g.objects(allv, OWL.unionOf):
                        collection = Collection(g, union_list)
                        for item in collection:
                            triples.add((str(cls), str(prop), str(item)))
                hasv = g.value(restriction, OWL.hasValue)
                if hasv:
                    triples.add((str(cls), str(prop), str(hasv)))

    return list(triples)

def factorize_triples(triples):
    subjects = sorted(set(t[0] for t in triples))
    predicates = sorted(set(t[1] for t in triples))
    objects = sorted(set(t[2] for t in triples))

    subject_to_idx = {s: i for i, s in enumerate(subjects)}
    predicate_to_idx = {p: i for i, p in enumerate(predicates)}
    object_to_idx = {o: i for i, o in enumerate(objects)}

    sp_to_obj = {}
    for s, p, o in triples:
        key = (s, p)
        if key not in sp_to_obj:
            sp_to_obj[key] = []
        sp_to_obj[key].append(object_to_idx[o])

    return {
        "subjects": subjects,
        "predicates": predicates,
        "objects": objects,
        "subject_to_idx": subject_to_idx,
        "predicate_to_idx": predicate_to_idx,
        "object_to_idx": object_to_idx,
        "sp_to_obj": sp_to_obj
    }

def train_gan(X, y, factorized, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = GraphGenerator(len(factorized["subjects"]), len(factorized["predicates"]), len(factorized["objects"])).to(device)
    D = GraphDiscriminator(len(factorized["subjects"]), len(factorized["predicates"]), len(factorized["objects"])).to(device)

    optim_G = optim.Adam(G.parameters(), lr=0.001)
    optim_D = optim.Adam(D.parameters(), lr=0.001)

    criterion = nn.BCELoss()
    X_tensor = torch.LongTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)

    for epoch in range(epochs):
        for i in range(len(X_tensor)):
            s_idx = X_tensor[i, 0].unsqueeze(0)
            p_idx = X_tensor[i, 1].unsqueeze(0)
            real_obj_idx = y_tensor[i].unsqueeze(0)

            
            D.zero_grad()
            real_logits = D(s_idx, p_idx, real_obj_idx)
            real_labels = torch.ones_like(real_logits)
            loss_real = criterion(real_logits, real_labels)

            fake_obj_idx = G(s_idx, p_idx).argmax(dim=1)
            fake_logits = D(s_idx, p_idx, fake_obj_idx.detach())
            fake_labels = torch.zeros_like(fake_logits)
            loss_fake = criterion(fake_logits, fake_labels)

            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optim_D.step()

            
            G.zero_grad()
            fake_logits = D(s_idx, p_idx, fake_obj_idx)
            loss_G = criterion(fake_logits, torch.ones_like(fake_logits))
            loss_G.backward()
            optim_G.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: D_loss={loss_D.item():.4f}, G_loss={loss_G.item():.4f}")

    return G, D

@app.post("/graphgan/upload_and_train")
async def upload_and_train(model_name: str, file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    triples = extract_explicit_triples(temp_path)
    if not triples:
        raise HTTPException(status_code=400, detail="No valid triples extracted.")

    factorized = factorize_triples(triples)
    X = np.array([[factorized["subject_to_idx"][s], factorized["predicate_to_idx"][p]] for s, p, o in triples])
    y = np.array([factorized["object_to_idx"][o] for _, _, o in triples])

    G, D = train_gan(X, y, factorized, epochs=100)

    await gan_collection.replace_one(
        {"model_name": model_name},
        {
            "model_name": model_name,
            "G_state": pickle.dumps(G.state_dict()),
            "D_state": pickle.dumps(D.state_dict()),
            "factorized_data": pickle.dumps(factorized)
        },
        upsert=True
    )

    return {"message": f"GraphGAN '{model_name}' trained successfully."}


from fastapi import HTTPException
import torch
from pydantic import BaseModel
from typing import Optional

class GenerateRequest(BaseModel):
    model_name: str
    subject: str
    predicate: str
    num_samples: int = 3
    distribution_type: Optional[str] = None
    distribution_params: Optional[dict] = None
    encode_categorical: bool = False

@app.post("/graphgan/generate")
async def generate_gan(req: GenerateRequest):
    
    doc = await gan_collection.find_one({"model_name": req.model_name})
    if not doc:
        raise HTTPException(status_code=404, detail="Model not found")

    factorized = pickle.loads(doc["factorized_data"])
    G = GraphGenerator(len(factorized["subjects"]), len(factorized["predicates"]), len(factorized["objects"]))
    G.load_state_dict(pickle.loads(doc["G_state"]))
    G.eval()

    if req.subject not in factorized["subject_to_idx"] or req.predicate not in factorized["predicate_to_idx"]:
        raise HTTPException(status_code=400, detail="Unknown subject or predicate")
    if (req.subject, req.predicate) not in factorized["sp_to_obj"]:
        raise HTTPException(status_code=400, detail="Invalid subject–predicate combination")

    s_idx = torch.LongTensor([factorized["subject_to_idx"][req.subject]])
    p_idx = torch.LongTensor([factorized["predicate_to_idx"][req.predicate]])
    valid_objects = factorized["sp_to_obj"][(req.subject, req.predicate)]

    
    results = []
    max_attempts = req.num_samples * 5
    attempts = 0

    with torch.no_grad():
        while len(results) < req.num_samples and attempts < max_attempts:
            logits = G(s_idx, p_idx)
            probs = torch.softmax(logits, dim=1)
            mask = torch.zeros_like(probs)
            mask[:, valid_objects] = 1
            probs = probs * mask
            if probs.sum() == 0:
                attempts += 1
                continue
            probs = probs / probs.sum()
            obj_idx = torch.multinomial(probs, 1).item()
            results.append(factorized["objects"][obj_idx])
            attempts += 1

    
    if req.distribution_type and req.distribution_params:
        results = filter_by_distribution(
            results,
            req.distribution_type,
            req.distribution_params,
            encode_categorical=req.encode_categorical
        )

    
    if not results:
        raise HTTPException(status_code=500, detail="No samples matched the distribution. Try relaxing the filter.")

    return {"generated_objects": results[:req.num_samples]}

@app.get("/graphgan/subject_predicates/{model_name}")
async def list_subject_predicates(model_name: str):
    doc = await gan_collection.find_one({"model_name": model_name})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    try:
        factorized = pickle.loads(doc["factorized_data"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load factorized data: {str(e)}")

    sp_map = factorized.get("sp_to_obj")
    if sp_map is None:
        raise HTTPException(status_code=500, detail="factorized_data missing 'sp_to_obj'")

    subject_predicates = {}
    for (subject, predicate) in sp_map.keys():
        subject_predicates.setdefault(subject, []).append(predicate)

    return {
        "model_name": model_name,
        "subject_predicates": subject_predicates
    }



import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

def filter_by_distribution(samples, dist_type, params, encode_categorical=False):
    filtered = []

    
    if dist_type == "categorical":
        mode = params.get("mode", "allowed_list")

        if mode not in ["allowed_list", "regex", "top_k"]:
            raise ValueError(f"Unsupported categorical mode: {mode}")

        if mode == "allowed_list":
            allowed = set(params.get("allowed", []))
            filtered = [s for s in samples if s in allowed]

        elif mode == "regex":
            pattern = params.get("pattern")
            if not pattern:
                raise ValueError("Regex pattern must be provided for regex mode")
            filtered = [s for s in samples if re.match(pattern, s)]

        elif mode == "top_k":
            top_k = params.get("top_k", len(samples))
            filtered = samples[:top_k]

    
    elif dist_type == "numeric":
        
        if encode_categorical:
            le = LabelEncoder()
            numeric_samples = le.fit_transform(samples)
        else:
            numeric_samples = []
            for s in samples:
                try:
                    numeric_samples.append(float(s))
                except ValueError:
                    continue

        numeric_samples = np.array(numeric_samples)
        mask = np.ones(len(numeric_samples), dtype=bool)

        mode = params.get("mode", "gaussian")

        if mode == "gaussian":
            mean = params["mean"]
            std = params["std"]
            mask &= np.abs(numeric_samples - mean) <= 3*std
            mask &= numeric_samples >= params.get("truncate_min", -np.inf)
            mask &= numeric_samples <= params.get("truncate_max", np.inf)

        elif mode == "uniform":
            min_val = params["min"]
            max_val = params["max"]
            mask &= (numeric_samples >= min_val) & (numeric_samples <= max_val)

        elif mode == "exponential":
            scale = params["scale"]
            mask &= numeric_samples >= 0

        elif mode == "poisson":
            lam = params["lambda"]
            mask &= np.round(numeric_samples) == numeric_samples

        elif mode == "beta":
            alpha = params["alpha"]
            beta = params["beta"]
            min_val = params.get("min", 0)
            max_val = params.get("max", 1)
            norm_samples = (numeric_samples - min_val) / (max_val - min_val)
            mask &= (norm_samples >= 0) & (norm_samples <= 1)

        else:
            raise ValueError(f"Unsupported numeric mode: {mode}")

        filtered = [samples[i] for i, keep in enumerate(mask) if keep]

    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

    return filtered



import time
import requests
import json
import re
from collections import deque
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

LLM_CACHE = {}
RECENT_RESPONSES_HISTORY = deque(maxlen=5)
RECENT_RESPONSES = set()
RECENT_CACHE_LIMIT = 50

DATATYPE_MAP = {
    "http://www.w3.org/2001/XMLSchema#string": "text",
    "http://www.w3.org/2001/XMLSchema#integer": "integer",
    "http://www.w3.org/2001/XMLSchema#decimal": "decimal number",
    "http://www.w3.org/2001/XMLSchema#boolean": "true/false value",
    "http://www.w3.org/2001/XMLSchema#date": "date (YYYY-MM-DD)",
    "http://www.w3.org/2001/XMLSchema#dateTime": "datetime (YYYY-MM-DDTHH:MM:SS)",
    "http://www.w3.org/ns/shacl#IRI": "IRI (e.g., http://example.org/resource/123)"
}

def simplify_key(path: str) -> str:
    return path.split("/")[-1]

def generate_prompt(subject_predicate: str, user_message: str = "", 
                    distribution_type: str = None, distribution_params: dict = None) -> str:
    """
    Build human-readable prompt for LLM using subject-predicate style.
    Force JSON output with "value" key.
    """
    base = f"""
Generate synthetic RDF values for the triple: {subject_predicate}

Return ONLY a JSON object with this format:
{{ "value": <generated_value> }}

Rules:
- No explanations, no extra text
- Only output valid JSON
- Generate {distribution_params.get('num_samples',1) if distribution_params else 1} sample(s)
"""
    if distribution_type == "categorical" and distribution_params:
        allowed = distribution_params.get("allowed_list", [])
        if allowed:
            base += f"\nAllowed values: {allowed}"
    if distribution_type == "numeric" and distribution_params:
        base += f"\nNumeric constraints: {distribution_params}"
    if user_message:
        base += f"\nExtra instruction: {user_message}"

    return base.strip()

def parse_llm_response(text: str):
    """
    Parse LLM output robustly, expecting JSON or JSON-like responses.
    """
    if not text or not text.strip():
        raise ValueError("Empty LLM output")
    
    
    try:
        data = json.loads(text)
        if "value" in data:
            return data["value"]
    except Exception:
        pass

    
    match = re.search(r'\{\s*"value"\s*:\s*(.*?)\s*\}', text)
    if match:
        val = match.group(1).strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        try:
            if '.' in val:
                val = float(val)
            else:
                val = int(val)
        except:
            pass
        if val != "":
            return val

    
    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
            return line

    raise ValueError(f"Invalid LLM output: {text}")

def fetch_from_llm(prompt: str, num_samples: int = 5):
    url = "http://host.docker.internal:11434/api/generate"
    payload = {
        "model": "llama3:8b",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.8, "num_predict": num_samples}
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        text = response.json().get("response", "")
        candidates = []
        for line in text.strip().split("\n"):
            try:
                val = parse_llm_response(line)
                if val is not None and str(val).strip() != "":
                    candidates.append(val)
            except Exception:
                continue
        return list(dict.fromkeys(candidates))
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return []

def enforce_distribution(value, distribution_type, params):
    if not distribution_type:
        return value
    if distribution_type == "categorical" and params:
        allowed = params.get("allowed_list", [])
        if value not in allowed:
            return None
    if distribution_type == "numeric" and params:
        try:
            value = float(value)
        except:
            return None
        min_v = params.get("min")
        max_v = params.get("max")
        if min_v is not None and value < min_v:
            return None
        if max_v is not None and value > max_v:
            return None
    return value

def build_llm_request(shape: str, path: str, datatype: str,
                      user_interactive_message: str = "",
                      num_samples: int = 1,
                      distribution_type: str = None,
                      distribution_params: dict = None,
                      encode_categorical: bool = False) -> dict:
    """
    Build JSON payload for /llm/generate endpoint
    """
    payload = {
        "shape": shape,
        "path": path,
        "datatype": datatype,
        "user_interactive_message": [user_interactive_message] if user_interactive_message else [],
        "num_samples": num_samples,
        "distribution_type": distribution_type if distribution_type else "",
        "distribution_params": distribution_params if distribution_params else {},
        "encode_categorical": encode_categorical
    }
    return payload

def generates_llm_data(
    shape: str,
    path: str,
    datatype: str,
    user_interactive_message: str = "",
    num_samples: int = 1,
    distribution_type: str = None,
    distribution_params: dict = None,
    max_retries: int = 50,
    batch_size: int = 10,
    encode_categorical: bool = False
) -> dict:
    """
    Generate RDF values for a subject-predicate pair in the new JSON style.
    """
    payload = build_llm_request(
        shape=shape,
        path=path,
        datatype=datatype,
        user_interactive_message=user_interactive_message,
        num_samples=num_samples,
        distribution_type=distribution_type,
        distribution_params=distribution_params,
        encode_categorical=encode_categorical
    )

    subject_predicate = f"{shape.split('#')[-1]} {path.split('#')[-1]}"
    prompt = generate_prompt(subject_predicate, user_interactive_message, distribution_type, distribution_params)

    results = set()
    attempt_count = 0
    backoff = 1

    while len(results) < num_samples and attempt_count < max_retries:
        if prompt not in LLM_CACHE or not LLM_CACHE[prompt]:
            LLM_CACHE[prompt] = fetch_from_llm(prompt, num_samples=batch_size)

        while LLM_CACHE[prompt] and len(results) < num_samples:
            value = LLM_CACHE[prompt].pop()
            if value in results or value in RECENT_RESPONSES or value in RECENT_RESPONSES_HISTORY:
                continue

            value = enforce_distribution(value, distribution_type, distribution_params)
            if value is None:
                continue

            results.add(value)
            RECENT_RESPONSES.add(value)
            if len(RECENT_RESPONSES) > RECENT_CACHE_LIMIT:
                RECENT_RESPONSES.pop()
            RECENT_RESPONSES_HISTORY.append(value)

        if len(results) < num_samples:
            attempt_count += 1
            time.sleep(backoff)
            backoff *= 2
            LLM_CACHE[prompt] = fetch_from_llm(prompt, num_samples=batch_size)

    if not results:
        raise ValueError(f"LLM failed to generate valid outputs for '{subject_predicate}'")

    payload["generated_samples"] = list(results)[:num_samples]
    return payload

class LLMGenerateRequest(BaseModel):
    shape: str
    path: str
    datatype: str
    user_interactive_message: str = ""
    num_samples: int = 1
    distribution_type: Optional[str] = None
    distribution_params: Optional[dict] = None
    encode_categorical: bool = False


@app.post("/llm/generate")
async def generate_llm(req: LLMGenerateRequest):
    try:
        result = generates_llm_data(
            shape=req.shape,
            path=req.path,
            datatype=req.datatype,
            user_interactive_message=req.user_interactive_message,
            num_samples=req.num_samples,
            distribution_type=req.distribution_type,
            distribution_params=req.distribution_params,
            encode_categorical=req.encode_categorical
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    if not result.get("generated_samples"):
        raise HTTPException(
            status_code=500,
            detail="No samples matched the distribution. Try relaxing the filter."
        )

    
    return {"generated_samples": result["generated_samples"]}

import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from rdflib import Graph, Namespace, RDF, URIRef, BNode
from typing import List, Dict, Tuple, Any


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

shape_map_storage: List[Dict[str, Any]] = []

DIST_NS = "http://example.org/distribution#"
SH = Namespace("http://www.w3.org/ns/shacl#")

from rdflib import Graph, RDF, BNode

def parse_rdf_list(graph: Graph, node) -> list:
    """Recursively parse an RDF list into a Python list."""
    result = []
    while node and node != RDF.nil:
        first = graph.value(node, RDF.first)
        if first is not None:
            result.append(str(first))
        node = graph.value(node, RDF.rest)
    return result

def extract_distribution_info(constraints: list, g: Graph) -> dict:
    """
    Extract distribution info from SHACL constraints.
    Converts RDF lists (BNodes) to Python lists automatically.
    """
    dist_info = {}
    for c in constraints:
        for key, val in c.items():
            if key.startswith(DIST_NS):
                short_key = key[len(DIST_NS):]
                
                if isinstance(val, BNode):
                    dist_info[short_key] = parse_rdf_list(g, val)
                else:
                    dist_info[short_key] = str(val)
    return dist_info


def extract_path_and_datatype(constraints: List[Dict[str, str]]) -> Tuple[str, str]:
    """Extract sh:path and sh:datatype (or IRI nodeKind) from constraints."""
    path = None
    datatype = "http://www.w3.org/2001/XMLSchema#string"
    
    for c in constraints:
        if str(SH.path) in c:
            path = c[str(SH.path)]
        if str(SH.datatype) in c:
            datatype = c[str(SH.datatype)]
        elif str(SH.nodeKind) in c and c[str(SH.nodeKind)] == str(SH.IRI):
            datatype = "IRI"
    
    return path, datatype

def get_cardinality(constraints: List[Dict[str, str]]) -> Tuple[int, int]:
    """Extract minCount and maxCount from constraints."""
    min_count = 1
    max_count = 1
    for c in constraints:
        if str(SH.minCount) in c:
            min_count = int(c[str(SH.minCount)])
        if str(SH.maxCount) in c:
            max_count = int(c[str(SH.maxCount)])
    return min_count, max_count


from rdflib import Graph, RDF, BNode
from rdflib.namespace import SH, XSD

def parse_shacl(file_path: str) -> list[dict]:
    """Parse a SHACL file and extract shapes, properties, constraints, and distributions."""
    g = Graph()
    g.parse(file_path, format="turtle")
    shapes = []

    for s in g.subjects(RDF.type, SH.NodeShape):
        
        target_classes = list(g.objects(s, SH.targetClass))
        if not target_classes:
            continue
        shape_entry = {
            "shape_iri": str(target_classes[0]),
            "target_classes": [str(t) for t in target_classes],
            "properties": []
        }

        for prop in g.objects(s, SH.property):
            prop_entry = {"property": str(prop), "constraints": []}

            
            for pred, val in g.predicate_objects(prop):
                prop_entry["constraints"].append({str(pred): val})

            
            prop_entry["distribution"] = extract_distribution_info(prop_entry["constraints"], g)

            
            path, datatype = extract_path_and_datatype(prop_entry["constraints"])
            min_count, max_count = get_cardinality(prop_entry["constraints"])
            prop_entry.update({
                "path": path,
                "datatype": datatype,
                "min_count": min_count,
                "max_count": max_count
            })

            shape_entry["properties"].append(prop_entry)

        shapes.append(shape_entry)

    return shapes

@app.post("/upload_shacl")
async def upload_shacl(file: UploadFile = File(...)):
    os.makedirs("shacl_files", exist_ok=True)
    file_location = f"shacl_files/{file.filename}"

    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    global shape_map_storage
    shape_map_storage = parse_shacl(file_location)

    return {
        "message": f"SHACL file uploaded successfully: {file_location}",
        "shape_map": shape_map_storage
    }


def shacl_to_json_schema(shapes: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """
    Convert parsed SHACL shapes into a generator-friendly JSON schema.
    
    Output format per property:
    {
        "path": "...",
        "datatype": "...",
        "min_count": 1,
        "max_count": 1,
        "distribution": {...}
    }
    """
    schema = []

    for shape in shapes:
        for prop in shape.get("properties", []):
            entry = {
                "path": prop.get("path"),
                "datatype": prop.get("datatype", "http://www.w3.org/2001/XMLSchema#string"),
                "min_count": prop.get("min_count", 1),
                "max_count": prop.get("max_count", 1),
                "distribution": prop.get("distribution", {})
            }
            schema.append(entry)

    return schema


@app.get("/shacl/json_schema")
async def get_shacl_json_schema():
    """
    Return a generator-ready JSON schema extracted from the uploaded SHACL shapes.
    """
    if not shape_map_storage:
        return {"message": "No SHACL file uploaded yet.", "json_schema": []}

    json_schema = shacl_to_json_schema(shape_map_storage)
    return {
        "message": "JSON schema generated successfully.",
        "json_schema": json_schema
    }



from fastapi import FastAPI, HTTPException

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import pickle
import numpy as np
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import XSD


class PropertySchema(BaseModel):
    shape: str
    path: str
    datatype: str
    min_count: int = 1
    max_count: int = 1
    distribution_type: Optional[str] = None
    distribution_params: Optional[Dict[str, Any]] = None
    model_type: str
    model_name: Optional[str] = None


class GenerateRequest(BaseModel):
    model_type: str
    model_name: str = None
    json_schema: List[PropertySchema]


from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import XSD

SHACL_NS = Namespace("http://www.w3.org/ns/shacl#")

from fastapi import HTTPException
from rdflib import Graph, URIRef, Literal
import torch
import pickle
import numpy as np


def clean_llm_output(val):

    if isinstance(val, list):
        val = val[0]

    if isinstance(val, str):

        val = val.strip()

        
        bad_prefixes = [
            "Here is",
            "Example",
            "Generated",
            "Output",
            ":"
        ]

        for p in bad_prefixes:
            if val.startswith(p):
                val = val.split(":")[-1].strip()

    return val



async def load_custom_llm(model_name: str):
    doc = await llm_collection.find_one({"model_name": model_name})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Custom LLM '{model_name}' not found")

    file_ids = doc["file_ids"]

    
    model_bytes = io.BytesIO()
    await fs_bucket.download_to_stream(file_ids["model"], model_bytes)
    model_bytes.seek(0)

    tokenizer_bytes = io.BytesIO()
    await fs_bucket.download_to_stream(file_ids["tokenizer"], tokenizer_bytes)
    tokenizer_bytes.seek(0)

    sp_to_obj_bytes = io.BytesIO()
    await fs_bucket.download_to_stream(file_ids["sp_to_obj"], sp_to_obj_bytes)
    sp_to_obj_bytes.seek(0)

    tokenizer = pickle.load(tokenizer_bytes)
    sp_to_obj = pickle.load(sp_to_obj_bytes)

    model = KGModel(len(tokenizer.sp_vocab), len(tokenizer.o_vocab))
    model.load_state_dict(torch.load(model_bytes))
    model.eval()

    return {
        "model_name": model_name,
        "model": model,
        "tokenizer": tokenizer,
        "sp_to_obj": sp_to_obj
    }


from fastapi import Query

@app.post("/generate_from_shacl")
async def generate_from_shacl(req: List[PropertySchema], num_samples: int = Query(1, ge=1)):
    """
    Generate multiple samples from SHACL schema.
    `num_samples` applies to the entire request (not per property).
    """

    all_samples_result = []
    all_samples_rdf = []
    models_cache = {}

    for sample_idx in range(num_samples):
        rdf_graph = Graph()
        sample_result = {}

        for prop in req:
            
            
            if prop.model_type not in ["LLM", "VAE", "GAN", "CUSTOM_LLM"]:
                raise HTTPException(status_code=400, detail=f"Invalid model_type for {prop.path}")

            factorized, model, G = None, None, None

            if prop.model_type in ["VAE", "GAN"]:
                if prop.model_name in models_cache:
                    factorized, model, G = models_cache[prop.model_name]
                else:
                    if prop.model_type == "VAE":
                        doc = await vae_collection.find_one({"model_name": prop.model_name})
                        if not doc:
                            raise HTTPException(status_code=404, detail="VAE model not found")
                        factorized = pickle.loads(doc["factorized_data"])
                        model = GraphVAE(len(factorized["subjects"]),
                                         len(factorized["predicates"]),
                                         len(factorized["objects"]))
                        model.load_state_dict(pickle.loads(doc["model_state"]))
                        model.eval()
                        models_cache[prop.model_name] = (factorized, model, None)
                    else:  # GAN
                        doc = await gan_collection.find_one({"model_name": prop.model_name})
                        if not doc:
                            raise HTTPException(status_code=404, detail="GAN model not found")
                        factorized = pickle.loads(doc["factorized_data"])
                        G = GraphGenerator(len(factorized["subjects"]),
                                           len(factorized["predicates"]),
                                           len(factorized["objects"]))
                        G.load_state_dict(pickle.loads(doc["G_state"]))
                        G.eval()
                        models_cache[prop.model_name] = (factorized, None, G)

            obj_value = None

            if prop.model_type == "LLM":
                val_list = generates_llm_data(
                    shape=prop.shape,
                    path=prop.path,
                    datatype=prop.datatype,
                    user_interactive_message="",
                    num_samples=1,
                    distribution_type=prop.distribution_type,
                    distribution_params=prop.distribution_params
                )
                obj_value = val_list["generated_samples"][0]
            elif prop.model_type == "CUSTOM_LLM":
                    
                    if prop.model_name in models_cache:
                        custom_llm_loaded = models_cache[prop.model_name]
                    else:
                        custom_llm_loaded = await load_custom_llm(prop.model_name)
                        models_cache[prop.model_name] = custom_llm_loaded

                    model = custom_llm_loaded["model"]
                    tokenizer = custom_llm_loaded["tokenizer"]
                    sp_to_obj = custom_llm_loaded["sp_to_obj"]

                    if (prop.shape, prop.path) in sp_to_obj:
                        sp_token = f"{prop.shape}|{prop.path}"
                        sp_id = tokenizer.sp_vocab[sp_token]
                        valid_objects = sp_to_obj[(prop.shape, prop.path)]

                        with torch.no_grad():
                            logits = model(torch.tensor([sp_id]), torch.tensor([0]))
                            probs = torch.softmax(logits, dim=1)

                            mask = torch.zeros_like(probs)
                            mask[:, valid_objects] = 1
                            probs = probs * mask

                            if probs.sum() > 0:
                                probs = probs / probs.sum()
                                obj_idx = torch.multinomial(probs, 1).item()
                                obj_value = tokenizer.o_inverse[obj_idx]
                            else:
                                obj_value = "Unknown"
                    else:
                        obj_value = "Unknown"

            else:
                
                if prop.shape in factorized["subject_to_idx"] and prop.path in factorized["predicate_to_idx"]:
                    s_idx = torch.LongTensor([factorized["subject_to_idx"][prop.shape]])
                    p_idx = torch.LongTensor([factorized["predicate_to_idx"][prop.path]])
                    valid_objects = factorized.get("sp_to_obj", {}).get((prop.shape, prop.path), [])

                    if valid_objects:
                        with torch.no_grad():
                            if prop.model_type == "VAE":
                                logits, _, _ = model(s_idx, p_idx)
                            else:
                                logits = G(s_idx, p_idx)

                            probs = torch.softmax(logits, dim=1)
                            mask = torch.zeros_like(probs)
                            mask[:, valid_objects] = 1
                            probs = probs * mask
                            if probs.sum() > 0:
                                probs = probs / probs.sum()
                                obj_idx = torch.multinomial(probs, 1).item()
                                obj_value = factorized["objects"][obj_idx]

            if obj_value is None:
                if prop.distribution_type == "categorical":
                    allowed = prop.distribution_params.get("allowed_list", ["UnknownValue"])
                    weights = prop.distribution_params.get("probabilities", None)
                    if weights:
                        weights = [float(w) for w in weights]
                        if len(weights) != len(allowed):
                            weights = None
                    obj_value = np.random.choice(allowed, p=weights if weights else None)
                elif prop.distribution_type == "numeric":
                    mean = float(prop.distribution_params.get("mean", 0))
                    std = float(prop.distribution_params.get("std", 1))
                    min_val = float(prop.distribution_params.get("min", -np.inf))
                    max_val = float(prop.distribution_params.get("max", np.inf))
                    val = np.random.normal(mean, std)
                    obj_value = max(min_val, min(val, max_val))
                else:
                    obj_value = "UnknownValue"

            sample_result[prop.path] = obj_value

            subj_uri = URIRef(prop.shape)
            pred_uri = URIRef(prop.path)

            if prop.datatype.lower() == "iri":
                obj = URIRef(obj_value)
            elif prop.datatype.startswith("http://www.w3.org/2001/XMLSchema#"):
                obj = Literal(obj_value, datatype=URIRef(prop.datatype))
            else:
                obj = Literal(obj_value)

            rdf_graph.add((subj_uri, pred_uri, obj))

        all_samples_result.append(sample_result)
        all_samples_rdf.append(rdf_graph.serialize(format="turtle"))

    return {
        "rdf_turtle_samples": all_samples_rdf,
        "generated_data_samples": all_samples_result
    }


@app.post("/upload_shacl_and_extract_schema")
async def upload_shacl_and_extract_schema(file: UploadFile = File(...)):
    
    os.makedirs("shacl_files", exist_ok=True)
    file_location = f"shacl_files/{file.filename}"
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    
    shapes = parse_shacl(file_location)
    json_schema = []

    g = Graph()
    g.parse(file_location, format="turtle")

    for shape in shapes:
        shape_iri = shape.get("shape_iri")
        for prop in shape["properties"]:
            path, datatype = extract_path_and_datatype(prop["constraints"])
            min_count, max_count = get_cardinality(prop["constraints"])
            distribution = prop.get("distribution", {})

            dist_type = distribution.get("distribution")
            dist_params = {}

            
            if dist_type == "categorical":
                allowed_list = distribution.get("categories", ["ExampleValue"])
                if isinstance(allowed_list, BNode):
                    allowed_list = parse_rdf_list(g, allowed_list)
                probabilities = distribution.get("probabilities", [1.0])
                if isinstance(probabilities, BNode):
                    probabilities = parse_rdf_list(g, probabilities)

                dist_params = {
                    "allowed_list": allowed_list,
                    "probabilities": probabilities
                }

            
            elif dist_type == "numeric":
                dist_params = {
                    "mean": distribution.get("mean", 10),
                    "std": distribution.get("std", 2),
                    "min": distribution.get("min", 0),
                    "max": distribution.get("max", 20)
                }

            
            if not dist_type:
                if datatype in ["IRI", "http://www.w3.org/2001/XMLSchema#string"]:
                    dist_type = "categorical"
                    dist_params = {"allowed_list": ["ExampleValue"], "probabilities": [1.0]}
                elif datatype in ["http://www.w3.org/2001/XMLSchema#integer", 
                                  "http://www.w3.org/2001/XMLSchema#decimal"]:
                    dist_type = "numeric"
                    dist_params = {"mean": 10, "std": 2, "min": 0, "max": 20}

            json_schema.append({
                "shape": shape_iri,
                "path": path,
                "datatype": datatype,
                "min_count": min_count,
                "max_count": max_count,
                "model_type": None,
                "model_name": None,
                "distribution_type": dist_type,
                "distribution_params": dist_params
            })

    return {"message": "SHACL uploaded and schema extracted", "json_schema": json_schema}


from fastapi.responses import StreamingResponse
import json
import asyncio
import torch
import numpy as np
from rdflib import Graph, URIRef, Literal

@app.post("/generate_from_shacl_stream")
async def generate_from_shacl_stream(req: List[PropertySchema], num_samples: int = 1):
    """
    Streaming version of generate_from_shacl.
    Streams progress updates and final results via SSE.
    """

    async def event_generator():
        all_samples_result = []
        all_samples_rdf = []
        models_cache = {}

        for sample_idx in range(num_samples):
            rdf_graph = Graph()
            sample_result = {}

            for prop in req:
                obj_value = None

                
                if prop.model_type not in ["LLM", "CUSTOM_LLM", "VAE", "GAN"]:
                    raise HTTPException(status_code=400, detail=f"Invalid model_type for {prop.path}")

                factorized, model, G = None, None, None

                
                if prop.model_type in ["VAE", "GAN"]:
                    if prop.model_name in models_cache:
                        factorized, model, G = models_cache[prop.model_name]
                    else:
                        if prop.model_type == "VAE":
                            doc = await vae_collection.find_one({"model_name": prop.model_name})
                            if not doc:
                                raise HTTPException(status_code=404, detail=f"VAE model '{prop.model_name}' not found")
                            factorized = pickle.loads(doc["factorized_data"])
                            model = GraphVAE(len(factorized["subjects"]),
                                             len(factorized["predicates"]),
                                             len(factorized["objects"]))
                            model.load_state_dict(pickle.loads(doc["model_state"]))
                            model.eval()
                            models_cache[prop.model_name] = (factorized, model, None)
                        else:
                            doc = await gan_collection.find_one({"model_name": prop.model_name})
                            if not doc:
                                raise HTTPException(status_code=404, detail=f"GAN model '{prop.model_name}' not found")
                            factorized = pickle.loads(doc["factorized_data"])
                            G = GraphGenerator(len(factorized["subjects"]),
                                               len(factorized["predicates"]),
                                               len(factorized["objects"]))
                            G.load_state_dict(pickle.loads(doc["G_state"]))
                            G.eval()
                            models_cache[prop.model_name] = (factorized, None, G)

                
                if prop.model_type == "LLM":
                    val_list = generates_llm_data(
                        shape=prop.shape,
                        path=prop.path,
                        datatype=prop.datatype,
                        user_interactive_message="",
                        num_samples=1,
                        distribution_type=prop.distribution_type,
                        distribution_params=prop.distribution_params
                    )
                    obj_value = val_list["generated_samples"][0]

                
                elif prop.model_type == "CUSTOM_LLM":
                    if prop.model_name in models_cache:
                        custom_llm_loaded = models_cache[prop.model_name]
                    else:
                        custom_llm_loaded = await load_custom_llm(prop.model_name)
                        models_cache[prop.model_name] = custom_llm_loaded

                    model = custom_llm_loaded["model"]
                    tokenizer = custom_llm_loaded["tokenizer"]
                    sp_to_obj = custom_llm_loaded["sp_to_obj"]

                    if (prop.shape, prop.path) in sp_to_obj:
                        sp_token = f"{prop.shape}|{prop.path}"
                        sp_id = tokenizer.sp_vocab[sp_token]
                        valid_objects = sp_to_obj[(prop.shape, prop.path)]

                        with torch.no_grad():
                            logits = model(torch.tensor([sp_id]), torch.tensor([0]))
                            probs = torch.softmax(logits, dim=1)

                            mask = torch.zeros_like(probs)
                            mask[:, valid_objects] = 1
                            probs = probs * mask

                            if probs.sum() > 0:
                                probs = probs / probs.sum()
                                obj_idx = torch.multinomial(probs, 1).item()
                                obj_value = tokenizer.o_inverse[obj_idx]
                            else:
                                obj_value = "Unknown"
                    else:
                        obj_value = "Unknown"

                
                elif prop.model_type in ["VAE", "GAN"]:
                    if prop.shape in factorized["subject_to_idx"] and prop.path in factorized["predicate_to_idx"]:
                        s_idx = torch.LongTensor([factorized["subject_to_idx"][prop.shape]])
                        p_idx = torch.LongTensor([factorized["predicate_to_idx"][prop.path]])
                        valid_objects = factorized.get("sp_to_obj", {}).get((prop.shape, prop.path), [])

                        if valid_objects:
                            with torch.no_grad():
                                if prop.model_type == "VAE":
                                    logits, _, _ = model(s_idx, p_idx)
                                else:
                                    logits = G(s_idx, p_idx)
                                probs = torch.softmax(logits, dim=1)
                                mask = torch.zeros_like(probs)
                                mask[:, valid_objects] = 1
                                probs = probs * mask
                                if probs.sum() > 0:
                                    probs = probs / probs.sum()
                                    obj_idx = torch.multinomial(probs, 1).item()
                                    obj_value = factorized["objects"][obj_idx]

                
                if obj_value is None:
                    if prop.distribution_type == "categorical":
                        allowed = prop.distribution_params.get("allowed_list", ["UnknownValue"])
                        weights = prop.distribution_params.get("probabilities", None)
                        if weights:
                            weights = [float(w) for w in weights]
                            if len(weights) != len(allowed):
                                weights = None
                        obj_value = np.random.choice(allowed, p=weights if weights else None)
                    elif prop.distribution_type == "numeric":
                        mean = float(prop.distribution_params.get("mean", 0))
                        std = float(prop.distribution_params.get("std", 1))
                        min_val = float(prop.distribution_params.get("min", -np.inf))
                        max_val = float(prop.distribution_params.get("max", np.inf))
                        val = np.random.normal(mean, std)
                        obj_value = max(min_val, min(val, max_val))
                    else:
                        obj_value = "UnknownValue"

                sample_result[prop.path] = obj_value

                
                subj_uri = URIRef(prop.shape)
                pred_uri = URIRef(prop.path)
                if prop.datatype.lower() == "iri":
                    obj = URIRef(obj_value)
                elif prop.datatype.startswith("http://www.w3.org/2001/XMLSchema#"):
                    obj = Literal(obj_value, datatype=URIRef(prop.datatype))
                else:
                    obj = Literal(obj_value)
                rdf_graph.add((subj_uri, pred_uri, obj))

            all_samples_result.append(sample_result)
            all_samples_rdf.append(rdf_graph.serialize(format="turtle"))

            
            progress = int(((sample_idx + 1) / num_samples) * 100)
            yield f"data: {json.dumps({'progress': progress})}\n\n"

            await asyncio.sleep(0.05)

        
        yield f"data: {json.dumps({'progress': 100, 'rdf_turtle_samples': all_samples_rdf, 'generated_data_samples': all_samples_result})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# -------------------------------------Custom-LLM-------------------------------------------------

import datetime
import pickle
import tempfile
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from fastapi import UploadFile, File, HTTPException
from pydantic import BaseModel
from rdflib import Graph

llm_collection = db["llm_models"]



class KGModel(nn.Module):

    def __init__(self, sp_vocab_size, o_vocab_size, hidden=128):
        super().__init__()

        self.sp_embed = nn.Embedding(sp_vocab_size, hidden)
        self.o_embed = nn.Embedding(o_vocab_size, hidden)

        self.transformer = nn.Transformer(
            d_model=hidden,
            nhead=4,
            num_encoder_layers=2
        )

        self.output = nn.Linear(hidden, o_vocab_size)

    def forward(self, sp_ids, o_ids):

        sp_vec = self.sp_embed(sp_ids)
        o_vec = self.o_embed(o_ids)

        x = sp_vec + o_vec
        x = self.transformer(x, x)

        logits = self.output(x)

        return logits


class KGTokenizer:

    def __init__(self):

        self.sp_vocab = {}
        self.o_vocab = {}

        self.sp_inverse = {}
        self.o_inverse = {}

    def build_from_graph(self, triples):

        sp_id = 0
        o_id = 0

        for s, p, o in triples:

            sp_token = f"{s}|{p}"

            if sp_token not in self.sp_vocab:
                self.sp_vocab[sp_token] = sp_id
                self.sp_inverse[sp_id] = sp_token
                sp_id += 1

            if o not in self.o_vocab:
                self.o_vocab[o] = o_id
                self.o_inverse[o_id] = o
                o_id += 1

    def encode(self, s, p, o):

        sp_token = f"{s}|{p}"

        return (
            self.sp_vocab.get(sp_token, -1),
            self.o_vocab.get(o, -1)
        )


class OntologyLoader:

    def __init__(self):
        self.graph = Graph()

    def load(self, path: str):
        self.graph.parse(path)

    def triples(self):

        for s, p, o in self.graph:
            yield str(s), str(p), str(o)



class CustomKGLLM:

    def __init__(self, triples: List[tuple]):

        self.triples = triples

        self.tokenizer = KGTokenizer()
        self.tokenizer.build_from_graph(triples)

        
        self.sp_to_obj = {}

        for s, p, o in triples:

            key = (s, p)

            if key not in self.sp_to_obj:
                self.sp_to_obj[key] = []

            self.sp_to_obj[key].append(self.tokenizer.o_vocab[o])



    async def train(self, epochs=100, lr=0.01, batch_size=32, device=None):

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = KGModel(
            len(self.tokenizer.sp_vocab),
            len(self.tokenizer.o_vocab)
        ).to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        sp_ids = []
        o_ids = []

        for s, p, o in self.triples:
            sp_id, o_id = self.tokenizer.encode(s, p, o)
            if sp_id != -1 and o_id != -1:
                sp_ids.append(sp_id)
                o_ids.append(o_id)

        if not sp_ids:
            return

        sp_ids = torch.tensor(sp_ids, dtype=torch.long, device=device)
        o_ids = torch.tensor(o_ids, dtype=torch.long, device=device)

        self.model.train()

        for epoch in range(epochs):
            perm = torch.randperm(len(sp_ids), device=device)

            for i in range(0, len(sp_ids), batch_size):
                idx = perm[i:i+batch_size]
                sp_batch = sp_ids[idx]
                o_batch = o_ids[idx]

                logits = self.model(sp_batch, torch.zeros_like(sp_batch, device=device))

                loss = loss_fn(logits.view(-1, logits.size(-1)), o_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

        self.model.eval()


    def generate(self, subject, predicate, num_samples):

        if not hasattr(self, "model"):
            return ["Unknown"]

        if (subject, predicate) not in self.sp_to_obj:
            return ["Unknown"]

        sp_token = f"{subject}|{predicate}"
        sp_id = self.tokenizer.sp_vocab[sp_token]

        valid_objects = self.sp_to_obj[(subject, predicate)]

        results = []

        with torch.no_grad():

            for _ in range(num_samples):

                sp_tensor = torch.tensor([sp_id])
                o_tensor = torch.tensor([0])

                logits = self.model(sp_tensor, o_tensor)

                probs = torch.softmax(logits, dim=1)

                mask = torch.zeros_like(probs)
                mask[:, valid_objects] = 1

                probs = probs * mask

                if probs.sum() == 0:
                    continue

                probs = probs / probs.sum()

                obj_idx = torch.multinomial(probs, 1).item()

                results.append(self.tokenizer.o_inverse[obj_idx])

        return results


kg_llm = None


@app.post("/llm/upload_ontology")
async def upload_ontology(file: UploadFile = File(...)):

    content = await file.read()

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".owl")

    temp.write(content)
    temp.close()

    return {"temp_path": temp.name}


@app.post("/llm/load_ontology")
async def load_ontology(temp_path: str):

    global kg_llm

    loader = OntologyLoader()
    loader.load(temp_path)

    triples = list(loader.triples())

    kg_llm = CustomKGLLM(triples)

    return {
        "num_triples": len(triples)
    }


class TrainRequest(BaseModel):
    model_name: str
    epochs: int = 100
    lr: float = 0.01



from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
import io

db = mongo_client["graphvae_db"]
llm_collection = db["llm_models"]
fs_bucket = AsyncIOMotorGridFSBucket(db)

@app.post("/llm/train")
async def train_llm(req: TrainRequest):
    global kg_llm

    if not kg_llm:
        raise HTTPException(status_code=400, detail="Load ontology first")

    
    await kg_llm.train(req.epochs, req.lr)

    
    model_bytes = io.BytesIO()
    torch.save(kg_llm.model.state_dict(), model_bytes)
    model_bytes.seek(0)

    tokenizer_bytes = io.BytesIO()
    pickle.dump(kg_llm.tokenizer, tokenizer_bytes)
    tokenizer_bytes.seek(0)

    sp_to_obj_bytes = io.BytesIO()
    pickle.dump(kg_llm.sp_to_obj, sp_to_obj_bytes)
    sp_to_obj_bytes.seek(0)

    
    existing = await llm_collection.find_one({"model_name": req.model_name})
    if existing:
        for file_id in existing.get("file_ids", {}).values():
            await fs_bucket.delete(file_id)

    
    model_id = await fs_bucket.upload_from_stream(f"{req.model_name}_model.pt", model_bytes)
    tokenizer_id = await fs_bucket.upload_from_stream(f"{req.model_name}_tokenizer.pkl", tokenizer_bytes)
    sp_to_obj_id = await fs_bucket.upload_from_stream(f"{req.model_name}_sp_to_obj.pkl", sp_to_obj_bytes)

    
    doc = {
        "model_name": req.model_name,
        "file_ids": {
            "model": model_id,
            "tokenizer": tokenizer_id,
            "sp_to_obj": sp_to_obj_id
        },
        "created_at": datetime.datetime.utcnow()
    }

    await llm_collection.replace_one({"model_name": req.model_name}, doc, upsert=True)

    return {"message": f"Model '{req.model_name}' trained and saved successfully in MongoDB"}


@app.post("/llm/load_model/{model_name}")
async def load_model(model_name: str):
    global kg_llm

    doc = await llm_collection.find_one({"model_name": model_name})
    if not doc:
        raise HTTPException(status_code=404, detail="Model not found")

    file_ids = doc["file_ids"]

    
    model_bytes = io.BytesIO()
    await fs_bucket.download_to_stream(file_ids["model"], model_bytes)
    model_bytes.seek(0)

    tokenizer_bytes = io.BytesIO()
    await fs_bucket.download_to_stream(file_ids["tokenizer"], tokenizer_bytes)
    tokenizer_bytes.seek(0)

    sp_to_obj_bytes = io.BytesIO()
    await fs_bucket.download_to_stream(file_ids["sp_to_obj"], sp_to_obj_bytes)
    sp_to_obj_bytes.seek(0)

    
    tokenizer = pickle.load(tokenizer_bytes)
    sp_to_obj = pickle.load(sp_to_obj_bytes)

    model = KGModel(len(tokenizer.sp_vocab), len(tokenizer.o_vocab))
    model.load_state_dict(torch.load(model_bytes))
    model.eval()

    kg_llm = {
        "model": model,
        "tokenizer": tokenizer,
        "sp_to_obj": sp_to_obj
    }

    return {"message": f"Model '{model_name}' loaded successfully from MongoDB"}



class GenerateRequest(BaseModel):
    model_name: str
    subject: str
    predicate: str
    num_samples: int = 3

    shape: Optional[str] = None
    path: Optional[str] = None
    datatype: Optional[str] = None

    distribution_type: Optional[str] = None
    distribution_params: Optional[dict] = None
    encode_categorical: bool = False



@app.post("/custom_llm/generate")
async def generate(req: GenerateRequest):
    global kg_llm

    
    if not kg_llm or kg_llm.get("model_name") != req.model_name:
        
        doc = await llm_collection.find_one({"model_name": req.model_name})
        if not doc:
            raise HTTPException(status_code=404, detail="Model not found")

        file_ids = doc["file_ids"]

        
        model_bytes = io.BytesIO()
        await fs_bucket.download_to_stream(file_ids["model"], model_bytes)
        model_bytes.seek(0)

        tokenizer_bytes = io.BytesIO()
        await fs_bucket.download_to_stream(file_ids["tokenizer"], tokenizer_bytes)
        tokenizer_bytes.seek(0)

        sp_to_obj_bytes = io.BytesIO()
        await fs_bucket.download_to_stream(file_ids["sp_to_obj"], sp_to_obj_bytes)
        sp_to_obj_bytes.seek(0)

        
        tokenizer = pickle.load(tokenizer_bytes)
        sp_to_obj = pickle.load(sp_to_obj_bytes)

        model = KGModel(len(tokenizer.sp_vocab), len(tokenizer.o_vocab))
        model.load_state_dict(torch.load(model_bytes))
        model.eval()

        
        kg_llm = {
            "model_name": req.model_name,
            "model": model,
            "tokenizer": tokenizer,
            "sp_to_obj": sp_to_obj
        }

    model = kg_llm["model"]
    tokenizer = kg_llm["tokenizer"]
    sp_to_obj = kg_llm["sp_to_obj"]

    
    if (req.subject, req.predicate) not in sp_to_obj:
        raise HTTPException(
            status_code=400,
            detail="Invalid subject-predicate combination"
        )

    sp_token = f"{req.subject}|{req.predicate}"
    sp_id = tokenizer.sp_vocab[sp_token]
    valid_objects = sp_to_obj[(req.subject, req.predicate)]

    results = []

    with torch.no_grad():
        for _ in range(req.num_samples):
            sp_tensor = torch.tensor([sp_id])
            o_tensor = torch.tensor([0])

            logits = model(sp_tensor, o_tensor)
            probs = torch.softmax(logits, dim=1)

            mask = torch.zeros_like(probs)
            mask[:, valid_objects] = 1
            probs = probs * mask

            if probs.sum() == 0:
                continue

            probs = probs / probs.sum()
            obj_idx = torch.multinomial(probs, 1).item()
            results.append(tokenizer.o_inverse[obj_idx])

    
    if req.distribution_type and req.distribution_params:
        results = filter_by_distribution(
            results,
            req.distribution_type,
            req.distribution_params,
            encode_categorical=req.encode_categorical
        )

    return {"generated_objects": results[:req.num_samples]}




@app.get("/llm/subject_predicates/{model_name}")
async def list_subject_predicates(model_name: str):

    doc = await llm_collection.find_one({"model_name": model_name})

    if not doc:
        raise HTTPException(status_code=404, detail="Model not found")

    sp_to_obj = pickle.loads(doc["sp_to_obj"])

    subject_predicates = {}

    for (s, p) in sp_to_obj.keys():

        subject_predicates.setdefault(s, []).append(p)

    return {
        "model_name": model_name,
        "subject_predicates": subject_predicates
    }



@app.get("/debug/custom_llm_keys/{model_name}")
async def debug_custom_llm_keys(model_name: str):
    custom_llm_loaded = await load_custom_llm(model_name)
    
    
    keys = list(custom_llm_loaded["sp_to_obj"].keys())
    print("sp_to_obj keys:", keys)
    
    
    return {"sp_to_obj_keys": keys}


@app.get("/custom_llm_models")
async def list_custom_llm_models():
    models = await llm_collection.distinct("model_name")
    return {"saved_models": models}