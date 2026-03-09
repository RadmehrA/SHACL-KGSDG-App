def find_uploaded_file(model_name, directory):
    for ext in [".ttl", ".owl", ".rdf", ".xml"]:
        path = os.path.join(directory, f"{model_name}{ext}")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Uploaded file for model '{model_name}' not found.")

async def get_vae_model(model_name: str):
    if model_name in loaded_models:
        return loaded_models[model_name]
    return await load_vae_from_mongo(model_name)


import rdflib
from rdflib.namespace import RDF, RDFS, OWL
import pandas as pd
import numpy as np

def load_rdf_graph(file_path):
    """
    Load an RDF/OWL graph and flatten OWL restrictions into explicit triples
    like (subject, predicate, object), so predicates like 'hasTopping' are included.
    """
    g = rdflib.Graph()
    g.parse(file_path)

    triples = []

    for s in g.subjects(RDF.type, OWL.Class):
        for o in g.objects(s, RDFS.subClassOf):
            
            if (o, RDF.type, OWL.Restriction) in g:
                
                prop = next(g.objects(o, OWL.onProperty))
                
                values = list(g.objects(o, OWL.someValuesFrom)) + list(g.objects(o, OWL.hasValue))
                for v in values:
                    triples.append((s, prop, v))
            else:
               
                triples.append((s, RDFS.subClassOf, o))

    str_triples = [(str(s), str(p), str(o)) for s, p, o in triples]
    return str_triples


def factorize_and_initialize_vae(file_path, latent_dim=64):
    """
    Factorize subjects, predicates, and objects for any RDF/OWL ontology.
    Works with any predicates like hasTopping, hasIngredient, etc.
    """
    triples = load_rdf_graph(file_path)
    df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])

    subjects = pd.factorize(df['subject'])[0]
    predicates = pd.factorize(df['predicate'])[0]
    objects = pd.factorize(df['object'])[0]

    subject_dim = len(np.unique(subjects))
    predicate_dim = len(np.unique(predicates))
    object_dim = len(np.unique(objects))

    predicate_map = {p.split("#")[-1]: p for p in df["predicate"].unique()}
    subject_map = {s.split("#")[-1]: s for s in df["subject"].unique()}

    factorized_data = {
        "df": df,
        "subjects": subjects,
        "predicates": predicates,
        "objects": objects,
        "subject_dim": subject_dim,
        "predicate_dim": predicate_dim,
        "object_dim": object_dim,
        "subject_uniques": df["subject"].unique(),
        "predicate_uniques": df["predicate"].unique(),
        "object_inverse_map": dict(enumerate(df["object"].unique())),
        "predicate_map": predicate_map,
        "subject_map": subject_map
    }


    model = VAE(subject_dim, predicate_dim, object_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return factorized_data, model, optimizer

loaded_models = {}



MODEL_DIR = "/app/uploaded/vae"


def save_vae_model(model_name, model, optimizer, ttl_path):
    os.makedirs(f"{MODEL_DIR}/{model_name}", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"/app/models/saved_models/vae/{model_name}/vae.pth")
    print(f"✅ VAE model '{model_name}' saved successfully.")

def load_vae_model(model_name):
    model_path = f"/app/models/saved_models/vae/{model_name}/vae.pth"
    factorized_data_path = f"/app/models/saved_models/vae/{model_name}/{model_name}_factorized_data.pkl"

    with open(factorized_data_path, "rb") as f:
        factorized_data = pickle.load(f)

    vae_model = VAE(factorized_data["subject_dim"], factorized_data["predicate_dim"], factorized_data["object_dim"], latent_dim=64)
    vae_model.load_state_dict(torch.load(model_path))
    
    optimizer = optim.Adam(vae_model.parameters(), lr=0.001)
    
    return factorized_data, vae_model, optimizer


def generate_data_vae_model(model, factorized_data, subject_input, predicate_input, num_samples=1):
    """
    Generate objects using a VAE model. Automatically maps local names to full URIs.
    """
    subject_dim = factorized_data["subject_dim"]
    predicate_dim = factorized_data["predicate_dim"]

    if subject_input in factorized_data.get("subject_map", {}):
        subject_input = factorized_data["subject_map"][subject_input]
    if predicate_input in factorized_data.get("predicate_map", {}):
        predicate_input = factorized_data["predicate_map"][predicate_input]

    subject_matches = [s for s in factorized_data["subject_uniques"] if subject_input.lower() in s.lower()]
    predicate_matches = [p for p in factorized_data["predicate_uniques"] if predicate_input.lower() in p.lower()]

    if not subject_matches:
        raise ValueError(f"Subject '{subject_input}' not found in model.")
    if not predicate_matches:
        raise ValueError(f"Predicate '{predicate_input}' not found in model.")

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
        mu, logvar = model.encode(x_cond)
        z = model.reparameterize(mu, logvar)
        generated = model.decode(z, x_cond)

    if model.numeric:
        return generated.squeeze().numpy().tolist()

    generated_idx = torch.argmax(generated, dim=1).numpy()
    return [factorized_data["object_inverse_map"].get(idx, "UNKNOWN") for idx in generated_idx]

def load_and_generate_vae_data(model_name, subject, predicate, num_samples, distribution="normal"):
    model_path = f"/app/models/saved_models/vae/{model_name}/vae.pth"
    factorized_data_path = f"/app/models/saved_models/vae/{model_name}/{model_name}_factorized_data.pkl"
    
    with open(factorized_data_path, "rb") as f:
        factorized_data = pickle.load(f)

    vae_model = VAE(factorized_data["subject_dim"], factorized_data["predicate_dim"], factorized_data["object_dim"], latent_dim=64)
    vae_model.load_state_dict(torch.load(model_path))
    
    optimizer = optim.Adam(vae_model.parameters(), lr=0.001)
    
    loaded_models[model_name] = {
        "vae_model": vae_model,
        "factorized_data": factorized_data,
        "optimizer": optimizer
    }

    subject_dim = factorized_data["subject_dim"]
    predicate_dim = factorized_data["predicate_dim"]

    subject_input_lower = subject.lower()
    predicate_input_lower = predicate.lower()

    subject_matches = [s for s in factorized_data["subject_uniques"] if subject_input_lower in s.lower()]
    predicate_matches = [p for p in factorized_data["predicate_uniques"] if predicate_input_lower in p.lower()]

    if len(subject_matches) == 0:
        raise ValueError(f"Subject '{subject}' not found.")
    if len(predicate_matches) == 0:
        raise ValueError(f"Predicate '{predicate}' not found.")

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
        mu, logvar = vae_model.encode(x_cond)
        z = vae_model.reparameterize(mu, logvar)
        generated = vae_model.decode(z, x_cond)
    
    generated_idx = torch.argmax(generated, dim=1).numpy()
    decoded_objects = [factorized_data["object_inverse_map"].get(idx, "UNKNOWN") for idx in generated_idx]
    
    return {
        "vae_model": vae_model,
        "factorized_data": factorized_data,
        "optimizer": optimizer,
        "generated_objects": decoded_objects
    }


import os
import io
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import rdflib
import pandas as pd
import numpy as np
from db.mongo import vae_collection
from typing import List, Tuple

class VAE(nn.Module):
    def __init__(self, subject_dim, predicate_dim, object_dim=None, latent_dim=64, numeric=False):
        super(VAE, self).__init__()
        self.subject_dim = subject_dim
        self.predicate_dim = predicate_dim
        self.object_dim = object_dim
        self.latent_dim = latent_dim
        self.numeric = numeric

        input_dim = subject_dim + predicate_dim


        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)


        if numeric:
            output_dim = 1
        else:
            output_dim = object_dim
        self.fc2 = nn.Linear(latent_dim + input_dim, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_cond):
        input = torch.cat([z, x_cond], dim=1)
        h2 = torch.relu(self.fc2(input))
        if self.numeric:
            return self.fc3(h2)
        return torch.tanh(self.fc3(h2))

    def forward(self, x_cond):
        mu, logvar = self.encode(x_cond)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, x_cond)
        return recon_x, mu, logvar

def vae_loss(recon_x, x_target, mu, logvar, numeric=False):
    if numeric:
        BCE = nn.functional.mse_loss(recon_x, x_target, reduction='sum')
    else:
        BCE = nn.functional.binary_cross_entropy_with_logits(recon_x, x_target, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def split_triples_by_type(triples):
    categorical_triples = []
    numeric_triples = []
    for s, p, o in triples:
        try:
            float(o)
            numeric_triples.append((s, p, float(o)))
        except ValueError:
            categorical_triples.append((s, p, o))
    return categorical_triples, numeric_triples

def factorize_triples(triples, numeric=False):
    df = pd.DataFrame(triples, columns=["subject","predicate","object"])
    subjects = pd.factorize(df['subject'])[0]
    predicates = pd.factorize(df['predicate'])[0]

    if numeric:
        objects = df['object'].astype(float).values
        object_dim = None
        object_map = None
    else:
        objects = pd.factorize(df['object'])[0]
        object_dim = len(df['object'].unique())
        object_map = dict(enumerate(df['object'].unique()))

    factorized_data = {
        "df": df,
        "subjects": subjects,
        "predicates": predicates,
        "objects": objects,
        "subject_dim": len(df['subject'].unique()),
        "predicate_dim": len(df['predicate'].unique()),
        "object_dim": object_dim,
        "subject_uniques": df["subject"].unique(),
        "predicate_uniques": df["predicate"].unique(),
        "object_inverse_map": object_map
    }
    return factorized_data


def train_vae(model, optimizer, factorized_data, num_epochs=100, batch_size=64):
    df = factorized_data["df"]
    subjects = factorized_data["subjects"]
    predicates = factorized_data["predicates"]
    objects = factorized_data["objects"]
    subject_dim = factorized_data["subject_dim"]
    predicate_dim = factorized_data["predicate_dim"]
    numeric = model.numeric

    total_samples = len(df)
    batch_size = min(batch_size, total_samples)

    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, total_samples, batch_size):
            s = torch.tensor(subjects[i:i+batch_size], dtype=torch.long)
            p = torch.tensor(predicates[i:i+batch_size], dtype=torch.long)
            o = torch.tensor(objects[i:i+batch_size], dtype=torch.float if numeric else torch.long)

            s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
            p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()
            x_cond = torch.cat([s_oh, p_oh], dim=1)
            x_target = o if numeric else torch.nn.functional.one_hot(o, num_classes=model.object_dim).float()

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x_cond)
            loss = vae_loss(recon_x, x_target, mu, logvar, numeric)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Loss: {total_loss/total_samples:.4f}")

# -----------------------------
# Generate data from VAE
# -----------------------------
def generate_from_vae(model, factorized_data, subject, predicate, num_samples=1):
    subject_dim = factorized_data["subject_dim"]
    predicate_dim = factorized_data["predicate_dim"]
    numeric = model.numeric

    # Match input subject/predicate
    subject_matches = [s for s in factorized_data["subject_uniques"] if subject.lower() in s.lower()]
    predicate_matches = [p for p in factorized_data["predicate_uniques"] if predicate.lower() in p.lower()]

    if len(subject_matches) == 0:
        raise ValueError(f"Subject '{subject}' not found.")
    if len(predicate_matches) == 0:
        raise ValueError(f"Predicate '{predicate}' not found.")

    s_idx = np.where(factorized_data["subject_uniques"] == subject_matches[0])[0][0]
    p_idx = np.where(factorized_data["predicate_uniques"] == predicate_matches[0])[0][0]

    s = torch.tensor([s_idx], dtype=torch.long)
    p = torch.tensor([p_idx], dtype=torch.long)

    s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
    p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()
    x_cond = torch.cat([s_oh, p_oh], dim=1).repeat(num_samples, 1)

    with torch.no_grad():
        mu, logvar = model.encode(x_cond)
        z = model.reparameterize(mu, logvar)
        generated = model.decode(z, x_cond)

    if numeric:
        return generated.squeeze().numpy().tolist()
    generated_idx = torch.argmax(generated, dim=1).numpy()
    return [factorized_data["object_inverse_map"].get(idx, "UNKNOWN") for idx in generated_idx]


async def save_vae_to_mongo(model_name, vae_model, optimizer, factorized_data):
    model_bytes = io.BytesIO()
    torch.save(vae_model.state_dict(), model_bytes)
    model_bytes.seek(0)

    factorized_bytes = pickle.dumps(factorized_data)

    doc = {
        "model_name": model_name,
        "vae_model": model_bytes.read(),
        "factorized_data": factorized_bytes,
        "timestamp": int(torch.randint(1, 1_000_000_000, (1,)).item())
    }

    await vae_collection.update_one({"model_name": model_name}, {"$set": doc}, upsert=True)
    print(f"✅ VAE model '{model_name}' saved to MongoDB.")

async def load_vae_from_mongo(model_name):
    doc = await vae_collection.find_one({"model_name": model_name})
    if doc is None:
        raise ValueError(f"VAE model '{model_name}' not found in MongoDB.")

    factorized_data = pickle.loads(doc["factorized_data"])
    numeric = factorized_data["object_dim"] is None
    vae_model = VAE(factorized_data["subject_dim"], factorized_data["predicate_dim"], factorized_data["object_dim"], numeric=numeric)
    vae_model.load_state_dict(torch.load(io.BytesIO(doc["vae_model"])))

    optimizer = optim.Adam(vae_model.parameters(), lr=0.001)
    return {"vae_model": vae_model, "factorized_data": factorized_data, "optimizer": optimizer}


async def upload_train_save_vae(file_path, model_name, latent_dim=64, num_epochs=50):
    triples = load_rdf_graph(file_path)
    cat_triples, num_triples = split_triples_by_type(triples)

    loaded_models = {}

    vae_models = []


    if cat_triples:
        cat_data = factorize_triples(cat_triples, numeric=False)
        cat_model = VAE(cat_data["subject_dim"], cat_data["predicate_dim"], cat_data["object_dim"], latent_dim, numeric=False)
        optimizer = optim.Adam(cat_model.parameters(), lr=0.001)
        train_vae(cat_model, optimizer, cat_data, num_epochs=num_epochs)
        await save_vae_to_mongo(f"{model_name}_categorical", cat_model, optimizer, cat_data)
        loaded_models[f"{model_name}_categorical"] = {"vae_model": cat_model, "factorized_data": cat_data, "optimizer": optimizer}
        vae_models.append(f"{model_name}_categorical")


    if num_triples:
        num_data = factorize_triples(num_triples, numeric=True)
        num_model = VAE(num_data["subject_dim"], num_data["predicate_dim"], object_dim=None, latent_dim=latent_dim, numeric=True)
        optimizer = optim.Adam(num_model.parameters(), lr=0.001)
        train_vae(num_model, optimizer, num_data, num_epochs=num_epochs)
        await save_vae_to_mongo(f"{model_name}_numeric", num_model, optimizer, num_data)
        loaded_models[f"{model_name}_numeric"] = {"vae_model": num_model, "factorized_data": num_data, "optimizer": optimizer}
        vae_models.append(f"{model_name}_numeric")

    return loaded_models, vae_models


async def generate_data(model_name, subject, predicate, num_samples=1):

    try:
        cat_model = await load_vae_from_mongo(f"{model_name}_categorical")
        if predicate in cat_model["factorized_data"]["predicate_uniques"]:
            return generate_from_vae(cat_model["vae_model"], cat_model["factorized_data"], subject, predicate, num_samples)
    except:
        pass

    try:
        num_model = await load_vae_from_mongo(f"{model_name}_numeric")
        if predicate in num_model["factorized_data"]["predicate_uniques"]:
            return generate_from_vae(num_model["vae_model"], num_model["factorized_data"], subject, predicate, num_samples)
    except:
        pass

    raise ValueError(f"Predicate '{predicate}' not found in model '{model_name}'.")
