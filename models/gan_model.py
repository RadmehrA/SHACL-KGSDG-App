import os
import torch
import torch.nn as nn
import torch.optim as optim
import rdflib
import numpy as np
import pandas as pd


factorized_data = {}
generator = None
discriminator = None
optimizer_g = None
optimizer_d = None
criterion = nn.BCELoss()
z_dim = 100

def load_rdf_graph(file_path):
    g = rdflib.Graph()
    g.parse(file_path, format='ttl')
    return [(str(s), str(p), str(o)) for s, p, o in g]

<<<<<<< HEAD
=======

def load_rdf_graph(file_path):
    g = rdflib.Graph()
    if file_path.endswith(".ttl"):
        g.parse(file_path, format="ttl")
    elif file_path.endswith(".owl") or file_path.endswith(".rdf") or file_path.endswith(".xml"):
        g.parse(file_path, format="xml")
    else:
        raise ValueError(f"Unsupported file format for: {file_path}")
    return [(str(s), str(p), str(o)) for s, p, o in g]

>>>>>>> 00bb4cf (Second_edition)
def factorize_and_initialize_gan(file_path):
    triples = load_rdf_graph(file_path)
    df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])

    subjects = pd.factorize(df['subject'])[0]
    predicates = pd.factorize(df['predicate'])[0]
    objects = pd.factorize(df['object'])[0]

    subject_dim = len(np.unique(subjects))
    predicate_dim = len(np.unique(predicates))
    object_dim = len(np.unique(objects))

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
        "object_inverse_map": dict(enumerate(df["object"].unique()))
    }

    generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
    discriminator = Discriminator(subject_dim, predicate_dim, object_dim)
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    return factorized_data, generator, discriminator, optimizer_g, optimizer_d



def factorize_and_initialize_gans(file_path):
    global factorized_data, generator, discriminator, optimizer_g, optimizer_d

    triples = load_rdf_graph(file_path)
    df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])

    subjects = pd.factorize(df['subject'])[0]
    predicates = pd.factorize(df['predicate'])[0]
    objects = pd.factorize(df['object'])[0]

    subject_dim = len(np.unique(subjects))
    predicate_dim = len(np.unique(predicates))
    object_dim = len(np.unique(objects))

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
        "object_inverse_map": dict(enumerate(df["object"].unique()))
    }

    generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
    discriminator = Discriminator(subject_dim, predicate_dim, object_dim)
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    return factorized_data, generator, discriminator, optimizer_g, optimizer_d



class Generator(nn.Module):
    def __init__(self, subject_dim, predicate_dim, object_dim, z_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + subject_dim + predicate_dim, 128),
            nn.ReLU(),
            nn.Linear(128, object_dim),
            nn.Tanh()
        )

    def forward(self, z, subject, predicate):
        return self.fc(torch.cat((z, subject, predicate), dim=1))

class Discriminator(nn.Module):
    def __init__(self, subject_dim, predicate_dim, object_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(subject_dim + predicate_dim + object_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, subject, predicate, object):
        return self.fc(torch.cat((subject, predicate, object), dim=1))

def sample_noise(batch_size, z_dim, distribution="normal", dist_params=None):
    """
    dist_params is a dict with parameters depending on distribution type.
    Example:
      - normal: {'mean': 0, 'std': 1}
      - uniform: {'low': -1, 'high': 1}
      - skewed: {'skew': 3}  # example param
      - categorical: {'probs': [0.1, 0.2, ..., 0.05]}  (length = z_dim)
    """
    if distribution == "uniform":
        low = dist_params.get("low", -1) if dist_params else -1
        high = dist_params.get("high", 1) if dist_params else 1
        return torch.rand(batch_size, z_dim) * (high - low) + low
    elif distribution == "skewed":
        skew = dist_params.get("skew", 3) if dist_params else 3
        base = torch.randn(batch_size, z_dim)
        return base ** skew
    elif distribution == "categorical":
        probs = dist_params.get("probs") if dist_params else None
        if probs is None:
            probs = torch.ones(z_dim) / z_dim  # uniform categorical by default
        else:
            probs = torch.tensor(probs)
        categorical_samples = torch.multinomial(probs, batch_size, replacement=True)
        return torch.nn.functional.one_hot(categorical_samples, num_classes=z_dim).float()
    else:  # default normal
        mean = dist_params.get("mean", 0) if dist_params else 0
        std = dist_params.get("std", 1) if dist_params else 1
        return torch.randn(batch_size, z_dim) * std + mean


def train_gan(num_epochs=1000, batch_size=64, distribution="normal",dist_params=None):
    df = factorized_data["df"]
    subjects = factorized_data["subjects"]
    predicates = factorized_data["predicates"]
    objects = factorized_data["objects"]
    subject_dim = factorized_data["subject_dim"]
    predicate_dim = factorized_data["predicate_dim"]
    object_dim = factorized_data["object_dim"]

    total_samples = len(df)
    dynamic_batch_size = min(batch_size, total_samples)

    for epoch in range(num_epochs):
        for i in range(0, total_samples, dynamic_batch_size):
            s = torch.tensor(subjects[i:i+dynamic_batch_size], dtype=torch.long)
            p = torch.tensor(predicates[i:i+dynamic_batch_size], dtype=torch.long)
            o = torch.tensor(objects[i:i+dynamic_batch_size], dtype=torch.long)

            s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
            p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()
            o_oh = torch.nn.functional.one_hot(o, num_classes=object_dim).float()

            current_batch_size = s_oh.size(0)
            real_labels = torch.ones(current_batch_size, 1)
            fake_labels = torch.zeros(current_batch_size, 1)

            # Train discriminator
            optimizer_d.zero_grad()
            real_preds = discriminator(s_oh, p_oh, o_oh)
            d_loss_real = criterion(real_preds, real_labels)

            #z = sample_noise(current_batch_size, z_dim, distribution)
            z = sample_noise(current_batch_size, z_dim, distribution, dist_params)
            fake_objects = generator(z, s_oh, p_oh)
            fake_preds = discriminator(s_oh, p_oh, fake_objects)
            d_loss_fake = criterion(fake_preds, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # Train generator
            optimizer_g.zero_grad()
            z = sample_noise(current_batch_size, z_dim, distribution)
            fake_objects = generator(z, s_oh, p_oh)
            fake_preds = discriminator(s_oh, p_oh, fake_objects)
            g_loss = criterion(fake_preds, real_labels)
            g_loss.backward()
            optimizer_g.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

def save_model(model_name):
    os.makedirs(f"models/saved_models/gan/{model_name}", exist_ok=True)
    torch.save(generator.state_dict(), f"models/saved_models/gan/{model_name}/generator.pth")
    torch.save(discriminator.state_dict(), f"models/saved_models/gan/{model_name}/discriminator.pth")

loaded_models = {}

def load_model(model_name):
    ttl_path = f"uploaded/{model_name}.ttl"
    factorized_data, generator, discriminator, optimizer_g, optimizer_d = factorize_and_initialize_gan(ttl_path)

    generator.load_state_dict(torch.load(f"models/saved_models/gan/{model_name}/generator.pth"))
    discriminator.load_state_dict(torch.load(f"models/saved_models/gan/{model_name}/discriminator.pth"))

    generator.eval()
    discriminator.eval()

    loaded_models[model_name] = {
        "generator": generator,
        "discriminator": discriminator,
        "factorized_data": factorized_data,
        "optimizer_g": optimizer_g,
        "optimizer_d": optimizer_d
    }

    print(f"✅ Model '{model_name}' loaded successfully.")

<<<<<<< HEAD
def generate_synthetic_data(model_name, subject_input, predicate_input, num_samples=1, distribution="normal"):
=======
def generate_synthetic_data(model_name, subject_input, predicate_input, num_samples=1, distribution="normal",dist_params=None):
>>>>>>> 00bb4cf (Second_edition)
    # If the model_name is "all", loop through all loaded models
    if model_name == "all":
        generated_objects = []
        for model_name, model in loaded_models.items():
            try:
                subject_dim = model["factorized_data"]["subject_dim"]
                predicate_dim = model["factorized_data"]["predicate_dim"]
                generator = model["generator"]
                factorized_data = model["factorized_data"]

                subject_input_lower = subject_input.lower()
                predicate_input_lower = predicate_input.lower()

                subject_matches = [s for s in factorized_data["subject_uniques"] if subject_input_lower in s.lower()]
                predicate_matches = [p for p in factorized_data["predicate_uniques"] if predicate_input_lower in p.lower()]

                if len(subject_matches) == 0 or len(predicate_matches) == 0:
                    continue  # Skip this model if no matches found

                subject_input = subject_matches[0]
                predicate_input = predicate_matches[0]

                subject_idx = np.where(factorized_data["subject_uniques"] == subject_input)[0][0]
                predicate_idx = np.where(factorized_data["predicate_uniques"] == predicate_input)[0][0]

                s = torch.tensor([subject_idx], dtype=torch.long)
                p = torch.tensor([predicate_idx], dtype=torch.long)
                s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
                p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()

                #z = sample_noise(num_samples, z_dim, distribution)
                z = sample_noise(num_samples, z_dim, distribution, dist_params)
                generated = generator(z, s_oh.repeat(num_samples, 1), p_oh.repeat(num_samples, 1)).detach().numpy()
                generated_idx = np.argmax(generated, axis=1)
                decoded_objects = [factorized_data["object_inverse_map"].get(idx, "UNKNOWN") for idx in generated_idx]
                generated_objects.extend(decoded_objects)
            except Exception as e:
                # Log error or handle exception for this particular model if needed
                continue
        return generated_objects

    # If the model_name is specific, proceed as usual
    if model_name not in loaded_models:
        raise RuntimeError(f"Model '{model_name}' is not loaded.")
    
    model = loaded_models[model_name]
    generator = model["generator"]
    factorized_data = model["factorized_data"]

    if generator is None:
        raise RuntimeError("Generator model is not loaded.")
    
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

    z = sample_noise(num_samples, z_dim, distribution)
    generated = generator(z, s_oh.repeat(num_samples, 1), p_oh.repeat(num_samples, 1)).detach().numpy()
    generated_idx = np.argmax(generated, axis=1)
    decoded_objects = [factorized_data["object_inverse_map"].get(idx, "UNKNOWN") for idx in generated_idx]
    return decoded_objects

