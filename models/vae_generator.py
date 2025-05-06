# import torch
# import torch.nn as nn
# import torch.optim as optim

# class VAE(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super(VAE, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 256)
#         self.fc21 = nn.Linear(256, latent_dim)  # Mean of latent space
#         self.fc22 = nn.Linear(256, latent_dim)  # Log variance of latent space
#         self.fc3 = nn.Linear(latent_dim, 256)
#         self.fc4 = nn.Linear(256, input_dim)

#     def encode(self, x):
#         h1 = torch.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         h3 = torch.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3))

#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, 28 * 28))  # Flatten if using image data
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar

# # Loss function for VAE
# def loss_function(recon_x, x, mu, logvar):
#     BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 28 * 28), reduction='sum')
#     MSE = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return BCE + MSE

# # Function to generate synthetic data using the trained VAE model
# def generate_synthetic_data(num_samples: int, model, input_dim):
#     z = torch.randn(num_samples, model.fc21.in_features)  # Latent dimension
#     generated_data = model.decode(z)
#     return generated_data.detach().numpy()  # Convert to NumPy array for further processing



# # vae_generator.py

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from typing import List, Dict, Any

# # Define the VAE model
# class VAE(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super(VAE, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 256)
#         self.fc21 = nn.Linear(256, latent_dim)  # Mean of latent space
#         self.fc22 = nn.Linear(256, latent_dim)  # Log variance of latent space
#         self.fc3 = nn.Linear(latent_dim, 256)
#         self.fc4 = nn.Linear(256, input_dim)

#     def encode(self, x):
#         h1 = torch.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def decode(self, z):
#         h3 = torch.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3))

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar

# # Loss function
# def loss_function(recon_x, x, mu, logvar):
#     BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 28*28), reduction='sum')
#     # KL divergence
#     MSE = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return BCE + MSE

# # Initialize VAE model and optimizer
# latent_dim = 20
# input_dim = 28*28  # For MNIST-like data, change for your case
# model_vae = VAE(input_dim, latent_dim)
# optimizer_vae = optim.Adam(model_vae.parameters(), lr=1e-3)

# # Generate synthetic data with VAE
# def generate_synthetic_data_vae(num_samples: int, constraints: List[Dict[str, str]]) -> List[Dict[str, Any]]:
#     model_vae.eval()  # Set to evaluation mode
#     synthetic_data = []

#     # Generate synthetic data using the VAE model
#     for _ in range(num_samples):
#         # Sample from latent space
#         z = torch.randn(1, latent_dim)
#         generated_data = model_vae.decode(z)
#         generated_data = generated_data.view(-1, input_dim).detach().numpy().tolist()  # Convert to list

#         sample = {
#             "generated_data": generated_data
#         }
#         synthetic_data.append(sample)
    
#     return synthetic_data




import os
import torch
import torch.nn as nn
import torch.optim as optim
import rdflib
import pandas as pd
import numpy as np
import pickle

class VAE(nn.Module):
    def __init__(self, subject_dim, predicate_dim, object_dim, latent_dim=64):
        super(VAE, self).__init__()
        self.subject_dim = subject_dim
        self.predicate_dim = predicate_dim
        self.object_dim = object_dim
        self.latent_dim = latent_dim

        input_dim = subject_dim + predicate_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim + input_dim, 128)
        self.fc3 = nn.Linear(128, object_dim)

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
        return torch.tanh(self.fc3(h2))

    def forward(self, x_cond):
        mu, logvar = self.encode(x_cond)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, x_cond)
        return recon_x, mu, logvar

def vae_loss(recon_x, x_target, mu, logvar):
    BCE = nn.functional.binary_cross_entropy_with_logits(recon_x, x_target, reduction='sum')
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD



def load_dbpedia_ttl(file_path):
    g = rdflib.Graph()
    g.parse(file_path, format='ttl')
    return [(str(s), str(p), str(o)) for s, p, o in g]


# ------ VAE Save/Load Functions ------

# def save_vae_model(model_name, model, optimizer):
#     os.makedirs(f"models/saved_models/{model_name}", exist_ok=True)
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict()
#     }, f"models/saved_models/{model_name}/vae.pth")
#     print(f"✅ VAE model '{model_name}' saved successfully.")

# def load_vae_model(model_name, ttl_path, latent_dim=64):
#     factorized_data, model, optimizer = factorize_and_initialize_vae(ttl_path, latent_dim=latent_dim)

#     checkpoint = torch.load(f"models/saved_models/{model_name}/vae.pth")
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#     model.eval()

#     loaded_models[model_name] = {
#         "vae_model": model,
#         "factorized_data": factorized_data,
#         "optimizer": optimizer
#     }

#     print(f"✅ VAE model '{model_name}' loaded successfully.")


# def save_vae_model(model_name, model, optimizer, ttl_path):
#     os.makedirs(f"models/saved_models/{model_name}", exist_ok=True)
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict()
#     }, f"models/saved_models/{model_name}/vae.pth")
    
#     # Optionally save the TTL path or any other metadata related to the dataset
#     with open(f"models/saved_models/{model_name}/metadata.txt", "w") as f:
#         f.write(f"TTL file used for training: {ttl_path}")
    
#     print(f"✅ VAE model '{model_name}' saved successfully with TTL path '{ttl_path}'.")


# def load_vae_model(model_name, ttl_path, latent_dim=64):
#     # Set the TTL path and initialize the model
#     factorized_data, model, optimizer = factorize_and_initialize_vae(ttl_path, latent_dim=latent_dim)
    
#     # Load the VAE model's state
#     checkpoint = torch.load(f"models/saved_models/{model_name}/vae.pth")
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#     model.eval()

#     loaded_models[model_name] = {
#         "vae_model": model,
#         "factorized_data": factorized_data,
#         "optimizer": optimizer,
#         "ttl_path": ttl_path  # Store the TTL path for reference
#     }

#     print(f"✅ VAE model '{model_name}' loaded successfully with TTL path '{ttl_path}'.")



# --------- Initialization Functions ---------
def factorize_and_initialize_vae(file_path, latent_dim=64):
    triples = load_dbpedia_ttl(file_path)
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

    model = VAE(subject_dim, predicate_dim, object_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return factorized_data, model, optimizer

# Define the global dictionary to store loaded models
loaded_models = {}

# ------ VAE Save/Load Functions ------

# def save_vae_model(model_name, model, optimizer, ttl_path):
#     os.makedirs(f"models/saved_models/{model_name}", exist_ok=True)
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict()
#     }, f"models/saved_models/{model_name}/vae.pth")
#     print(f"✅ VAE model '{model_name}' saved successfully.")

# def load_vae_model(model_name, ttl_path, latent_dim=64):
#     # Factorize and initialize the VAE model
#     factorized_data, model, optimizer = factorize_and_initialize_vae(ttl_path, latent_dim=latent_dim)

#     # Load the model state dict
#     checkpoint = torch.load(f"models/saved_models/{model_name}/vae.pth")
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#     model.eval()

#     # Store the model in loaded_models
#     loaded_models[model_name] = {
#         "vae_model": model,
#         "factorized_data": factorized_data,
#         "optimizer": optimizer
#     }

#     print(f"✅ VAE model '{model_name}' loaded successfully.")



# Path to store models persistently (using the mounted directory in Docker)
MODEL_DIR = "/app/uploaded/vae"  # This points to the uploaded directory inside the container

# Save the trained model to the persistent location
def save_vae_model(model_name, model, optimizer, ttl_path):
    os.makedirs(f"{MODEL_DIR}/{model_name}", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"/app/models/saved_models/vae/{model_name}/vae.pth")
    print(f"✅ VAE model '{model_name}' saved successfully.")

# # Load the trained model from the persistent location
# def load_vae_model(model_name, ttl_path, latent_dim=64):
#     factorized_data, model, optimizer = factorize_and_initialize_vae(ttl_path, latent_dim=latent_dim)

#     # Check if the model file exists
#     model_path = f"{MODEL_DIR}/{model_name}/vae.pth"
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file '{model_name}' not found at {model_path}")

#     checkpoint = torch.load(model_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#     model.eval()

#     loaded_models[model_name] = {
#         "vae_model": model,
#         "factorized_data": factorized_data,
#         "optimizer": optimizer
#     }

#     print(f"✅ VAE model '{model_name}' loaded successfully.")


def load_vae_model(model_name, ttl_path):
    model_path = f"/app/models/saved_models/vae/{model_name}/vae.pth"
    factorized_data_path = f"/app/models/saved_models/vae/{model_name}/{model_name}_factorized_data.pkl"

    # Load the factorized data
    with open(factorized_data_path, "rb") as f:
        factorized_data = pickle.load(f)

    # Initialize the model (you should use the appropriate class here)
    vae_model = VAE(factorized_data["subject_dim"], factorized_data["predicate_dim"], factorized_data["object_dim"], latent_dim=64)
    vae_model.load_state_dict(torch.load(model_path))
    
    # Initialize optimizer (assuming Adam, you can customize this)
    optimizer = optim.Adam(vae_model.parameters(), lr=0.001)
    
    return factorized_data, vae_model, optimizer



# --------- Training Function ---------
def train_vae(model, optimizer, factorized_data, num_epochs=100, batch_size=64):
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
        total_loss = 0
        for i in range(0, total_samples, dynamic_batch_size):
            s = torch.tensor(subjects[i:i+dynamic_batch_size], dtype=torch.long)
            p = torch.tensor(predicates[i:i+dynamic_batch_size], dtype=torch.long)
            o = torch.tensor(objects[i:i+dynamic_batch_size], dtype=torch.long)

            s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
            p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()
            o_oh = torch.nn.functional.one_hot(o, num_classes=object_dim).float()

            x_cond = torch.cat((s_oh, p_oh), dim=1)
            x_target = o_oh

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x_cond)
            loss = vae_loss(recon_x, x_target, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Loss: {total_loss/total_samples}")

#--------- Generation Function ---------
# def generate_data_vae_model(model, factorized_data, subject_input, predicate_input, num_samples=1):
#     subject_dim = factorized_data["subject_dim"]
#     predicate_dim = factorized_data["predicate_dim"]

#     subject_input_lower = subject_input.lower()
#     predicate_input_lower = predicate_input.lower()

#     subject_matches = [s for s in factorized_data["subject_uniques"] if subject_input_lower in s.lower()]
#     predicate_matches = [p for p in factorized_data["predicate_uniques"] if predicate_input_lower in p.lower()]

#     if len(subject_matches) == 0:
#         raise ValueError(f"Subject '{subject_input}' not found.")
#     if len(predicate_matches) == 0:
#         raise ValueError(f"Predicate '{predicate_input}' not found.")

#     subject_input = subject_matches[0]
#     predicate_input = predicate_matches[0]

#     subject_idx = np.where(factorized_data["subject_uniques"] == subject_input)[0][0]
#     predicate_idx = np.where(factorized_data["predicate_uniques"] == predicate_input)[0][0]

#     s = torch.tensor([subject_idx], dtype=torch.long)
#     p = torch.tensor([predicate_idx], dtype=torch.long)

#     s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
#     p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()

#     x_cond = torch.cat((s_oh, p_oh), dim=1).repeat(num_samples, 1)
#     with torch.no_grad():
#         mu, logvar = model.encode(x_cond)
#         z = model.reparameterize(mu, logvar)
#         generated = model.decode(z, x_cond)
    
#     generated_idx = torch.argmax(generated, dim=1).numpy()
#     decoded_objects = [factorized_data["object_inverse_map"].get(idx, "UNKNOWN") for idx in generated_idx]
#     return decoded_objects

def generate_data_vae_model(model, factorized_data, subject_input, predicate_input, num_samples=1):
    subject_dim = factorized_data["subject_dim"]
    predicate_dim = factorized_data["predicate_dim"]

    subject_input_lower = subject_input.lower()
    predicate_input_lower = predicate_input.lower()

    # Find matches for the subject and predicate
    subject_matches = [s for s in factorized_data["subject_uniques"] if subject_input_lower in s.lower()]
    predicate_matches = [p for p in factorized_data["predicate_uniques"] if predicate_input_lower in p.lower()]

    # If no match for the subject
    if len(subject_matches) == 0:
        print(f"Debug: Available subjects: {factorized_data['subject_uniques']}")
        raise ValueError(f"Subject '{subject_input}' not found. Available subjects: {', '.join(factorized_data['subject_uniques'])}")

    # If no match for the predicate
    if len(predicate_matches) == 0:
        print(f"Debug: Available predicates: {factorized_data['predicate_uniques']}")
        raise ValueError(f"Predicate '{predicate_input}' not found. Available predicates: {', '.join(factorized_data['predicate_uniques'])}")

    # Use the first matching subject and predicate
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
    
    generated_idx = torch.argmax(generated, dim=1).numpy()
    decoded_objects = [factorized_data["object_inverse_map"].get(idx, "UNKNOWN") for idx in generated_idx]
    
    return decoded_objects

def load_and_generate_vae_data(model_name, subject, predicate, num_samples, distribution="normal"):
    model_path = f"/app/models/saved_models/vae/{model_name}/vae.pth"
    factorized_data_path = f"/app/models/saved_models/vae/{model_name}/{model_name}_factorized_data.pkl"
    
    # Load the factorized data
    with open(factorized_data_path, "rb") as f:
        factorized_data = pickle.load(f)

    # Initialize the model (you should use the appropriate class here)
    vae_model = VAE(factorized_data["subject_dim"], factorized_data["predicate_dim"], factorized_data["object_dim"], latent_dim=64)
    vae_model.load_state_dict(torch.load(model_path))
    
    # Initialize optimizer (assuming Adam, you can customize this)
    optimizer = optim.Adam(vae_model.parameters(), lr=0.001)
    
    # Store the model in loaded_models dictionary after loading
    loaded_models[model_name] = {
        "vae_model": vae_model,
        "factorized_data": factorized_data,
        "optimizer": optimizer
    }

    # Generate data using the loaded VAE model
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