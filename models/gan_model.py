
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import rdflib
# import numpy as np
# import pandas as pd
# from db.mongo import gan_collection
# import io
# import pickle
# import time
# from rdflib.namespace import RDF, RDFS, OWL


# factorized_data = {}
# generator = None
# discriminator = None
# optimizer_g = None
# optimizer_d = None
# criterion = nn.BCELoss()
# z_dim = 100

# # def load_rdf_graph(file_path):
# #     g = rdflib.Graph()
# #     g.parse(file_path, format='ttl')
# #     return [(str(s), str(p), str(o)) for s, p, o in g]


# def load_rdf_graph(file_path):
#     g = rdflib.Graph()
#     if file_path.endswith(".ttl"):
#         g.parse(file_path, format="ttl")
#     elif file_path.endswith(".owl") or file_path.endswith(".rdf") or file_path.endswith(".xml"):
#         g.parse(file_path, format="xml")
#     else:
#         raise ValueError(f"Unsupported file format for: {file_path}")
#     return [(str(s), str(p), str(o)) for s, p, o in g]

# def load_rdf_graph_gan(file_path):
#     """
#     Load RDF/OWL file and flatten restrictions for GAN.
#     """
#     g = rdflib.Graph()
#     if file_path.endswith(".ttl"):
#         g.parse(file_path, format="ttl")
#     elif file_path.endswith((".owl", ".rdf", ".xml")):
#         g.parse(file_path, format="xml")
#     else:
#         raise ValueError(f"Unsupported file format: {file_path}")

#     triples = []

#     for s in g.subjects(RDF.type, OWL.Class):
#         for o in g.objects(s, RDFS.subClassOf):
#             if (o, RDF.type, OWL.Restriction) in g:
#                 # Extract property and its value(s)
#                 prop = next(g.objects(o, OWL.onProperty))
#                 values = list(g.objects(o, OWL.someValuesFrom)) + list(g.objects(o, OWL.hasValue))
#                 for v in values:
#                     triples.append((str(s), str(prop), str(v)))
#             else:
#                 triples.append((str(s), str(RDFS.subClassOf), str(o)))

#     return triples


# # def factorize_and_initialize_gan(file_path):
# #     triples = load_rdf_graph(file_path)
# #     df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])

# #     subjects = pd.factorize(df['subject'])[0]
# #     predicates = pd.factorize(df['predicate'])[0]
# #     objects = pd.factorize(df['object'])[0]

# #     subject_dim = len(np.unique(subjects))
# #     predicate_dim = len(np.unique(predicates))
# #     object_dim = len(np.unique(objects))

# #     factorized_data = {
# #         "df": df,
# #         "subjects": subjects,
# #         "predicates": predicates,
# #         "objects": objects,
# #         "subject_dim": subject_dim,
# #         "predicate_dim": predicate_dim,
# #         "object_dim": object_dim,
# #         "subject_uniques": df["subject"].unique(),
# #         "predicate_uniques": df["predicate"].unique(),
# #         "object_inverse_map": dict(enumerate(df["object"].unique()))
# #     }

# #     generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
# #     discriminator = Discriminator(subject_dim, predicate_dim, object_dim)
# #     optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# #     optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
# #     return factorized_data, generator, discriminator, optimizer_g, optimizer_d


# def factorize_and_initialize_gan(file_path):
#     global factorized_data, generator, discriminator, optimizer_g, optimizer_d

#     triples = load_rdf_graph_gan(file_path)  # use the flattened loader
#     df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])

#     subjects = pd.factorize(df['subject'])[0]
#     predicates = pd.factorize(df['predicate'])[0]
#     objects = pd.factorize(df['object'])[0]

#     subject_dim = len(np.unique(subjects))
#     predicate_dim = len(np.unique(predicates))
#     object_dim = len(np.unique(objects))

#     factorized_data = {
#         "df": df,
#         "subjects": subjects,
#         "predicates": predicates,
#         "objects": objects,
#         "subject_dim": subject_dim,
#         "predicate_dim": predicate_dim,
#         "object_dim": object_dim,
#         "subject_uniques": df["subject"].unique(),
#         "predicate_uniques": df["predicate"].unique(),
#         "object_inverse_map": dict(enumerate(df["object"].unique())),
#         # convenience maps
#         "subject_map": {s.split("#")[-1]: s for s in df["subject"].unique()},
#         "predicate_map": {p.split("#")[-1]: p for p in df["predicate"].unique()},
#     }

#     # Initialize GAN
#     generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
#     discriminator = Discriminator(subject_dim, predicate_dim, object_dim)
#     optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

#     return factorized_data, generator, discriminator, optimizer_g, optimizer_d


# # def save_gan_to_mongo(model_name):
# #     global generator, discriminator, factorized_data, optimizer_g, optimizer_d

# #     # Serialize model states
# #     gen_bytes = io.BytesIO()
# #     disc_bytes = io.BytesIO()
# #     torch.save(generator.state_dict(), gen_bytes)
# #     torch.save(discriminator.state_dict(), disc_bytes)
# #     gen_bytes.seek(0)
# #     disc_bytes.seek(0)

# #     # Serialize factorized data
# #     factorized_bytes = pickle.dumps(factorized_data)

# #     doc = {
# #         "model_name": model_name,
# #         "generator": gen_bytes.read(),
# #         "discriminator": disc_bytes.read(),
# #         "factorized_data": factorized_bytes,
# #         "timestamp": torch.tensor([int(torch.time.time())])  # optional
# #     }

# #     # Upsert: replace if already exists
# #     gan_collection.update_one(
# #         {"model_name": model_name},
# #         {"$set": doc},
# #         upsert=True
# #     )
# #     print(f"✅ GAN model '{model_name}' saved to MongoDB.")

# # def load_gan_from_mongo(model_name):
# #     global generator, discriminator, factorized_data, optimizer_g, optimizer_d

# #     doc = gan_collection.find_one({"model_name": model_name})
# #     if not doc:
# #         raise FileNotFoundError(f"GAN model '{model_name}' not found in MongoDB.")

# #     # Load generator
# #     gen_bytes = io.BytesIO(doc["generator"])
# #     disc_bytes = io.BytesIO(doc["discriminator"])
# #     generator.load_state_dict(torch.load(gen_bytes))
# #     discriminator.load_state_dict(torch.load(disc_bytes))

# #     # Load factorized data
# #     factorized_data = pickle.loads(doc["factorized_data"])

# #     # Re-create optimizers
# #     optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# #     optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# #     loaded_models[model_name] = {
# #         "generator": generator,
# #         "discriminator": discriminator,
# #         "factorized_data": factorized_data,
# #         "optimizer_g": optimizer_g,
# #         "optimizer_d": optimizer_d
# #     }

# #     print(f"✅ GAN model '{model_name}' loaded from MongoDB.")




# # # --- Save GAN to MongoDB ---
# # def save_gan_to_mongo(model_name):
# #     global generator, discriminator, factorized_data, optimizer_g, optimizer_d

# #     # Serialize model states
# #     gen_bytes = io.BytesIO()
# #     disc_bytes = io.BytesIO()
# #     torch.save(generator.state_dict(), gen_bytes)
# #     torch.save(discriminator.state_dict(), disc_bytes)
# #     gen_bytes.seek(0)
# #     disc_bytes.seek(0)

# #     # Serialize factorized data
# #     factorized_bytes = pickle.dumps(factorized_data)

# #     doc = {
# #         "model_name": model_name,
# #         "generator": gen_bytes.read(),
# #         "discriminator": disc_bytes.read(),
# #         "factorized_data": factorized_bytes,
# #         "timestamp": int(time.time())  # current UNIX timestamp
# #     }

# #     # Upsert: replace if already exists
# #     gan_collection.update_one(
# #         {"model_name": model_name},
# #         {"$set": doc},
# #         upsert=True
# #     )
# #     print(f"✅ GAN model '{model_name}' saved to MongoDB.")


# # # --- Load GAN from MongoDB ---
# # def load_gan_from_mongo(model_name):
# #     global generator, discriminator, factorized_data, optimizer_g, optimizer_d

# #     doc = gan_collection.find_one({"model_name": model_name})
# #     if doc is None:
# #         raise ValueError(f"Model '{model_name}' not found in MongoDB.")

# #     # Deserialize factorized data
# #     factorized_data = pickle.loads(doc["factorized_data"])

# #     # Initialize generator and discriminator
# #     subject_dim = factorized_data["subject_dim"]
# #     predicate_dim = factorized_data["predicate_dim"]
# #     object_dim = factorized_data["object_dim"]

# #     generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
# #     discriminator = Discriminator(subject_dim, predicate_dim, object_dim)

# #     # Load weights
# #     generator.load_state_dict(torch.load(io.BytesIO(doc["generator"])))
# #     discriminator.load_state_dict(torch.load(io.BytesIO(doc["discriminator"])))

# #     # Initialize optimizers
# #     optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# #     optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# #     # Store in loaded_models for easy access
# #     loaded_models[model_name] = {
# #         "generator": generator,
# #         "discriminator": discriminator,
# #         "factorized_data": factorized_data,
# #         "optimizer_g": optimizer_g,
# #         "optimizer_d": optimizer_d
# #     }

# #     print(f"✅ GAN model '{model_name}' loaded from MongoDB.")
# #     return loaded_models[model_name]




# # --- Save GAN model to MongoDB ---
# def save_gan_to_mongo(model_name: str):
#     global generator, discriminator, factorized_data, optimizer_g, optimizer_d

#     if generator is None or discriminator is None:
#         raise RuntimeError("GAN models are not initialized.")

#     # Serialize model weights into bytes
#     gen_bytes = io.BytesIO()
#     disc_bytes = io.BytesIO()
#     torch.save(generator.state_dict(), gen_bytes)
#     torch.save(discriminator.state_dict(), disc_bytes)
#     gen_bytes.seek(0)
#     disc_bytes.seek(0)

#     # Serialize factorized data
#     factorized_bytes = pickle.dumps(factorized_data)

#     doc = {
#         "model_name": model_name,
#         "generator": gen_bytes.read(),
#         "discriminator": disc_bytes.read(),
#         "factorized_data": factorized_bytes,
#         "timestamp": int(torch.randint(1, 1_000_000_000, (1,)).item())  # optional integer timestamp
#     }

#     # Upsert: replace if already exists
#     gan_collection.update_one(
#         {"model_name": model_name},
#         {"$set": doc},
#         upsert=True
#     )

#     print(f"✅ GAN model '{model_name}' saved to MongoDB.")

# # # --- Load GAN model from MongoDB ---
# # def load_gan_from_mongo(model_name: str):
# #     global generator, discriminator, factorized_data, optimizer_g, optimizer_d

# #     doc = gan_collection.find_one({"model_name": model_name})
# #     if doc is None:
# #         raise ValueError(f"Model '{model_name}' not found in MongoDB.")

# #     # Deserialize factorized data
# #     factorized_data = pickle.loads(doc["factorized_data"])

# #     subject_dim = factorized_data["subject_dim"]
# #     predicate_dim = factorized_data["predicate_dim"]
# #     object_dim = factorized_data["object_dim"]

# #     # Initialize models
# #     generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
# #     discriminator = Discriminator(subject_dim, predicate_dim, object_dim)

# #     # Load weights safely
# #     gen_bytes = io.BytesIO(doc["generator"])
# #     gen_bytes.seek(0)
# #     generator.load_state_dict(torch.load(gen_bytes))

# #     disc_bytes = io.BytesIO(doc["discriminator"])
# #     disc_bytes.seek(0)
# #     discriminator.load_state_dict(torch.load(disc_bytes))

# #     # Initialize optimizers
# #     optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# #     optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# #     # Store in memory for quick access
# #     loaded_models[model_name] = {
# #         "generator": generator,
# #         "discriminator": discriminator,
# #         "factorized_data": factorized_data,
# #         "optimizer_g": optimizer_g,
# #         "optimizer_d": optimizer_d
# #     }

# #     print(f"✅ GAN model '{model_name}' loaded from MongoDB.")
# #     return loaded_models[model_name]


# def load_gan_from_mongo(model_name: str):
#     global generator, discriminator, factorized_data, optimizer_g, optimizer_d

#     doc = gan_collection.find_one({"model_name": model_name})
#     if doc is None:
#         raise ValueError(f"Model '{model_name}' not found in MongoDB.")

#     # Deserialize factorized data
#     factorized_data = pickle.loads(doc["factorized_data"])

#     subject_dim = factorized_data["subject_dim"]
#     predicate_dim = factorized_data["predicate_dim"]
#     object_dim = factorized_data["object_dim"]

#     # Initialize models
#     generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
#     discriminator = Discriminator(subject_dim, predicate_dim, object_dim)

#     # Load weights safely
#     generator.load_state_dict(torch.load(io.BytesIO(doc["generator"])))
#     discriminator.load_state_dict(torch.load(io.BytesIO(doc["discriminator"])))

#     # Initialize optimizers
#     optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

#     # Store in memory
#     loaded_models[model_name] = {
#         "generator": generator,
#         "discriminator": discriminator,
#         "factorized_data": factorized_data,
#         "optimizer_g": optimizer_g,
#         "optimizer_d": optimizer_d
#     }

#     print(f"✅ GAN model '{model_name}' loaded from MongoDB.")
#     return loaded_models[model_name]


# # --- Helper to load model if not already in memory ---
# def load_model_gan(model_name: str):
#     if model_name not in loaded_models:
#         return load_gan_from_mongo(model_name)
#     return loaded_models[model_name]

# async def load_gan_from_mongo_async(model_name: str):
#     doc = await gan_collection.find_one({"model_name": model_name})  # await it
#     if doc is None:
#         raise ValueError(f"Model '{model_name}' not found in MongoDB.")

#     factorized_data = pickle.loads(doc["factorized_data"])
#     subject_dim = factorized_data["subject_dim"]
#     predicate_dim = factorized_data["predicate_dim"]
#     object_dim = factorized_data["object_dim"]

#     generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
#     discriminator = Discriminator(subject_dim, predicate_dim, object_dim)
#     generator.load_state_dict(torch.load(io.BytesIO(doc["generator"])))
#     discriminator.load_state_dict(torch.load(io.BytesIO(doc["discriminator"])))

#     optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

#     loaded_models[model_name] = {
#         "generator": generator,
#         "discriminator": discriminator,
#         "factorized_data": factorized_data,
#         "optimizer_g": optimizer_g,
#         "optimizer_d": optimizer_d
#     }
#     return loaded_models[model_name]


# # # --- Async load from MongoDB ---
# # async def load_gan_from_mongo_async(model_name: str):
# #     global generator, discriminator, factorized_data, optimizer_g, optimizer_d

# #     doc = await gan_collection.find_one({"model_name": model_name})
# #     if doc is None:
# #         raise ValueError(f"Model '{model_name}' not found in MongoDB.")

# #     # Deserialize factorized data
# #     factorized_data = pickle.loads(doc["factorized_data"])

# #     subject_dim = factorized_data["subject_dim"]
# #     predicate_dim = factorized_data["predicate_dim"]
# #     object_dim = factorized_data["object_dim"]

# #     generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
# #     discriminator = Discriminator(subject_dim, predicate_dim, object_dim)

# #     # Load weights
# #     generator.load_state_dict(torch.load(io.BytesIO(doc["generator"])))
# #     discriminator.load_state_dict(torch.load(io.BytesIO(doc["discriminator"])))

# #     optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# #     optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# #     loaded_models[model_name] = {
# #         "generator": generator,
# #         "discriminator": discriminator,
# #         "factorized_data": factorized_data,
# #         "optimizer_g": optimizer_g,
# #         "optimizer_d": optimizer_d
# #     }

# #     return loaded_models[model_name]


# def factorize_and_initialize_gans(file_path):
#     global factorized_data, generator, discriminator, optimizer_g, optimizer_d

#     triples = load_rdf_graph(file_path)
#     df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])

#     subjects = pd.factorize(df['subject'])[0]
#     predicates = pd.factorize(df['predicate'])[0]
#     objects = pd.factorize(df['object'])[0]

#     subject_dim = len(np.unique(subjects))
#     predicate_dim = len(np.unique(predicates))
#     object_dim = len(np.unique(objects))

#     factorized_data = {
#         "df": df,
#         "subjects": subjects,
#         "predicates": predicates,
#         "objects": objects,
#         "subject_dim": subject_dim,
#         "predicate_dim": predicate_dim,
#         "object_dim": object_dim,
#         "subject_uniques": df["subject"].unique(),
#         "predicate_uniques": df["predicate"].unique(),
#         "object_inverse_map": dict(enumerate(df["object"].unique()))
#     }

#     generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
#     discriminator = Discriminator(subject_dim, predicate_dim, object_dim)
#     optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

#     return factorized_data, generator, discriminator, optimizer_g, optimizer_d



# class Generator(nn.Module):
#     def __init__(self, subject_dim, predicate_dim, object_dim, z_dim):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(z_dim + subject_dim + predicate_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, object_dim),
#             nn.Tanh()
#         )

#     def forward(self, z, subject, predicate):
#         return self.fc(torch.cat((z, subject, predicate), dim=1))

# class Discriminator(nn.Module):
#     def __init__(self, subject_dim, predicate_dim, object_dim):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(subject_dim + predicate_dim + object_dim, 128),
#             nn.LeakyReLU(0.2),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, subject, predicate, object):
#         return self.fc(torch.cat((subject, predicate, object), dim=1))


# def sample_noise(batch_size, z_dim, distribution="normal", dist_params=None):
#     """
#     dist_params is a dict with parameters depending on distribution type.
#     Example:
#       - normal: {'mean': 0, 'std': 1}
#       - uniform: {'low': -1, 'high': 1}
#       - skewed: {'skew': 3}  # example param
#       - categorical: {'probs': [0.1, 0.2, ..., 0.05]}  (length = z_dim)
#     """
#     if distribution == "uniform":
#         low = dist_params.get("low", -1) if dist_params else -1
#         high = dist_params.get("high", 1) if dist_params else 1
#         return torch.rand(batch_size, z_dim) * (high - low) + low
#     elif distribution == "skewed":
#         skew = dist_params.get("skew", 3) if dist_params else 3
#         base = torch.randn(batch_size, z_dim)
#         return base ** skew
#     elif distribution == "categorical":
#         probs = dist_params.get("probs") if dist_params else None
#         if probs is None:
#             probs = torch.ones(z_dim) / z_dim  # uniform categorical by default
#         else:
#             probs = torch.tensor(probs)
#         categorical_samples = torch.multinomial(probs, batch_size, replacement=True)
#         return torch.nn.functional.one_hot(categorical_samples, num_classes=z_dim).float()
#     else:  # default normal
#         mean = dist_params.get("mean", 0) if dist_params else 0
#         std = dist_params.get("std", 1) if dist_params else 1
#         return torch.randn(batch_size, z_dim) * std + mean



# def sample_noise(batch_size, z_dim, distribution="normal", dist_params=None):
#     """
#     dist_params is a dict with parameters depending on distribution type.
#     Example:
#       - normal: {'mean': 0, 'std': 1}
#       - uniform: {'low': -1, 'high': 1}
#       - skewed: {'skew': 3}  # example param
#       - categorical: {'probs': [0.1, 0.2, ..., 0.05]}  (length = z_dim)
#     """
#     if distribution == "uniform":
#         low = dist_params.get("low", -1) if dist_params else -1
#         high = dist_params.get("high", 1) if dist_params else 1
#         return torch.rand(batch_size, z_dim) * (high - low) + low
#     elif distribution == "skewed":
#         skew = dist_params.get("skew", 3) if dist_params else 3
#         base = torch.randn(batch_size, z_dim)
#         return base ** skew
#     elif distribution == "categorical":
#         probs = dist_params.get("probs") if dist_params else None
#         if probs is None:
#             probs = torch.ones(z_dim) / z_dim  # uniform categorical by default
#         else:
#             probs = torch.tensor(probs)
#         categorical_samples = torch.multinomial(probs, batch_size, replacement=True)
#         return torch.nn.functional.one_hot(categorical_samples, num_classes=z_dim).float()
#     else:  # default normal
#         mean = dist_params.get("mean", 0) if dist_params else 0
#         std = dist_params.get("std", 1) if dist_params else 1
#         return torch.randn(batch_size, z_dim) * std + mean



# def train_gan(num_epochs=1000, batch_size=64, distribution="normal",dist_params=None):
#     df = factorized_data["df"]
#     subjects = factorized_data["subjects"]
#     predicates = factorized_data["predicates"]
#     objects = factorized_data["objects"]
#     subject_dim = factorized_data["subject_dim"]
#     predicate_dim = factorized_data["predicate_dim"]
#     object_dim = factorized_data["object_dim"]

#     total_samples = len(df)
#     dynamic_batch_size = min(batch_size, total_samples)

#     for epoch in range(num_epochs):
#         for i in range(0, total_samples, dynamic_batch_size):
#             s = torch.tensor(subjects[i:i+dynamic_batch_size], dtype=torch.long)
#             p = torch.tensor(predicates[i:i+dynamic_batch_size], dtype=torch.long)
#             o = torch.tensor(objects[i:i+dynamic_batch_size], dtype=torch.long)

#             s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
#             p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()
#             o_oh = torch.nn.functional.one_hot(o, num_classes=object_dim).float()

#             current_batch_size = s_oh.size(0)
#             real_labels = torch.ones(current_batch_size, 1)
#             fake_labels = torch.zeros(current_batch_size, 1)

#             # Train discriminator
#             optimizer_d.zero_grad()
#             real_preds = discriminator(s_oh, p_oh, o_oh)
#             d_loss_real = criterion(real_preds, real_labels)

#             #z = sample_noise(current_batch_size, z_dim, distribution)
#             z = sample_noise(current_batch_size, z_dim, distribution, dist_params)
#             fake_objects = generator(z, s_oh, p_oh)
#             fake_preds = discriminator(s_oh, p_oh, fake_objects)
#             d_loss_fake = criterion(fake_preds, fake_labels)

#             d_loss = d_loss_real + d_loss_fake
#             d_loss.backward()
#             optimizer_d.step()

#             # Train generator
#             optimizer_g.zero_grad()
#             z = sample_noise(current_batch_size, z_dim, distribution)
#             fake_objects = generator(z, s_oh, p_oh)
#             fake_preds = discriminator(s_oh, p_oh, fake_objects)
#             g_loss = criterion(fake_preds, real_labels)
#             g_loss.backward()
#             optimizer_g.step()

#         if epoch % 100 == 0:
#             print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# def save_model(model_name):
#     os.makedirs(f"models/saved_models/gan/{model_name}", exist_ok=True)
#     torch.save(generator.state_dict(), f"models/saved_models/gan/{model_name}/generator.pth")
#     torch.save(discriminator.state_dict(), f"models/saved_models/gan/{model_name}/discriminator.pth")

# loaded_models = {}

# def find_uploaded_file(model_name):
#     for ext in [".ttl", ".owl", ".rdf", ".xml"]:
#         path = f"uploaded/{model_name}{ext}"
#         if os.path.exists(path):
#             return path
#     raise FileNotFoundError(f"Uploaded file for model '{model_name}' not found.")

# # def load_model(model_name):
# #     ttl_path = f"uploaded/{model_name}.ttl"
# #     factorized_data, generator, discriminator, optimizer_g, optimizer_d = factorize_and_initialize_gan(ttl_path)

# #     generator.load_state_dict(torch.load(f"models/saved_models/gan/{model_name}/generator.pth"))
# #     discriminator.load_state_dict(torch.load(f"models/saved_models/gan/{model_name}/discriminator.pth"))

# #     generator.eval()
# #     discriminator.eval()

# #     loaded_models[model_name] = {
# #         "generator": generator,
# #         "discriminator": discriminator,
# #         "factorized_data": factorized_data,
# #         "optimizer_g": optimizer_g,
# #         "optimizer_d": optimizer_d
# #     }

# #     print(f"✅ Model '{model_name}' loaded successfully.")

# def load_model(model_name):
#     global factorized_data, generator, discriminator, optimizer_g, optimizer_d

#     file_path = find_uploaded_file(model_name)

#     factorized_data, generator, discriminator, optimizer_g, optimizer_d = \
#         factorize_and_initialize_gan(file_path)

#     generator.load_state_dict(
#         torch.load(f"models/saved_models/gan/{model_name}/generator.pth")
#     )
#     discriminator.load_state_dict(
#         torch.load(f"models/saved_models/gan/{model_name}/discriminator.pth")
#     )

#     generator.eval()
#     discriminator.eval()

#     loaded_models[model_name] = {
#         "generator": generator,
#         "discriminator": discriminator,
#         "factorized_data": factorized_data,
#         "optimizer_g": optimizer_g,
#         "optimizer_d": optimizer_d
#     }

#     print(f"✅ Model '{model_name}' loaded successfully.")


# # def generate_synthetic_data(model_name, subject_input, predicate_input, num_samples=1, distribution="normal",dist_params=None):
# #     # If the model_name is "all", loop through all loaded models
# #     if model_name == "all":
# #         generated_objects = []
# #         for model_name, model in loaded_models.items():
# #             try:
# #                 subject_dim = model["factorized_data"]["subject_dim"]
# #                 predicate_dim = model["factorized_data"]["predicate_dim"]
# #                 generator = model["generator"]
# #                 factorized_data = model["factorized_data"]

# #                 subject_input_lower = subject_input.lower()
# #                 predicate_input_lower = predicate_input.lower()

# #                 subject_matches = [s for s in factorized_data["subject_uniques"] if subject_input_lower in s.lower()]
# #                 predicate_matches = [p for p in factorized_data["predicate_uniques"] if predicate_input_lower in p.lower()]

# #                 if len(subject_matches) == 0 or len(predicate_matches) == 0:
# #                     continue  # Skip this model if no matches found

# #                 subject_input = subject_matches[0]
# #                 predicate_input = predicate_matches[0]

# #                 subject_idx = np.where(factorized_data["subject_uniques"] == subject_input)[0][0]
# #                 predicate_idx = np.where(factorized_data["predicate_uniques"] == predicate_input)[0][0]

# #                 s = torch.tensor([subject_idx], dtype=torch.long)
# #                 p = torch.tensor([predicate_idx], dtype=torch.long)
# #                 s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
# #                 p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()

# #                 #z = sample_noise(num_samples, z_dim, distribution)
# #                 z = sample_noise(num_samples, z_dim, distribution, dist_params)
# #                 generated = generator(z, s_oh.repeat(num_samples, 1), p_oh.repeat(num_samples, 1)).detach().numpy()
# #                 generated_idx = np.argmax(generated, axis=1)
# #                 decoded_objects = [factorized_data["object_inverse_map"].get(idx, "UNKNOWN") for idx in generated_idx]
# #                 generated_objects.extend(decoded_objects)
# #             except Exception as e:
# #                 # Log error or handle exception for this particular model if needed
# #                 continue
# #         return generated_objects

# #     # If the model_name is specific, proceed as usual
# #     if model_name not in loaded_models:
# #         raise RuntimeError(f"Model '{model_name}' is not loaded.")
    
# #     model = loaded_models[model_name]
# #     generator = model["generator"]
# #     factorized_data = model["factorized_data"]

# #     if generator is None:
# #         raise RuntimeError("Generator model is not loaded.")
    
# #     subject_dim = factorized_data["subject_dim"]
# #     predicate_dim = factorized_data["predicate_dim"]

# #     subject_input_lower = subject_input.lower()
# #     predicate_input_lower = predicate_input.lower()

# #     subject_matches = [s for s in factorized_data["subject_uniques"] if subject_input_lower in s.lower()]
# #     predicate_matches = [p for p in factorized_data["predicate_uniques"] if predicate_input_lower in p.lower()]

# #     if len(subject_matches) == 0:
# #         raise ValueError(f"Subject '{subject_input}' not found.")
# #     if len(predicate_matches) == 0:
# #         raise ValueError(f"Predicate '{predicate_input}' not found.")

# #     subject_input = subject_matches[0]
# #     predicate_input = predicate_matches[0]

# #     subject_idx = np.where(factorized_data["subject_uniques"] == subject_input)[0][0]
# #     predicate_idx = np.where(factorized_data["predicate_uniques"] == predicate_input)[0][0]

# #     s = torch.tensor([subject_idx], dtype=torch.long)
# #     p = torch.tensor([predicate_idx], dtype=torch.long)
# #     s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
# #     p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()

# #     z = sample_noise(num_samples, z_dim, distribution)
# #     generated = generator(z, s_oh.repeat(num_samples, 1), p_oh.repeat(num_samples, 1)).detach().numpy()
# #     generated_idx = np.argmax(generated, axis=1)
# #     decoded_objects = [factorized_data["object_inverse_map"].get(idx, "UNKNOWN") for idx in generated_idx]
# #     return decoded_objects


# def generate_synthetic_data(model_name, subject_input, predicate_input, num_samples=1, distribution="normal", dist_params=None):
#     """
#     Generate synthetic objects using GAN, supporting both 'all' models or a specific model.
#     Supports short names for predicates/subjects.
#     """
#     def map_local_name(factorized_data, local_name, key="predicate"):
#         """
#         Map local name to full URI if available.
#         """
#         mapping = factorized_data.get(f"{key}_map", {})
#         return mapping.get(local_name, local_name)

#     # If the model_name is "all", loop through all loaded models
#     if model_name == "all":
#         generated_objects = []
#         for model_name, model in loaded_models.items():
#             try:
#                 factorized_data = model["factorized_data"]
#                 generator = model["generator"]

#                 subject_uri = map_local_name(factorized_data, subject_input, "subject")
#                 predicate_uri = map_local_name(factorized_data, predicate_input, "predicate")

#                 # Match URIs to indices
#                 subject_idx = np.where(factorized_data["subject_uniques"] == subject_uri)[0]
#                 predicate_idx = np.where(factorized_data["predicate_uniques"] == predicate_uri)[0]

#                 if len(subject_idx) == 0 or len(predicate_idx) == 0:
#                     continue  # Skip if no match

#                 s = torch.tensor([subject_idx[0]], dtype=torch.long)
#                 p = torch.tensor([predicate_idx[0]], dtype=torch.long)
#                 s_oh = torch.nn.functional.one_hot(s, num_classes=factorized_data["subject_dim"]).float()
#                 p_oh = torch.nn.functional.one_hot(p, num_classes=factorized_data["predicate_dim"]).float()

#                 z = sample_noise(num_samples, z_dim, distribution, dist_params)
#                 generated = generator(z, s_oh.repeat(num_samples, 1), p_oh.repeat(num_samples, 1)).detach().numpy()
#                 generated_idx = np.argmax(generated, axis=1)
#                 decoded_objects = [factorized_data["object_inverse_map"].get(idx, "UNKNOWN") for idx in generated_idx]
#                 generated_objects.extend(decoded_objects)
#             except Exception:
#                 continue  # skip model on error
#         return generated_objects

#     # If the model_name is specific, proceed as usual
#     if model_name not in loaded_models:
#         raise RuntimeError(f"Model '{model_name}' is not loaded.")
    
#     model = loaded_models[model_name]
#     generator = model["generator"]
#     factorized_data = model["factorized_data"]

#     if generator is None:
#         raise RuntimeError("Generator model is not loaded.")

#     # Map short names to full URIs
#     subject_uri = map_local_name(factorized_data, subject_input, "subject")
#     predicate_uri = map_local_name(factorized_data, predicate_input, "predicate")

#     # Match URIs to indices
#     subject_idx = np.where(factorized_data["subject_uniques"] == subject_uri)[0]
#     predicate_idx = np.where(factorized_data["predicate_uniques"] == predicate_uri)[0]

#     if len(subject_idx) == 0:
#         raise ValueError(f"Subject '{subject_input}' not found.")
#     if len(predicate_idx) == 0:
#         raise ValueError(f"Predicate '{predicate_input}' not found.")

#     s = torch.tensor([subject_idx[0]], dtype=torch.long)
#     p = torch.tensor([predicate_idx[0]], dtype=torch.long)
#     s_oh = torch.nn.functional.one_hot(s, num_classes=factorized_data["subject_dim"]).float()
#     p_oh = torch.nn.functional.one_hot(p, num_classes=factorized_data["predicate_dim"]).float()

#     z = sample_noise(num_samples, z_dim, distribution, dist_params)
#     generated = generator(z, s_oh.repeat(num_samples, 1), p_oh.repeat(num_samples, 1)).detach().numpy()
#     generated_idx = np.argmax(generated, axis=1)
#     decoded_objects = [factorized_data["object_inverse_map"].get(idx, "UNKNOWN") for idx in generated_idx]

#     return decoded_objects


import os
import torch
import torch.nn as nn
import torch.optim as optim
import rdflib
import numpy as np
import pandas as pd
from db.mongo import gan_collection
import io
import pickle
import time

factorized_data = {}
generator = None
discriminator = None
optimizer_g = None
optimizer_d = None
criterion = nn.BCELoss()
z_dim = 100

# def load_rdf_graph(file_path):
#     g = rdflib.Graph()
#     g.parse(file_path, format='ttl')
#     return [(str(s), str(p), str(o)) for s, p, o in g]


def load_rdf_graph(file_path):
    g = rdflib.Graph()
    if file_path.endswith(".ttl"):
        g.parse(file_path, format="ttl")
    elif file_path.endswith(".owl") or file_path.endswith(".rdf") or file_path.endswith(".xml"):
        g.parse(file_path, format="xml")
    else:
        raise ValueError(f"Unsupported file format for: {file_path}")
    return [(str(s), str(p), str(o)) for s, p, o in g]

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

# def save_gan_to_mongo(model_name):
#     global generator, discriminator, factorized_data, optimizer_g, optimizer_d

#     # Serialize model states
#     gen_bytes = io.BytesIO()
#     disc_bytes = io.BytesIO()
#     torch.save(generator.state_dict(), gen_bytes)
#     torch.save(discriminator.state_dict(), disc_bytes)
#     gen_bytes.seek(0)
#     disc_bytes.seek(0)

#     # Serialize factorized data
#     factorized_bytes = pickle.dumps(factorized_data)

#     doc = {
#         "model_name": model_name,
#         "generator": gen_bytes.read(),
#         "discriminator": disc_bytes.read(),
#         "factorized_data": factorized_bytes,
#         "timestamp": torch.tensor([int(torch.time.time())])  # optional
#     }

#     # Upsert: replace if already exists
#     gan_collection.update_one(
#         {"model_name": model_name},
#         {"$set": doc},
#         upsert=True
#     )
#     print(f"✅ GAN model '{model_name}' saved to MongoDB.")

# def load_gan_from_mongo(model_name):
#     global generator, discriminator, factorized_data, optimizer_g, optimizer_d

#     doc = gan_collection.find_one({"model_name": model_name})
#     if not doc:
#         raise FileNotFoundError(f"GAN model '{model_name}' not found in MongoDB.")

#     # Load generator
#     gen_bytes = io.BytesIO(doc["generator"])
#     disc_bytes = io.BytesIO(doc["discriminator"])
#     generator.load_state_dict(torch.load(gen_bytes))
#     discriminator.load_state_dict(torch.load(disc_bytes))

#     # Load factorized data
#     factorized_data = pickle.loads(doc["factorized_data"])

#     # Re-create optimizers
#     optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

#     loaded_models[model_name] = {
#         "generator": generator,
#         "discriminator": discriminator,
#         "factorized_data": factorized_data,
#         "optimizer_g": optimizer_g,
#         "optimizer_d": optimizer_d
#     }

#     print(f"✅ GAN model '{model_name}' loaded from MongoDB.")




# # --- Save GAN to MongoDB ---
# def save_gan_to_mongo(model_name):
#     global generator, discriminator, factorized_data, optimizer_g, optimizer_d

#     # Serialize model states
#     gen_bytes = io.BytesIO()
#     disc_bytes = io.BytesIO()
#     torch.save(generator.state_dict(), gen_bytes)
#     torch.save(discriminator.state_dict(), disc_bytes)
#     gen_bytes.seek(0)
#     disc_bytes.seek(0)

#     # Serialize factorized data
#     factorized_bytes = pickle.dumps(factorized_data)

#     doc = {
#         "model_name": model_name,
#         "generator": gen_bytes.read(),
#         "discriminator": disc_bytes.read(),
#         "factorized_data": factorized_bytes,
#         "timestamp": int(time.time())  # current UNIX timestamp
#     }

#     # Upsert: replace if already exists
#     gan_collection.update_one(
#         {"model_name": model_name},
#         {"$set": doc},
#         upsert=True
#     )
#     print(f"✅ GAN model '{model_name}' saved to MongoDB.")


# # --- Load GAN from MongoDB ---
# def load_gan_from_mongo(model_name):
#     global generator, discriminator, factorized_data, optimizer_g, optimizer_d

#     doc = gan_collection.find_one({"model_name": model_name})
#     if doc is None:
#         raise ValueError(f"Model '{model_name}' not found in MongoDB.")

#     # Deserialize factorized data
#     factorized_data = pickle.loads(doc["factorized_data"])

#     # Initialize generator and discriminator
#     subject_dim = factorized_data["subject_dim"]
#     predicate_dim = factorized_data["predicate_dim"]
#     object_dim = factorized_data["object_dim"]

#     generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
#     discriminator = Discriminator(subject_dim, predicate_dim, object_dim)

#     # Load weights
#     generator.load_state_dict(torch.load(io.BytesIO(doc["generator"])))
#     discriminator.load_state_dict(torch.load(io.BytesIO(doc["discriminator"])))

#     # Initialize optimizers
#     optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

#     # Store in loaded_models for easy access
#     loaded_models[model_name] = {
#         "generator": generator,
#         "discriminator": discriminator,
#         "factorized_data": factorized_data,
#         "optimizer_g": optimizer_g,
#         "optimizer_d": optimizer_d
#     }

#     print(f"✅ GAN model '{model_name}' loaded from MongoDB.")
#     return loaded_models[model_name]




# --- Save GAN model to MongoDB ---
def save_gan_to_mongo(model_name: str):
    global generator, discriminator, factorized_data, optimizer_g, optimizer_d

    if generator is None or discriminator is None:
        raise RuntimeError("GAN models are not initialized.")

    # Serialize model weights into bytes
    gen_bytes = io.BytesIO()
    disc_bytes = io.BytesIO()
    torch.save(generator.state_dict(), gen_bytes)
    torch.save(discriminator.state_dict(), disc_bytes)
    gen_bytes.seek(0)
    disc_bytes.seek(0)

    # Serialize factorized data
    factorized_bytes = pickle.dumps(factorized_data)

    doc = {
        "model_name": model_name,
        "generator": gen_bytes.read(),
        "discriminator": disc_bytes.read(),
        "factorized_data": factorized_bytes,
        "timestamp": int(torch.randint(1, 1_000_000_000, (1,)).item())  # optional integer timestamp
    }

    # Upsert: replace if already exists
    gan_collection.update_one(
        {"model_name": model_name},
        {"$set": doc},
        upsert=True
    )

    print(f"✅ GAN model '{model_name}' saved to MongoDB.")

# --- Load GAN model from MongoDB ---
def load_gan_from_mongo(model_name: str):
    global generator, discriminator, factorized_data, optimizer_g, optimizer_d

    doc = gan_collection.find_one({"model_name": model_name})
    if doc is None:
        raise ValueError(f"Model '{model_name}' not found in MongoDB.")

    # Deserialize factorized data
    factorized_data = pickle.loads(doc["factorized_data"])

    subject_dim = factorized_data["subject_dim"]
    predicate_dim = factorized_data["predicate_dim"]
    object_dim = factorized_data["object_dim"]

    # Initialize models
    generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
    discriminator = Discriminator(subject_dim, predicate_dim, object_dim)

    # Load weights safely
    gen_bytes = io.BytesIO(doc["generator"])
    gen_bytes.seek(0)
    generator.load_state_dict(torch.load(gen_bytes))

    disc_bytes = io.BytesIO(doc["discriminator"])
    disc_bytes.seek(0)
    discriminator.load_state_dict(torch.load(disc_bytes))

    # Initialize optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Store in memory for quick access
    loaded_models[model_name] = {
        "generator": generator,
        "discriminator": discriminator,
        "factorized_data": factorized_data,
        "optimizer_g": optimizer_g,
        "optimizer_d": optimizer_d
    }

    print(f"✅ GAN model '{model_name}' loaded from MongoDB.")
    return loaded_models[model_name]

# --- Helper to load model if not already in memory ---
def load_model_gan(model_name: str):
    if model_name not in loaded_models:
        return load_gan_from_mongo(model_name)
    return loaded_models[model_name]



# --- Async load from MongoDB ---
async def load_gan_from_mongo_async(model_name: str):
    global generator, discriminator, factorized_data, optimizer_g, optimizer_d

    doc = await gan_collection.find_one({"model_name": model_name})
    if doc is None:
        raise ValueError(f"Model '{model_name}' not found in MongoDB.")

    # Deserialize factorized data
    factorized_data = pickle.loads(doc["factorized_data"])

    subject_dim = factorized_data["subject_dim"]
    predicate_dim = factorized_data["predicate_dim"]
    object_dim = factorized_data["object_dim"]

    generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
    discriminator = Discriminator(subject_dim, predicate_dim, object_dim)

    # Load weights
    generator.load_state_dict(torch.load(io.BytesIO(doc["generator"])))
    discriminator.load_state_dict(torch.load(io.BytesIO(doc["discriminator"])))

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    loaded_models[model_name] = {
        "generator": generator,
        "discriminator": discriminator,
        "factorized_data": factorized_data,
        "optimizer_g": optimizer_g,
        "optimizer_d": optimizer_d
    }

    return loaded_models[model_name]


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

def find_uploaded_file(model_name):
    for ext in [".ttl", ".owl", ".rdf", ".xml"]:
        path = f"uploaded/{model_name}{ext}"
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Uploaded file for model '{model_name}' not found.")

# def load_model(model_name):
#     ttl_path = f"uploaded/{model_name}.ttl"
#     factorized_data, generator, discriminator, optimizer_g, optimizer_d = factorize_and_initialize_gan(ttl_path)

#     generator.load_state_dict(torch.load(f"models/saved_models/gan/{model_name}/generator.pth"))
#     discriminator.load_state_dict(torch.load(f"models/saved_models/gan/{model_name}/discriminator.pth"))

#     generator.eval()
#     discriminator.eval()

#     loaded_models[model_name] = {
#         "generator": generator,
#         "discriminator": discriminator,
#         "factorized_data": factorized_data,
#         "optimizer_g": optimizer_g,
#         "optimizer_d": optimizer_d
#     }

#     print(f"✅ Model '{model_name}' loaded successfully.")

def load_model(model_name):
    global factorized_data, generator, discriminator, optimizer_g, optimizer_d

    file_path = find_uploaded_file(model_name)

    factorized_data, generator, discriminator, optimizer_g, optimizer_d = \
        factorize_and_initialize_gan(file_path)

    generator.load_state_dict(
        torch.load(f"models/saved_models/gan/{model_name}/generator.pth")
    )
    discriminator.load_state_dict(
        torch.load(f"models/saved_models/gan/{model_name}/discriminator.pth")
    )

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


def generate_synthetic_data(model_name, subject_input, predicate_input, num_samples=1, distribution="normal",dist_params=None):
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

