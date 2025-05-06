# # import torch
# # import torch.nn as nn
# # import torch.optim as optim

# # # Define the Generator and Discriminator models (same as before)
# # class Generator(nn.Module):
# #     def __init__(self, input_dim, output_dim):
# #         super(Generator, self).__init__()
# #         self.fc = nn.Sequential(
# #             nn.Linear(input_dim, 128),
# #             nn.ReLU(),
# #             nn.Linear(128, output_dim),
# #             nn.Tanh()  # To scale output values between -1 and 1
# #         )

# #     def forward(self, z):
# #         return self.fc(z)

# # class Discriminator(nn.Module):
# #     def __init__(self, input_dim):
# #         super(Discriminator, self).__init__()
# #         self.fc = nn.Sequential(
# #             nn.Linear(input_dim, 128),
# #             nn.LeakyReLU(0.2),
# #             nn.Linear(128, 1),
# #             nn.Sigmoid()
# #         )

# #     def forward(self, x):
# #         return self.fc(x)

# # # Instantiate the Generator
# # input_dim = 100  # Noise dimension
# # data_dim = 10    # Dimensions of generated data (e.g., RDF triples or a simplified representation)
# # generator = Generator(input_dim, data_dim)

# # # Load the pre-trained weights (if applicable)
# # # generator.load_state_dict(torch.load("path_to_saved_model.pth"))
# # # generator.eval()

# # # Function to generate synthetic data using the trained generator
# # def generate_synthetic_data(num_samples: int):
# #     z = torch.randn(num_samples, input_dim)
# #     generated_data = generator(z).detach().numpy()  # Convert to NumPy array
# #     return generated_data

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import rdflib
# import numpy as np
# import pandas as pd

# # Load DBpedia data from a TTL file
# def load_dbpedia_ttl(file_path):
#     g = rdflib.Graph()
#     g.parse(file_path, format='ttl')

#     # Extract subject-predicate-object triples
#     triples = []
#     for subj, pred, obj in g:
#         # Convert URIs to string and store as a tuple (subject, predicate, object)
#         triples.append((str(subj), str(pred), str(obj)))

#     return triples

# # Example: Load DBpedia TTL file (replace with your actual file path)
# file_path = "dbpedia_data.ttl"
# triples = load_dbpedia_ttl(file_path)

# # Create a DataFrame for easier manipulation
# df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])

# # Preprocess the data (e.g., map subjects, predicates, and objects to integer indices)
# subjects = pd.factorize(df['subject'])[0]
# predicates = pd.factorize(df['predicate'])[0]
# objects = pd.factorize(df['object'])[0]

# # Create a mapping for subjects, predicates, and objects
# subject_dim = len(np.unique(subjects))
# predicate_dim = len(np.unique(predicates))
# object_dim = len(np.unique(objects))

# # Define the Generator and Discriminator models (Conditional GAN)
# class Generator(nn.Module):
#     def __init__(self, subject_dim, predicate_dim, object_dim, z_dim):
#         super(Generator, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(z_dim + subject_dim + predicate_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, object_dim),
#             nn.Tanh()  # To scale output values between -1 and 1 (or between valid object range)
#         )

#     def forward(self, z, subject, predicate):
#         # Concatenate the noise vector with subject and predicate embeddings
#         input_data = torch.cat((z, subject, predicate), dim=1)
#         return self.fc(input_data)

# class Discriminator(nn.Module):
#     def __init__(self, subject_dim, predicate_dim, object_dim):
#         super(Discriminator, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(subject_dim + predicate_dim + object_dim, 128),
#             nn.LeakyReLU(0.2),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, subject, predicate, object):
#         # Concatenate subject, predicate, and object for discriminator
#         input_data = torch.cat((subject, predicate, object), dim=1)
#         return self.fc(input_data)

# # Instantiate the Generator and Discriminator
# z_dim = 100  # Noise dimension
# generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
# discriminator = Discriminator(subject_dim, predicate_dim, object_dim)

# # Define the optimizers
# lr = 0.0002
# optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
# optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# # Loss function
# criterion = nn.BCELoss()

# # Training loop
# def train_gan(num_epochs=1000, batch_size=64):
#     for epoch in range(num_epochs):
#         for i in range(0, len(df), batch_size):
#             # Get the batch of data
#             subjects_batch = torch.tensor(subjects[i:i+batch_size], dtype=torch.long)
#             predicates_batch = torch.tensor(predicates[i:i+batch_size], dtype=torch.long)
#             objects_batch = torch.tensor(objects[i:i+batch_size], dtype=torch.long)
            
#             # Create one-hot encodings for subjects, predicates, and objects
#             subject_onehot = torch.nn.functional.one_hot(subjects_batch, num_classes=subject_dim).float()
#             predicate_onehot = torch.nn.functional.one_hot(predicates_batch, num_classes=predicate_dim).float()
#             object_onehot = torch.nn.functional.one_hot(objects_batch, num_classes=object_dim).float()

#             # Train Discriminator
#             real_labels = torch.ones(batch_size, 1)
#             fake_labels = torch.zeros(batch_size, 1)

#             # Compute discriminator loss on real data
#             optimizer_d.zero_grad()
#             real_preds = discriminator(subject_onehot, predicate_onehot, object_onehot)
#             d_loss_real = criterion(real_preds, real_labels)

#             # Generate fake data and compute discriminator loss
#             z = torch.randn(batch_size, z_dim)
#             fake_objects = generator(z, subject_onehot, predicate_onehot)
#             fake_preds = discriminator(subject_onehot, predicate_onehot, fake_objects)
#             d_loss_fake = criterion(fake_preds, fake_labels)

#             # Total discriminator loss and backprop
#             d_loss = d_loss_real + d_loss_fake
#             d_loss.backward()
#             optimizer_d.step()

#             # Train Generator
#             optimizer_g.zero_grad()
#             fake_objects = generator(z, subject_onehot, predicate_onehot)
#             fake_preds = discriminator(subject_onehot, predicate_onehot, fake_objects)

#             # Generator loss (we want fake data to be classified as real)
#             g_loss = criterion(fake_preds, real_labels)
#             g_loss.backward()
#             optimizer_g.step()

#         if epoch % 100 == 0:
#             print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# # Train the GAN
# train_gan()

# # # Function to generate synthetic data (with subject-predicate input)
# # def generate_synthetic_data(subject_input, predicate_input, num_samples=1):
# #     subject_tensor = torch.tensor(subject_input, dtype=torch.long).unsqueeze(0)
# #     predicate_tensor = torch.tensor(predicate_input, dtype=torch.long).unsqueeze(0)
    
# #     # One-hot encode the inputs
# #     subject_onehot = torch.nn.functional.one_hot(subject_tensor, num_classes=subject_dim).float()
# #     predicate_onehot = torch.nn.functional.one_hot(predicate_tensor, num_classes=predicate_dim).float()

# #     # Generate synthetic object data
# #     z = torch.randn(num_samples, z_dim)
# #     generated_objects = generator(z, subject_onehot, predicate_onehot).detach().numpy()

# #     return generated_objects

# # # Example usage: generate synthetic object for a given subject and predicate
# # subject_example = 5  # Example subject index
# # predicate_example = 10  # Example predicate index
# # synthetic_objects = generate_synthetic_data(subject_example, predicate_example, num_samples=5)
# # print(synthetic_objects)

# # Function to generate synthetic data (with subject-predicate input)
# def generate_synthetic_data(subject_input, predicate_input, num_samples=1):
#     subject_tensor = torch.tensor(subject_input, dtype=torch.long).unsqueeze(0)
#     predicate_tensor = torch.tensor(predicate_input, dtype=torch.long).unsqueeze(0)
    
#     # One-hot encode the inputs
#     subject_onehot = torch.nn.functional.one_hot(subject_tensor, num_classes=subject_dim).float()
#     predicate_onehot = torch.nn.functional.one_hot(predicate_tensor, num_classes=predicate_dim).float()

#     # Generate synthetic object data
#     z = torch.randn(num_samples, z_dim)
#     generated_objects = generator(z, subject_onehot, predicate_onehot).detach().numpy()

#     return generated_objects
# if __name__ == "__main__":
#     # Train the GAN
#     train_gan()

#     # Example usage: generate synthetic object for a given subject and predicate
#     subject_example = 5  # Example subject index
#     predicate_example = 10  # Example predicate index
#     synthetic_objects = generate_synthetic_data(subject_example, predicate_example, num_samples=5)
#     print(synthetic_objects)


# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List
# from models.gan_model import generate_synthetic_data

# app = FastAPI()

# class SyntheticRequest(BaseModel):
#     subject_index: int
#     predicate_index: int
#     num_samples: int = 1

# class SyntheticResponse(BaseModel):
#     synthetic_objects: List[List[float]]

# @app.post("/generate", response_model=SyntheticResponse)
# def generate(request: SyntheticRequest):
#     result = generate_synthetic_data(
#         subject_input=request.subject_index,
#         predicate_input=request.predicate_index,
#         num_samples=request.num_samples
#     )
#     return {"synthetic_objects": result.tolist()}



# # models/gan_model.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import rdflib
# import numpy as np
# import pandas as pd

# # Load DBpedia data
# file_path = "dbpedia_data.ttl"
# def load_dbpedia_ttl(file_path):
#     g = rdflib.Graph()
#     g.parse(file_path, format='ttl')
#     return [(str(s), str(p), str(o)) for s, p, o in g]

# triples = load_dbpedia_ttl(file_path)
# df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])

# subjects = pd.factorize(df['subject'])[0]
# predicates = pd.factorize(df['predicate'])[0]
# objects = pd.factorize(df['object'])[0]

# subject_dim = len(np.unique(subjects))
# predicate_dim = len(np.unique(predicates))
# object_dim = len(np.unique(objects))

# z_dim = 100

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

# generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
# discriminator = Discriminator(subject_dim, predicate_dim, object_dim)
# optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# criterion = nn.BCELoss()

# def train_gan(num_epochs=1000, batch_size=64):
#     df = factorized_data["df"]
#     subjects = factorized_data["subjects"]
#     predicates = factorized_data["predicates"]
#     objects = factorized_data["objects"]
#     subject_dim = factorized_data["subject_dim"]
#     predicate_dim = factorized_data["predicate_dim"]
#     object_dim = factorized_data["object_dim"]
#     for epoch in range(num_epochs):
#         for i in range(0, len(df), batch_size):
#             s = torch.tensor(subjects[i:i+batch_size], dtype=torch.long)
#             p = torch.tensor(predicates[i:i+batch_size], dtype=torch.long)
#             o = torch.tensor(objects[i:i+batch_size], dtype=torch.long)

#             s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
#             p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()
#             o_oh = torch.nn.functional.one_hot(o, num_classes=object_dim).float()

#             real_labels = torch.ones(batch_size, 1)
#             fake_labels = torch.zeros(batch_size, 1)

#             optimizer_d.zero_grad()
#             real_preds = discriminator(s_oh, p_oh, o_oh)
#             d_loss_real = criterion(real_preds, real_labels)

#             z = torch.randn(batch_size, z_dim)
#             fake_objects = generator(z, s_oh, p_oh)
#             fake_preds = discriminator(s_oh, p_oh, fake_objects)
#             d_loss_fake = criterion(fake_preds, fake_labels)

#             d_loss = d_loss_real + d_loss_fake
#             d_loss.backward()
#             optimizer_d.step()

#             optimizer_g.zero_grad()
#             fake_objects = generator(z, s_oh, p_oh)
#             fake_preds = discriminator(s_oh, p_oh, fake_objects)
#             g_loss = criterion(fake_preds, real_labels)
#             g_loss.backward()
#             optimizer_g.step()

#         if epoch % 100 == 0:
#             print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# # def generate_synthetic_data(subject_input, predicate_input, num_samples=1):
# #     s = torch.tensor(subject_input, dtype=torch.long).unsqueeze(0)
# #     p = torch.tensor(predicate_input, dtype=torch.long).unsqueeze(0)
# #     s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
# #     p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()
# #     z = torch.randn(num_samples, z_dim)
# #     generated = generator(z, s_oh, p_oh).detach().numpy()
# #     return generated


# def generate_synthetic_data(subject_input, predicate_input, num_samples=1):
#     subject_dim = factorized_data["subject_dim"]
#     predicate_dim = factorized_data["predicate_dim"]
    
#     s = torch.tensor(subject_input, dtype=torch.long).unsqueeze(0)
#     p = torch.tensor(predicate_input, dtype=torch.long).unsqueeze(0)
#     s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
#     p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()
#     z = torch.randn(num_samples, z_dim)
#     generated = generator(z, s_oh, p_oh).detach().numpy()
#     return generated


# # models/gan_model.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import rdflib
# import numpy as np
# import pandas as pd

# factorized_data = {}
# generator = None
# discriminator = None
# optimizer_g = None
# optimizer_d = None
# criterion = nn.BCELoss()
# z_dim = 100

# def load_dbpedia_ttl(file_path):
#     g = rdflib.Graph()
#     g.parse(file_path, format='ttl')
#     return [(str(s), str(p), str(o)) for s, p, o in g]

# def factorize_and_initialize_gan(file_path):
#     global factorized_data, generator, discriminator, optimizer_g, optimizer_d

#     triples = load_dbpedia_ttl(file_path)
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
#     }

#     generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
#     discriminator = Discriminator(subject_dim, predicate_dim, object_dim)
#     optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

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

# def train_gan(num_epochs=1000, batch_size=64):
#     df = factorized_data["df"]
#     subjects = factorized_data["subjects"]
#     predicates = factorized_data["predicates"]
#     objects = factorized_data["objects"]
#     subject_dim = factorized_data["subject_dim"]
#     predicate_dim = factorized_data["predicate_dim"]
#     object_dim = factorized_data["object_dim"]

#     for epoch in range(num_epochs):
#         for i in range(0, len(df), batch_size):
#             s = torch.tensor(subjects[i:i+batch_size], dtype=torch.long)
#             p = torch.tensor(predicates[i:i+batch_size], dtype=torch.long)
#             o = torch.tensor(objects[i:i+batch_size], dtype=torch.long)

#             s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
#             p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()
#             o_oh = torch.nn.functional.one_hot(o, num_classes=object_dim).float()

#             real_labels = torch.ones(batch_size, 1)
#             fake_labels = torch.zeros(batch_size, 1)

#             optimizer_d.zero_grad()
#             real_preds = discriminator(s_oh, p_oh, o_oh)
#             d_loss_real = criterion(real_preds, real_labels)

#             z = torch.randn(batch_size, z_dim)
#             fake_objects = generator(z, s_oh, p_oh)
#             fake_preds = discriminator(s_oh, p_oh, fake_objects)
#             d_loss_fake = criterion(fake_preds, fake_labels)

#             d_loss = d_loss_real + d_loss_fake
#             d_loss.backward()
#             optimizer_d.step()

#             optimizer_g.zero_grad()
#             fake_objects = generator(z, s_oh, p_oh)
#             fake_preds = discriminator(s_oh, p_oh, fake_objects)
#             g_loss = criterion(fake_preds, real_labels)
#             g_loss.backward()
#             optimizer_g.step()

#         if epoch % 100 == 0:
#             print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# def generate_synthetic_data(subject_input, predicate_input, num_samples=1):
#     subject_dim = factorized_data["subject_dim"]
#     predicate_dim = factorized_data["predicate_dim"]

#     s = torch.tensor(subject_input, dtype=torch.long).unsqueeze(0)
#     p = torch.tensor(predicate_input, dtype=torch.long).unsqueeze(0)
#     s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
#     p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()
#     z = torch.randn(num_samples, z_dim)
#     generated = generator(z, s_oh, p_oh).detach().numpy()
#     return generated


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import rdflib
# import numpy as np
# import pandas as pd

# factorized_data = {}
# generator = None
# discriminator = None
# optimizer_g = None
# optimizer_d = None
# criterion = nn.BCELoss()
# z_dim = 100

# def load_dbpedia_ttl(file_path):
#     g = rdflib.Graph()
#     g.parse(file_path, format='ttl')
#     return [(str(s), str(p), str(o)) for s, p, o in g]

# # def factorize_and_initialize_gan(file_path):
# #     global factorized_data, generator, discriminator, optimizer_g, optimizer_d

# #     triples = load_dbpedia_ttl(file_path)
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
# #     }

# #     generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
# #     discriminator = Discriminator(subject_dim, predicate_dim, object_dim)
# #     optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# #     optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# def factorize_and_initialize_gan(file_path):
#     global factorized_data, generator, discriminator, optimizer_g, optimizer_d

#     # Load and factorize the data
#     triples = load_dbpedia_ttl(file_path)
#     df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])

#     subjects = pd.factorize(df['subject'])[0]
#     predicates = pd.factorize(df['predicate'])[0]
#     objects = pd.factorize(df['object'])[0]

#     subject_dim = len(np.unique(subjects))
#     predicate_dim = len(np.unique(predicates))
#     object_dim = len(np.unique(objects))

#     # Store the data in the global dictionary
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
#     }

#     factorized_data["object_inverse_map"] = dict(enumerate(df["object"].unique()))

#     # Initialize GAN components
#     generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
#     discriminator = Discriminator(subject_dim, predicate_dim, object_dim)
#     optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    


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

# # def train_gan(num_epochs=1000, batch_size=64):
# #     df = factorized_data["df"]
# #     subjects = factorized_data["subjects"]
# #     predicates = factorized_data["predicates"]
# #     objects = factorized_data["objects"]
# #     subject_dim = factorized_data["subject_dim"]
# #     predicate_dim = factorized_data["predicate_dim"]
# #     object_dim = factorized_data["object_dim"]

# #     for epoch in range(num_epochs):
# #         for i in range(0, len(df), batch_size):
# #             s = torch.tensor(subjects[i:i+batch_size], dtype=torch.long)
# #             p = torch.tensor(predicates[i:i+batch_size], dtype=torch.long)
# #             o = torch.tensor(objects[i:i+batch_size], dtype=torch.long)

# #             s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
# #             p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()
# #             o_oh = torch.nn.functional.one_hot(o, num_classes=object_dim).float()

# #             real_labels = torch.ones(batch_size, 1)
# #             fake_labels = torch.zeros(batch_size, 1)

# #             optimizer_d.zero_grad()
# #             real_preds = discriminator(s_oh, p_oh, o_oh)
# #             d_loss_real = criterion(real_preds, real_labels)

# #             z = torch.randn(batch_size, z_dim)
# #             fake_objects = generator(z, s_oh, p_oh)
# #             fake_preds = discriminator(s_oh, p_oh, fake_objects)
# #             d_loss_fake = criterion(fake_preds, fake_labels)

# #             d_loss = d_loss_real + d_loss_fake
# #             d_loss.backward()
# #             optimizer_d.step()

# #             optimizer_g.zero_grad()
# #             fake_objects = generator(z, s_oh, p_oh)
# #             fake_preds = discriminator(s_oh, p_oh, fake_objects)
# #             g_loss = criterion(fake_preds, real_labels)
# #             g_loss.backward()
# #             optimizer_g.step()

# #         if epoch % 100 == 0:
# #             print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")


# def train_gan(num_epochs=1000, batch_size=64):
#     df = factorized_data["df"]  # Ensure this is using the global dictionary
#     subjects = factorized_data["subjects"]
#     predicates = factorized_data["predicates"]
#     objects = factorized_data["objects"]
#     subject_dim = factorized_data["subject_dim"]
#     predicate_dim = factorized_data["predicate_dim"]
#     object_dim = factorized_data["object_dim"]

#     for epoch in range(num_epochs):
#         for i in range(0, len(df), batch_size):
#             s = torch.tensor(subjects[i:i+batch_size], dtype=torch.long)
#             p = torch.tensor(predicates[i:i+batch_size], dtype=torch.long)
#             o = torch.tensor(objects[i:i+batch_size], dtype=torch.long)

#             s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
#             p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()
#             o_oh = torch.nn.functional.one_hot(o, num_classes=object_dim).float()

#             # real_labels = torch.ones(batch_size, 1)
#             # fake_labels = torch.zeros(batch_size, 1)

#             current_batch_size = s_oh.size(0)
#             real_labels = torch.ones(current_batch_size, 1)
#             fake_labels = torch.zeros(current_batch_size, 1)


#             # Train discriminator
#             optimizer_d.zero_grad()
#             real_preds = discriminator(s_oh, p_oh, o_oh)
#             d_loss_real = criterion(real_preds, real_labels)

#             z = torch.randn(batch_size, z_dim)
#             fake_objects = generator(z, s_oh, p_oh)
#             fake_preds = discriminator(s_oh, p_oh, fake_objects)
#             d_loss_fake = criterion(fake_preds, fake_labels)

#             d_loss = d_loss_real + d_loss_fake
#             d_loss.backward()
#             optimizer_d.step()

#             # Train generator
#             optimizer_g.zero_grad()
#             fake_objects = generator(z, s_oh, p_oh)
#             fake_preds = discriminator(s_oh, p_oh, fake_objects)
#             g_loss = criterion(fake_preds, real_labels)
#             g_loss.backward()
#             optimizer_g.step()

#         if epoch % 100 == 0:
#             print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")


# # def save_model(generator_path="generator.pth", discriminator_path="discriminator.pth"):
# #     torch.save(generator.state_dict(), generator_path)
# #     torch.save(discriminator.state_dict(), discriminator_path)
# #     print("Model saved.")

# # def load_model(generator_path="generator.pth", discriminator_path="discriminator.pth"):
# #     generator.load_state_dict(torch.load(generator_path))
# #     discriminator.load_state_dict(torch.load(discriminator_path))
# #     print("Model loaded.")

# # def save_model(model_name):
# #     torch.save(generator.state_dict(), f"models/{model_name}/generator.pth")
# #     torch.save(discriminator.state_dict(), f"models/{model_name}/discriminator.pth")

# def save_model(model_name):
#     print(f"Saving model to models/{model_name}")
#     torch.save(generator.state_dict(), f"models/{model_name}/generator.pth")
#     torch.save(discriminator.state_dict(), f"models/{model_name}/discriminator.pth")


# def load_model(model_name):
#     global generator, discriminator
#     generator.load_state_dict(torch.load(f"models/{model_name}/generator.pth"))
#     discriminator.load_state_dict(torch.load(f"models/{model_name}/discriminator.pth"))


# # def generate_synthetic_data(subject_input, predicate_input, num_samples=1):
# #     subject_dim = factorized_data["subject_dim"]
# #     predicate_dim = factorized_data["predicate_dim"]

# #     s = torch.tensor(subject_input, dtype=torch.long).unsqueeze(0)
# #     p = torch.tensor(predicate_input, dtype=torch.long).unsqueeze(0)
# #     s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
# #     p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()
# #     z = torch.randn(num_samples, z_dim)
# #     generated = generator(z, s_oh, p_oh).detach().numpy()
# #     return generated

# def generate_synthetic_data(subject_input, predicate_input, num_samples=1):
#     subject_dim = factorized_data["subject_dim"]
#     predicate_dim = factorized_data["predicate_dim"]

#     # Map subject and predicate input to their factorized indices
#     if subject_input not in factorized_data["subject_uniques"]:
#         raise ValueError(f"Subject '{subject_input}' not found in factorized data.")
#     if predicate_input not in factorized_data["predicate_uniques"]:
#         raise ValueError(f"Predicate '{predicate_input}' not found in factorized data.")
    
#     subject_idx = np.where(factorized_data["subject_uniques"] == subject_input)[0][0]
#     predicate_idx = np.where(factorized_data["predicate_uniques"] == predicate_input)[0][0]

#     # Convert input to one-hot encoded tensors
#     s = torch.tensor([subject_idx], dtype=torch.long)
#     p = torch.tensor([predicate_idx], dtype=torch.long)
    
#     s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
#     p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()
#     z = torch.randn(num_samples, z_dim)

#     generated = generator(z, s_oh, p_oh).detach().numpy()
#     # Convert one-hot-like vector into predicted index (argmax over object dimension)
#     generated_idx = np.argmax(generated, axis=1)

#     # Decode back to original object strings using the inverse map
#     decoded_objects = [factorized_data["object_inverse_map"].get(idx, "UNKNOWN") for idx in generated_idx]

#     return decoded_objects



# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import rdflib
# import numpy as np
# import pandas as pd


# factorized_data = {}
# generator = None
# discriminator = None
# optimizer_g = None
# optimizer_d = None
# criterion = nn.BCELoss()
# z_dim = 100

# def load_dbpedia_ttl(file_path):
#     g = rdflib.Graph()
#     g.parse(file_path, format='ttl')
#     return [(str(s), str(p), str(o)) for s, p, o in g]

# def factorize_and_initialize_gan(file_path):
#     global factorized_data, generator, discriminator, optimizer_g, optimizer_d

#     triples = load_dbpedia_ttl(file_path)
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

# # def train_gan(num_epochs=1000, batch_size=64):
# #     df = factorized_data["df"]
# #     subjects = factorized_data["subjects"]
# #     predicates = factorized_data["predicates"]
# #     objects = factorized_data["objects"]
# #     subject_dim = factorized_data["subject_dim"]
# #     predicate_dim = factorized_data["predicate_dim"]
# #     object_dim = factorized_data["object_dim"]

# #     for epoch in range(num_epochs):
# #         for i in range(0, len(df), batch_size):
# #             s = torch.tensor(subjects[i:i+batch_size], dtype=torch.long)
# #             p = torch.tensor(predicates[i:i+batch_size], dtype=torch.long)
# #             o = torch.tensor(objects[i:i+batch_size], dtype=torch.long)

# #             s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
# #             p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()
# #             o_oh = torch.nn.functional.one_hot(o, num_classes=object_dim).float()

# #             current_batch_size = s_oh.size(0)
# #             real_labels = torch.ones(current_batch_size, 1)
# #             fake_labels = torch.zeros(current_batch_size, 1)

# #             # Train discriminator
# #             optimizer_d.zero_grad()
# #             real_preds = discriminator(s_oh, p_oh, o_oh)
# #             d_loss_real = criterion(real_preds, real_labels)

# #             z = torch.randn(current_batch_size, z_dim)
# #             fake_objects = generator(z, s_oh, p_oh)
# #             fake_preds = discriminator(s_oh, p_oh, fake_objects)
# #             d_loss_fake = criterion(fake_preds, fake_labels)

# #             d_loss = d_loss_real + d_loss_fake
# #             d_loss.backward()
# #             optimizer_d.step()

# #             # Train generator
# #             optimizer_g.zero_grad()
# #             fake_objects = generator(z, s_oh, p_oh)
# #             fake_preds = discriminator(s_oh, p_oh, fake_objects)
# #             g_loss = criterion(fake_preds, real_labels)
# #             g_loss.backward()
# #             optimizer_g.step()

# #         if epoch % 100 == 0:
# #             print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# def train_gan(num_epochs=1000, batch_size=64):
#     df = factorized_data["df"]
#     subjects = factorized_data["subjects"]
#     predicates = factorized_data["predicates"]
#     objects = factorized_data["objects"]
#     subject_dim = factorized_data["subject_dim"]
#     predicate_dim = factorized_data["predicate_dim"]
#     object_dim = factorized_data["object_dim"]

#     # Calculate dynamic batch size based on the dataset size
#     total_samples = len(df)
#     dynamic_batch_size = min(batch_size, total_samples)  # Ensure batch size doesn't exceed total samples

#     for epoch in range(num_epochs):
#         for i in range(0, total_samples, dynamic_batch_size):
#             # Adjust indices dynamically
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

#             z = torch.randn(current_batch_size, z_dim)
#             fake_objects = generator(z, s_oh, p_oh)
#             fake_preds = discriminator(s_oh, p_oh, fake_objects)
#             d_loss_fake = criterion(fake_preds, fake_labels)

#             d_loss = d_loss_real + d_loss_fake
#             d_loss.backward()
#             optimizer_d.step()

#             # Train generator
#             optimizer_g.zero_grad()
#             fake_objects = generator(z, s_oh, p_oh)
#             fake_preds = discriminator(s_oh, p_oh, fake_objects)
#             g_loss = criterion(fake_preds, real_labels)
#             g_loss.backward()
#             optimizer_g.step()

#         if epoch % 100 == 0:
#             print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")



# def save_model(model_name):
#     os.makedirs(f"models/saved_models/{model_name}", exist_ok=True)
#     torch.save(generator.state_dict(), f"models/saved_models/{model_name}/generator.pth")
#     torch.save(discriminator.state_dict(), f"models/saved_models/{model_name}/discriminator.pth")

# # def load_model(model_name):
# #     global generator, discriminator
# #     generator.load_state_dict(torch.load(f"models/saved_models/{model_name}/generator.pth"))
# #     discriminator.load_state_dict(torch.load(f"models/saved_models/{model_name}/discriminator.pth"))

# # def load_all_models(models_root="/models/saved_models"):
# #     global generator, discriminator

# #     if not os.path.exists(models_root):
# #         print(f"Directory {models_root} does not exist.")
# #         return

# #     model_names = [d for d in os.listdir(models_root) if os.path.isdir(os.path.join(models_root, d))]

# #     for model_name in model_names:
# #         gen_path = os.path.join(models_root, model_name, "generator.pth")
# #         disc_path = os.path.join(models_root, model_name, "discriminator.pth")

# #         if os.path.exists(gen_path) and os.path.exists(disc_path):
# #             print(f"Loading model: {model_name}")

# #             # Re-initialize the models before loading (required if you load multiple models in a loop)
# #             generator = Generator(
# #                 factorized_data["subject_dim"],
# #                 factorized_data["predicate_dim"],
# #                 factorized_data["object_dim"],
# #                 z_dim
# #             )
# #             discriminator = Discriminator(
# #                 factorized_data["subject_dim"],
# #                 factorized_data["predicate_dim"],
# #                 factorized_data["object_dim"]
# #             )

# #             generator.load_state_dict(torch.load(gen_path, map_location=torch.device('cpu')))
# #             discriminator.load_state_dict(torch.load(disc_path, map_location=torch.device('cpu')))
# #         else:
# #             print(f"Missing generator/discriminator files in model {model_name}, skipping.")


# # def load_all_models(models_root="/app/models/saved_models"):
# #     global generator, discriminator

# #     if not os.path.exists(models_root):
# #         print(f"Directory {models_root} does not exist.")
# #         return

# #     model_names = [d for d in os.listdir(models_root) if os.path.isdir(os.path.join(models_root, d))]

# #     for model_name in model_names:
# #         gen_path = os.path.join(models_root, model_name, "generator.pth")
# #         disc_path = os.path.join(models_root, model_name, "discriminator.pth")

# #         if os.path.exists(gen_path) and os.path.exists(disc_path):
# #             print(f"Loading model: {model_name}")

# #             # Re-initialize the models before loading (required if you load multiple models in a loop)
# #             generator = Generator(
# #                 factorized_data["subject_dim"],
# #                 factorized_data["predicate_dim"],
# #                 factorized_data["object_dim"],
# #                 z_dim
# #             )
# #             discriminator = Discriminator(
# #                 factorized_data["subject_dim"],
# #                 factorized_data["predicate_dim"],
# #                 factorized_data["object_dim"]
# #             )

# #             generator.load_state_dict(torch.load(gen_path, map_location=torch.device('cpu')))
# #             discriminator.load_state_dict(torch.load(disc_path, map_location=torch.device('cpu')))
# #         else:
# #             print(f"Missing generator/discriminator files in model {model_name}, skipping.")

# # def load_model(model_name):
# #     global generator, discriminator

# #     # Load the generator model
# #     generator = Generator(
# #         factorized_data["subject_dim"], 
# #         factorized_data["predicate_dim"], 
# #         factorized_data["object_dim"], 
# #         z_dim
# #     )
# #     generator.load_state_dict(torch.load(f"models/saved_models/{model_name}/generator.pth"))
# #     generator.eval()  # Set to evaluation mode

# #     # Load the discriminator model
# #     discriminator = Discriminator(
# #         factorized_data["subject_dim"], 
# #         factorized_data["predicate_dim"], 
# #         factorized_data["object_dim"]
# #     )
# #     discriminator.load_state_dict(torch.load(f"models/saved_models/{model_name}/discriminator.pth"))
# #     discriminator.eval()  # Set to evaluation mode

# #     print(f"✅ Model '{model_name}' loaded successfully.")


# # Store models in a global dictionary
# loaded_models = {}

# def load_model(model_name):
#     # Initialize generator and discriminator
#     generator = Generator(
#         factorized_data["subject_dim"], 
#         factorized_data["predicate_dim"], 
#         factorized_data["object_dim"], 
#         z_dim
#     )
#     generator.load_state_dict(torch.load(f"models/saved_models/{model_name}/generator.pth"))
#     generator.eval()

#     discriminator = Discriminator(
#         factorized_data["subject_dim"], 
#         factorized_data["predicate_dim"], 
#         factorized_data["object_dim"]
#     )
#     discriminator.load_state_dict(torch.load(f"models/saved_models/{model_name}/discriminator.pth"))
#     discriminator.eval()

#     # Store the models in the dictionary
#     loaded_models[model_name] = {
#         "generator": generator,
#         "discriminator": discriminator
#     }

#     print(f"✅ Model '{model_name}' loaded successfully.")


# # def generate_synthetic_data(subject_input, predicate_input, num_samples=1):
# #     subject_dim = factorized_data["subject_dim"]
# #     predicate_dim = factorized_data["predicate_dim"]

# #     if subject_input not in factorized_data["subject_uniques"]:
# #         raise ValueError(f"Subject '{subject_input}' not found.")
# #     if predicate_input not in factorized_data["predicate_uniques"]:
# #         raise ValueError(f"Predicate '{predicate_input}' not found.")
    
# #     subject_idx = np.where(factorized_data["subject_uniques"] == subject_input)[0][0]
# #     predicate_idx = np.where(factorized_data["predicate_uniques"] == predicate_input)[0][0]

# #     s = torch.tensor([subject_idx], dtype=torch.long)
# #     p = torch.tensor([predicate_idx], dtype=torch.long)
# #     s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
# #     p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()

# #     z = torch.randn(num_samples, z_dim)
# #     generated = generator(z, s_oh.repeat(num_samples, 1), p_oh.repeat(num_samples, 1)).detach().numpy()
# #     generated_idx = np.argmax(generated, axis=1)
# #     decoded_objects = [factorized_data["object_inverse_map"].get(idx, "UNKNOWN") for idx in generated_idx]
# #     return decoded_objects


# # def generate_synthetic_data(subject_input, predicate_input, num_samples=1):
# #     subject_dim = factorized_data["subject_dim"]
# #     predicate_dim = factorized_data["predicate_dim"]

# #     # Check if the input is a full URI or a part of it (like Nepali_language, spokenIn)
# #     subject_matches = [s for s in factorized_data["subject_uniques"] if subject_input in s]
# #     predicate_matches = [p for p in factorized_data["predicate_uniques"] if predicate_input in p]

# #     if len(subject_matches) == 0:
# #         raise ValueError(f"Subject '{subject_input}' not found.")
# #     if len(predicate_matches) == 0:
# #         raise ValueError(f"Predicate '{predicate_input}' not found.")

# #     # If there are multiple matches, you can decide how to handle them (e.g., choose the first match)
# #     # In this case, I am choosing the first match
# #     subject_input = subject_matches[0]
# #     predicate_input = predicate_matches[0]

# #     subject_idx = np.where(factorized_data["subject_uniques"] == subject_input)[0][0]
# #     predicate_idx = np.where(factorized_data["predicate_uniques"] == predicate_input)[0][0]

# #     s = torch.tensor([subject_idx], dtype=torch.long)
# #     p = torch.tensor([predicate_idx], dtype=torch.long)
# #     s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
# #     p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()

# #     z = torch.randn(num_samples, z_dim)
# #     generated = generator(z, s_oh.repeat(num_samples, 1), p_oh.repeat(num_samples, 1)).detach().numpy()
# #     generated_idx = np.argmax(generated, axis=1)
# #     decoded_objects = [factorized_data["object_inverse_map"].get(idx, "UNKNOWN") for idx in generated_idx]
# #     return decoded_objects


# # def generate_synthetic_data(subject_input, predicate_input, num_samples=1):
# #     subject_dim = factorized_data["subject_dim"]
# #     predicate_dim = factorized_data["predicate_dim"]

# #     # Convert input to lowercase for case-insensitive comparison
# #     subject_input_lower = subject_input.lower()
# #     predicate_input_lower = predicate_input.lower()

# #     # Check if the input is a full URI or a part of it (like Nepali_language, spokenIn)
# #     subject_matches = [s for s in factorized_data["subject_uniques"] if subject_input_lower in s.lower()]
# #     predicate_matches = [p for p in factorized_data["predicate_uniques"] if predicate_input_lower in p.lower()]

# #     if len(subject_matches) == 0:
# #         raise ValueError(f"Subject '{subject_input}' not found.")
# #     if len(predicate_matches) == 0:
# #         raise ValueError(f"Predicate '{predicate_input}' not found.")

# #     # If there are multiple matches, you can decide how to handle them (e.g., choose the first match)
# #     # In this case, I am choosing the first match
# #     subject_input = subject_matches[0]
# #     predicate_input = predicate_matches[0]

# #     subject_idx = np.where(factorized_data["subject_uniques"] == subject_input)[0][0]
# #     predicate_idx = np.where(factorized_data["predicate_uniques"] == predicate_input)[0][0]

# #     s = torch.tensor([subject_idx], dtype=torch.long)
# #     p = torch.tensor([predicate_idx], dtype=torch.long)
# #     s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
# #     p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()

# #     z = torch.randn(num_samples, z_dim)
# #     generated = generator(z, s_oh.repeat(num_samples, 1), p_oh.repeat(num_samples, 1)).detach().numpy()
# #     generated_idx = np.argmax(generated, axis=1)
# #     decoded_objects = [factorized_data["object_inverse_map"].get(idx, "UNKNOWN") for idx in generated_idx]
# #     return decoded_objects


# def generate_synthetic_data(subject_input, predicate_input, num_samples=1):
#     # Ensure generator is loaded
#     if generator is None:
#         raise RuntimeError("Generator model is not loaded.")
    
#     subject_dim = factorized_data["subject_dim"]
#     predicate_dim = factorized_data["predicate_dim"]

#     # Convert input to lowercase for case-insensitive comparison
#     subject_input_lower = subject_input.lower()
#     predicate_input_lower = predicate_input.lower()

#     # Check if the input is a full URI or a part of it (like Nepali_language, spokenIn)
#     subject_matches = [s for s in factorized_data["subject_uniques"] if subject_input_lower in s.lower()]
#     predicate_matches = [p for p in factorized_data["predicate_uniques"] if predicate_input_lower in p.lower()]

#     if len(subject_matches) == 0:
#         raise ValueError(f"Subject '{subject_input}' not found.")
#     if len(predicate_matches) == 0:
#         raise ValueError(f"Predicate '{predicate_input}' not found.")

#     # If there are multiple matches, you can decide how to handle them (e.g., choose the first match)
#     # In this case, I am choosing the first match
#     subject_input = subject_matches[0]
#     predicate_input = predicate_matches[0]

#     subject_idx = np.where(factorized_data["subject_uniques"] == subject_input)[0][0]
#     predicate_idx = np.where(factorized_data["predicate_uniques"] == predicate_input)[0][0]

#     s = torch.tensor([subject_idx], dtype=torch.long)
#     p = torch.tensor([predicate_idx], dtype=torch.long)
#     s_oh = torch.nn.functional.one_hot(s, num_classes=subject_dim).float()
#     p_oh = torch.nn.functional.one_hot(p, num_classes=predicate_dim).float()

#     # Generate synthetic data using the pre-loaded generator
#     z = torch.randn(num_samples, z_dim)
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


factorized_data = {}
generator = None
discriminator = None
optimizer_g = None
optimizer_d = None
criterion = nn.BCELoss()
z_dim = 100

def load_dbpedia_ttl(file_path):
    g = rdflib.Graph()
    g.parse(file_path, format='ttl')
    return [(str(s), str(p), str(o)) for s, p, o in g]

# def factorize_and_initialize_gan(file_path):
#     global factorized_data, generator, discriminator, optimizer_g, optimizer_d

#     triples = load_dbpedia_ttl(file_path)
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

# def factorize_and_initialize_gan(file_path):
#     triples = load_dbpedia_ttl(file_path)
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



def factorize_and_initialize_gan(file_path):
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

    generator = Generator(subject_dim, predicate_dim, object_dim, z_dim)
    discriminator = Discriminator(subject_dim, predicate_dim, object_dim)
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    return factorized_data, generator, discriminator, optimizer_g, optimizer_d



def factorize_and_initialize_gans(file_path):
    global factorized_data, generator, discriminator, optimizer_g, optimizer_d

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

def sample_noise(batch_size, z_dim, distribution="normal"):
    if distribution == "uniform":
        return torch.rand(batch_size, z_dim) * 2 - 1  # Range [-1, 1]
    elif distribution == "skewed":
        return torch.randn(batch_size, z_dim) ** 3
    elif distribution == "categorical":
        return torch.nn.functional.one_hot(torch.randint(0, z_dim, (batch_size,)), num_classes=z_dim).float()
    else:  # Default to normal
        return torch.randn(batch_size, z_dim)

def train_gan(num_epochs=1000, batch_size=64, distribution="normal"):
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

            z = sample_noise(current_batch_size, z_dim, distribution)
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

# def load_model(model_name):
#     global generator, discriminator

#     generator = Generator(
#         factorized_data["subject_dim"], 
#         factorized_data["predicate_dim"], 
#         factorized_data["object_dim"], 
#         z_dim
#     )
#     generator.load_state_dict(torch.load(f"models/saved_models/{model_name}/generator.pth"))
#     generator.eval()

#     discriminator = Discriminator(
#         factorized_data["subject_dim"], 
#         factorized_data["predicate_dim"], 
#         factorized_data["object_dim"]
#     )
#     discriminator.load_state_dict(torch.load(f"models/saved_models/{model_name}/discriminator.pth"))
#     discriminator.eval()

#     loaded_models[model_name] = {
#         "generator": generator,
#         "discriminator": discriminator
#     }

#     print(f"✅ Model '{model_name}' loaded successfully.")

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


# def generate_synthetic_data(model_name, subject_input, predicate_input, num_samples=1, distribution="normal"):

#     if model_name not in loaded_models:
#         raise RuntimeError(f"Model '{model_name}' is not loaded.")
    
#     model = loaded_models[model_name]
#     generator = model["generator"]
#     factorized_data = model["factorized_data"]

#     if generator is None:
#         raise RuntimeError("Generator model is not loaded.")
    
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

#     z = sample_noise(num_samples, z_dim, distribution)
#     generated = generator(z, s_oh.repeat(num_samples, 1), p_oh.repeat(num_samples, 1)).detach().numpy()
#     generated_idx = np.argmax(generated, axis=1)
#     decoded_objects = [factorized_data["object_inverse_map"].get(idx, "UNKNOWN") for idx in generated_idx]
#     return decoded_objects


def generate_synthetic_data(model_name, subject_input, predicate_input, num_samples=1, distribution="normal"):
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

                z = sample_noise(num_samples, z_dim, distribution)
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

