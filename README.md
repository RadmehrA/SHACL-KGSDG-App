# 🧪 Synthetic RDF Data Generator (SRDF-GEN)

An application for generating synthetic RDF data using SHACL schemas and W3C standards. 
The system supports three generative models—**LLM**, **GAN**, and **VAE**—to produce high-quality RDF triples based on user-defined shapes and distributions.

## 🌐 Key Features

- Upload SHACL `.ttl` files to define your data schema
- Tree-based visualization of target classes and properties
- Per-property selection of:
  - Generative model: `LLM`, `GAN`, or `VAE`
  - Data distribution
  - Number of samples to generate
- Download generated data in:
  - `.ttl` (Turtle)
  - `.json`
  - `.json-ld`
- Uses **pretrained GAN/VAE models** trained on [DBpedia Core Triples](https://databus.dbpedia.org/dbpedia/collections/latest-core)
- No training data required from the user
- Easily extensible to support other knowledge bases like Wikidata

## 🛠️ Technologies Used

- **FastAPI** – Backend API
- **Streamlit** – Web UI for interactive input and output
- **PyTorch / TensorFlow** – For VAE and GAN model inference
- **SPARQLWrapper**, **rdflib**, **PySHACL** – For RDF manipulation and validation
- **Docker** – Containerized deployment
- **SHACL** – Schema constraint definitions
- **GPT API** – Used by the LLM model for generating RDF triples
