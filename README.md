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


# 🚀 How to Deploy

You can deploy the app locally using **Docker Compose**, which will spin up:

- **FastAPI backend**
- **Streamlit frontend**
- **MongoDB database**

## ✅ Prerequisites

Make sure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Visual Studio Code](https://code.visualstudio.com/) (recommended for easy setup and file navigation)

> **Important:** MongoDB and Docker must be running before building the project.

## 📦 Steps

```bash
# Clone the repository
git clone https://github.com/RadmehrA/SHACL-KGSDG-App.git
cd SHACL-KGSDG-App

# Start MongoDB and all services via Docker
docker-compose up --build

This will:

Build the FastAPI and Streamlit services from the local Dockerfile

Mount ./models/saved_models and ./uploaded into the backend container

Set up MongoDB with a local volume mongo-data

🌐 Access the App
Backend (FastAPI): http://localhost:8000

Frontend (Streamlit): http://localhost:8501

Use Visual Studio Code to:

Open and edit the codebase easily

Launch Docker containers with the Docker extension (optional but helpful)

