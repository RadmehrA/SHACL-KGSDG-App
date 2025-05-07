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

💻 How to Use the Models
LLM (Large Language Model)
To use the LLM, you need to provide an API key from a GPT provider. For example, you can create an account on Groq Console and generate an API key from Groq API Keys.

Create an account and get your API key.

Input the API key in the .env file located in the project directory.

Note: To unlock the full capabilities of GPT and generate unlimited data, you need a premium account. However, the free tier is sufficient for testing, though there may be limitations on the number of samples generated.

GAN and VAE Models
The GAN and VAE models are domain-independent. Pretrained models are already available in the repository for testing purposes.

To extend or customize the app for your research domain or production environment:

Access the backend API documentation by navigating to http://localhost:8000/docs.

Use the /upload_and_train_gan/ and /upload_and_train_vae/ endpoints to train and save models in the repository.

Note: You can download domain-specific triples for more accurate models from these sources:

DBpedia Latest Core

Wikidata Database Dump

After downloading the .ttl files, upload them to the app for training the models:

For VAE:

Open the following URL in your browser:
http://localhost:8000/docs#/default/upload_and_train_vae_upload_and_train_vae_post

Click Try it out in the top right.

In the file section, select the .ttl file.

Specify the number of triples to train on (epochs).

Provide a name for your model in the model_name section.

Click Execute to start training.

The same approach can be followed for the GAN model at this URL:
http://localhost:8000/docs#/default/upload_ttl_upload_and_train_gan__post

Once training is complete, restart the Docker containers. The models will be automatically loaded in the next app run.

🔧 How to Prepare the App (Local Development)
If you want to run the app locally without Docker:

Create a virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the backend:

bash
Copy
Edit
uvicorn main:app --reload
Run the frontend:

bash
Copy
Edit
streamlit run frontend/app.py
This guide should help users get the app running locally or in a Docker environment, and also provide details on how to use the models for synthetic data generation.

vbnet
Copy
Edit

This section provides the complete steps for deploying the app, as well as detailed instructions for using the LLM, GAN, and VAE models. Let me know if you need further updates!









