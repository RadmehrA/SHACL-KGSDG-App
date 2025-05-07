
# 🧪 Synthetic RDF Data Generator (SRDF-GEN)

An application designed for generating synthetic RDF data based on SHACL schemas and W3C standards. The system supports three powerful generative models—**LLM**, **GAN**, and **VAE**—to create high-quality RDF triples based on user-defined shapes and distributions.

## 🌐 Key Features

- **SHACL File Upload**: Upload SHACL `.ttl` files to define your data schema.
- **Tree-based Visualization**: Visualize the target classes and properties of your schema in a tree structure.
- **Per-property Customization**:
  - Choose a generative model: `LLM`, `GAN`, or `VAE`.
  - Select data distribution type.
  - Set the number of samples to generate.
- **Data Download**: Download the generated data in multiple formats:
  - `.ttl` (Turtle)
  - `.json`
  - `.json-ld`
- **Pretrained Models**: Uses pretrained GAN and VAE models trained on [DBpedia Core Triples](https://databus.dbpedia.org/dbpedia/collections/latest-core).
- **No Training Data Required**: No need to provide training data, the models are pre-trained.
- **Extensibility**: Easily extendable to support other knowledge bases like Wikidata.

## 🛠️ Technologies Used

- **FastAPI**: Backend API
- **Streamlit**: Web UI for interactive input and output
- **PyTorch / TensorFlow**: For VAE and GAN model inference
- **SPARQLWrapper**, **rdflib**, **PySHACL**: For RDF manipulation and validation
- **Docker**: Containerized deployment
- **SHACL**: Schema constraint definitions
- **GPT API**: Used by the LLM model for generating RDF triples

## 🚀 How to Deploy

You can deploy the app locally using **Docker Compose**. This will start the following services:

- **FastAPI backend**
- **Streamlit frontend**
- **MongoDB database**

### ✅ Prerequisites

Make sure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Visual Studio Code](https://code.visualstudio.com/) (recommended for easy setup and file navigation)

> **Important:** Ensure MongoDB and Docker are running before building the project.

### 📦 Steps

```bash
# Clone the repository
git clone https://github.com/RadmehrA/SHACL-KGSDG-App.git
cd SHACL-KGSDG-App

# Start MongoDB and all services via Docker
docker-compose up --build
```

This will:

- Build the FastAPI and Streamlit services from the local Dockerfile.
- Mount `./models/saved_models` and `./uploaded` directories into the backend container.
- Set up MongoDB with a local volume (`mongo-data`).

### 🌐 Access the App
- Backend (FastAPI): [http://localhost:8000](http://localhost:8000)
- Frontend (Streamlit): [http://localhost:8501](http://localhost:8501)

You can use **Visual Studio Code** to:

- Open and edit the codebase easily.
- Launch Docker containers with the Docker extension (optional but helpful).

## 💻 How to Use the Models

### LLM (Large Language Model)

To use the LLM model, you need an API key from a GPT provider, such as Groq.

1. Create an account on Groq and obtain an API key.
2. Input the API key into the `.env` file located in the project directory.

> **Note**: You need a premium GPT account to unlock full capabilities and generate unlimited data. The free tier is suitable for testing, though it may have sample generation limitations.

### GAN and VAE Models

The GAN and VAE models are domain-independent and come with pretrained models for general use. To train the models on domain-specific data:

1. **Download Domain-Specific Triples**:
   - [DBpedia Latest Core](https://databus.dbpedia.org/dbpedia/collections/latest-core)
   - [Wikidata Database Dump](https://www.wikidata.org/wiki/Wikidata:Database_download)

2. **Upload and Train the Models**:
   - Access the backend API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).
   - Use the `/upload_and_train_gan/` and `/upload_and_train_vae/` endpoints to upload training data (`.ttl` files) and train the models.

> **Important**: Once training is complete, restart the Docker containers. The models will be automatically loaded in the next app run.

## 🔧 Local Development Setup

If you prefer running the app locally without Docker, follow these steps:

### 1. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Backend

```bash
uvicorn main:app --reload
```

### 4. Run the Frontend

```bash
streamlit run frontend/app.py
```

## 🚀 How to Use the App

1. **Open the Streamlit frontend**: Access it via [http://localhost:8501](http://localhost:8501).
   
2. **Upload SHACL `.ttl` File**:
   - In the **Settings** section, find the option to **Upload SHACL File**.
   - Select and upload your SHACL `.ttl` file.

3. **Tree View of Target Classes and Properties**:
   - The app will display a tree view of the target classes and their associated properties.

4. **Configure Property Settings**:
   - For each property, choose the generative model:
     - **LLM**, **GAN**, or **VAE** (with pre-trained options available).
     - Select a data distribution type: **Uniform**, **Normal**, or **Skewed**.
     - Set the **number of samples** you want to generate.

5. **Generate Synthetic Data**:
   - After configuring the settings, click **Generate Synthetic Data (Batch Request)**.
   - The app will show the progress of the data generation.

6. **Preview the Generated Data**:
   - Once generated, you can preview the synthetic RDF data.

7. **Interactive Chat for Refining Models**:
   - If necessary, use the **interactive chat box** to refine the model and regenerate data based on new instructions.

8. **Download the Generated Data**:
   - Download the generated data in one of the following formats:
     - `.ttl`
     - `.json`
     - `.json-ld`
