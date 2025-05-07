
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
```

This will:

- Build the FastAPI and Streamlit services from the local Dockerfile
- Mount `./models/saved_models` and `./uploaded` into the backend container
- Set up MongoDB with a local volume `mongo-data`

🌐 **Access the App**
- Backend (FastAPI): [http://localhost:8000](http://localhost:8000)
- Frontend (Streamlit): [http://localhost:8501](http://localhost:8501)

Use Visual Studio Code to:

- Open and edit the codebase easily
- Launch Docker containers with the Docker extension (optional but helpful)

# 🚀 How to Use the App

1. **Open the Streamlit frontend** in your browser at [http://localhost:8501](http://localhost:8501).
2. **Upload your SHACL .ttl file**:
   - On the left-hand side of the app, in the **Settings** section, you will find an option to **Upload SHACL File**.
   - Upload the desired SHACL .ttl file to the app.
3. **Tree View of Target Classes and Properties**:
   - Once the SHACL file is uploaded, the app will display a tree-based representation of the available target classes and their associated properties.
4. **Configure Property Settings**:
   - For each property, you can define the model to use:
     - By default, the model is set to **LLM**.
     - If you want to choose a different model (such as **GAN** or **VAE**), a **dropdown list** will appear where you can select the desired model.
     - The list will also include models you pre-trained in the deployment step.
     - If you're unsure which model to use, you can select the **All** option, which assigns the closest model to the property.
5. **Choose Data Distribution**:
   - You can select the **data distribution** type for each property. Available options include:
     - **Uniform**
     - **Normal**
     - **Skewed**
   - Then, input the **number of samples** you want to generate. 
     - If you have a **premium GPT account**, the number of samples can be unlimited. 
     - For free accounts, there is a limitation on the number of samples.
6. **Generate Synthetic Data**:
   - After configuring the settings for each property, click on the **Generate synthetic data (batch request)** button.
   - The app will start generating the RDF data and show the **progress** of the data generation process.
7. **Preview the Generated Data**:
   - Once the data is generated, you will see a **preview** of the synthetic RDF data.
8. **Interactive Chat for Model Refinement**:
   - If you're not satisfied with the results, use the **interactive chat box** on the left-hand side to interact with the LLM model.
   - Provide instructions to improve the data generation, and regenerate the data based on your new instructions.
9. **Download the Generated Data**:
   - Once you're happy with the generated data, you can **download** it in one of the following formats:
     - `.json`
     - `.json-ld`
     - `.ttl`

## 💻 How to Use the Models

### LLM (Large Language Model)

To use the LLM, you need to provide an API key from a GPT provider. For example, you can create an account on Groq Console and generate an API key from Groq API Keys.

- Create an account and get your API key.
- Input the API key in the .env file located in the project directory.

> **Note**: To unlock the full capabilities of GPT and generate unlimited data, you need a premium account. However, the free tier is sufficient for testing, though there may be limitations on the number of samples generated.

### GAN and VAE Models

The GAN and VAE models are domain-independent. Pretrained models are already available in the repository for testing purposes.

In case you want to extend or customize the app for your research domain or production environment, you can generate more pretrained models. After running the app, you need to access the backend APIs at [http://localhost:8000/docs](http://localhost:8000/docs) and use the `/upload_and_train_gan/` and `/upload_and_train_vae/` endpoints to train and save the models in the repository.

Based on the domain of SHACL properties you want to generate data for, you can download domain-specific triples from these sources:

- [DBpedia Latest Core](https://databus.dbpedia.org/dbpedia/collections/latest-core)
- [Wikidata Database Dump](https://www.wikidata.org/wiki/Wikidata:Database_download)

Once you've downloaded the `.ttl` files, upload them to the app for training the models:

- For VAE, open the following URL in your browser: [http://localhost:8000/docs#/default/upload_and_train_vae_upload_and_train_vae_post](http://localhost:8000/docs#/default/upload_and_train_vae_upload_and_train_vae_post)
  - Click on the **Try it out** button on the top right.
  - In the file section, choose the `.ttl` file you downloaded.
  - In the **epochs** section, define the number of triples you want the model to train on.
  - In the **model_name** section, choose a name for your specific model.
  - Click **Execute**. The app will start training the model. If the model is trained and stored successfully, you will see a notification in the **Responses** section.

The same approach can be used for the GAN model at this URL: [http://localhost:8000/docs#/default/upload_ttl_upload_and_train_gan__post](http://localhost:8000/docs#/default/upload_ttl_upload_and_train_gan__post)

Once the models are trained and saved, restart the Docker containers. The models will be automatically loaded in the next app run.

> In the next section, you will find an explanation of how to use the pretrained models.
