# 🧠 Agentic AI Tumor Classification System

An advanced, multi-agent AI medical assistant for the autonomous detection, diagnosis, and treatment recommendation of brain tumors. Built with **LangGraph**, **PyTorch**, and **Llama 3**, this system provides explainable clinical insights with real-time RAG (Retrieval-Augmented Generation) medical guideline retrieval.

![Dashboard Preview](frontend/src/assets/hero.png)

## 🚀 Features

- **Agentic Multi-Step Workflow**: Utilizes a LangGraph architecture to orchestrate multiple specialized AI agents (Image Agent, XAI Agent, Segmentation Agent, RAG Agent, Diagnosis Agent, and Treatment Agent).
- **Accurate Classification**: Powered by a custom-trained **ResNet-18** CNN to detect Glioma, Meningioma, Pituitary tumors, and Healthy Brains.
- **Explainable AI (XAI)**: Generates highly robust **Grad-CAM Attention Heatmaps** to visualize the precise focal point of the AI's diagnosis.
- **Autonomous RAG Assistant**: Retrieves offline clinical guidelines (NCCN/WHO) via **ChromaDB** and queries a local **Llama 3** engine (via Ollama) to dynamically recommend treatments tailored to the tumor type and patient age/gender.
- **Smart Routing**: Bypasses intensive XAI, RAG, and Treatment models if a healthy "No Tumor" scan is detected, outputting a completely clean clinical report and "Low" risk level.
- **Clinical Reporting**: Autonomously compiles the diagnostic workflow into a downloadable PDF Radiology Report for the attending physician.

## 🛠️ Technology Stack

- **Frontend**: React, Vite, Lucide Icons, Axios
- **Backend API**: FastAPI, Uvicorn, Python
- **AI / Deep Learning**: PyTorch, Torchvision, OpenCV
- **Agentic AI**: LangGraph, LangChain
- **Vector Database**: ChromaDB (Medical Guidelines)
- **Local LLM Engine**: Ollama (Llama 3)

## ⚙️ Local Setup Instructions

### 1. Backend Setup

First, navigate to the project directory and activate your virtual environment:
```bash
# Navigate to the backend or root folder
source .venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Start the FastAPI Server
cd backend
uvicorn api:app --reload --port 8000
```

### 2. Frontend Setup

In a new terminal window, start the React application:
```bash
cd frontend
npm install
npm run dev
```

### 3. Local Llama 3 Engine (Critical for Chat Assistant)

The clinical chat and autonomous treatment generator require a local LLM to preserve patient data privacy.
1. Install [Ollama](https://ollama.com/)
2. Run the following command in your terminal to download and start the model:
```bash
ollama run llama3
```

## 🏥 Clinical Workflow

1. **Upload MRI**: Provide a T1-weighted contrast MRI and optional clinical notes.
2. **Analysis**: The LangGraph state machine passes the image to the Vision model.
3. **XAI Extraction**: If a tumor is detected, the XAI agent extracts the top 15% attention map.
4. **Treatment**: The RAG agent fetches treatment guidelines, and Llama 3 generates personalized next steps.
5. **Review & Chat**: The doctor can review the PDF report or ask the Chat Assistant for reasoning.

---
*Disclaimer: This is a research project and should not be used as a sole diagnostic tool in real clinical environments without human supervision.*
