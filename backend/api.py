import sys
from pathlib import Path

# -------------------------------------------------
# Fix: ensure project root is on PYTHONPATH
# (needed for uvicorn reload on Windows)
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import torch
import torch.nn as nn
from torchvision import models

# -------------------------
# Import Agents
# -------------------------
# from backend.agents.orchestrator import Orchestrator
from backend.langgraph_agents.workflow import agentic_tumor_app
from backend.utils.helpers import save_temp_file, cleanup_file
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, FileResponse
from backend.agents.report_generator import generate_pdf

# -------------------------
# Paths (match your backend structure)
# -------------------------
BACKEND_DIR = Path(__file__).resolve().parent  # backend/

MODEL_PATH = BACKEND_DIR / "trained_models" / "tumor_model.pt"
TEMPLATES_DIR = BACKEND_DIR / "templates"
STATIC_DIR = BACKEND_DIR / "static"

# -------------------------
# Load Trained Model
# -------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"❌ {MODEL_PATH} not found. Run train_model.py first."
    )

print("🧠 Loading trained model...")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)

model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))
model.eval()

print("✅ Model loaded")

# -------------------------
# Orchestrator (Replaced by LangGraph)
# -------------------------
# orch = Orchestrator(model)

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(
    title="Agentic AI Tumor Detection API",
    version="1.0"
)

# allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # ok for local dev; restrict when deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Templates + Static
# -------------------------
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Serve files from backend/static at /static/*
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# -------------------------
# Routes
# -------------------------

# ✅ Connected welcome page
@app.get("/", response_class=HTMLResponse)
async def welcome(request: Request):
    """
    Serves backend/templates/welcome.html
    Background image should be in backend/static/bg.jpeg
    Use in CSS: background: url('/static/bg.jpeg')
    """
    return templates.TemplateResponse("welcome.html", {"request": request})


# ✅ Your existing analyze endpoint
@app.post("/analyze")
async def analyze_tumor(
    file: UploadFile = File(...),
    clinical_text: str = Form(""),
    patient_age: int = Form(None),
    patient_gender: str = Form(None)
):
    """
    Upload MRI image + optional clinical notes.
    Triggers the LangGraph multi-agent workflow.
    """
    contents = await file.read()
    temp_path = save_temp_file(contents, ".jpg")

    try:
        # Initialize state for LangGraph
        initial_state = {
            "image_path": temp_path,
            "clinical_text": clinical_text,
            "patient_age": patient_age,
            "patient_gender": patient_gender
        }

        # Run the workflow
        result_state = agentic_tumor_app.invoke(initial_state)

        return {
            "prediction": result_state.get("tumor_type"),
            "resnet_confidence": result_state.get("resnet_confidence"),
            "xai_confidence": result_state.get("xai_confidence"),
            "clinical_input": clinical_text,
            "agent_reasoning": result_state.get("rag_context"),
            "tumor_size_cm": result_state.get("tumor_size_cm"),
            "probable_diagnosis": result_state.get("probable_diagnosis"),
            "treatment_recommendations": result_state.get("treatment_recommendations"),
            "final_report": result_state.get("final_report"),
            "gradcam_image": result_state.get("gradcam_path"),
            "segmentation_mask": result_state.get("segmentation_mask_path")
        }

    finally:
        # cleanup_file(temp_path) # Keep it for UI demo
        pass

class ChatRequest(BaseModel):
    message: str
    patient_context: dict = None

@app.post("/chat")
async def clinical_chat(request: ChatRequest):
    """
    Conversational Assistant Endpoint
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage
        from backend.rag.vector_store import query_guidelines
        
        # 1. Retrieve Medical Guidelines from ChromaDB
        docs = query_guidelines(request.message, k=2)
        guidelines_text = "\n".join([doc.page_content for doc in docs]) if docs else "No specific guidelines found."

        # 2. Extract Patient Context
        ctx = request.patient_context or {}
        patient_info = f"""
        Age: {ctx.get('age', 'Unknown')}
        Gender: {ctx.get('gender', 'Unknown')}
        Tumor Prediction: {ctx.get('prediction', 'Unknown')}
        Tumor Size: {ctx.get('tumor_size', 'Unknown')} cm
        Clinical Notes: {ctx.get('notes', 'None')}
        """

        # 3. Initialize Llama3
        llm = ChatOllama(model="llama3", temperature=0.2)
        
        system_prompt = f"""You are an expert AI clinical assistant for a radiologist. 
        You must answer the radiologist's query based strictly on the provided patient context and medical guidelines.
        
        [PATIENT CONTEXT]
        {patient_info}
        
        [RELEVANT MEDICAL GUIDELINES (NCCN/WHO)]
        {guidelines_text}
        
        Provide a concise, professional, and evidence-based response.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.message)
        ]
        
        response = llm.invoke(messages)
        return {"reply": response.content}
        
    except Exception as e:
        print(f"LLM Error: {e}")
        ctx = request.patient_context or {}
        tumor = ctx.get('prediction') or 'the tumor'
        
        fallback = (
            f"I am currently operating in offline mode (Llama3 engine unreachable). "
            f"However, regarding the {tumor}, standard NCCN/WHO guidelines generally suggest "
            f"maximal safe resection followed by histopathological analysis. "
            f"Please ensure `ollama run llama3` is active in your terminal for full dynamic reasoning."
        )
        return {"reply": fallback}

@app.get("/cases/similar")
async def similar_cases(diagnosis: str):
    """
    Similar Case Retrieval Endpoint
    """
    from backend.database import get_similar_cases
    cases = get_similar_cases(diagnosis)
    if not cases:
        # Mock some cases if DB is empty
        cases = [
            {"id": "C1", "probable_diagnosis": "Glioma", "outcome": "Successful Resection"},
            {"id": "C2", "probable_diagnosis": "Glioma", "outcome": "Radiotherapy Ongoing"}
        ]
    return {"similar_cases": cases}

class ReportRequest(BaseModel):
    patient: dict
    analysis: dict
    img_url: str = None

@app.post("/generate-report")
async def create_report(request: ReportRequest):
    """
    Generates a PDF report and returns it for download.
    """
    img_path = None
    if request.img_url and "gradcam_output.jpg" in request.img_url:
        img_path = str(Path(__file__).resolve().parent / "static" / "gradcam_output.jpg")
    
    pdf_path = generate_pdf(request.patient, request.analysis, img_path)
    
    return FileResponse(
        path=pdf_path,
        filename="Radiology_Report.pdf",
        media_type="application/pdf"
    )