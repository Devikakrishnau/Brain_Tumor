from langgraph.graph import StateGraph, END
from backend.langgraph_agents.state import AgenticTumorState
from backend.langgraph_agents.nodes import (
    image_agent,
    segmentation_agent,
    xai_agent,
    rag_medical_agent,
    diagnosis_agent,
    treatment_agent,
    report_agent
)

def build_graph():
    # 1. Initialize Graph
    workflow = StateGraph(AgenticTumorState)

    # 2. Add Nodes
    workflow.add_node("image_agent", image_agent)
    workflow.add_node("segmentation_agent", segmentation_agent)
    workflow.add_node("xai_agent", xai_agent)
    workflow.add_node("rag_medical_agent", rag_medical_agent)
    workflow.add_node("diagnosis_agent", diagnosis_agent)
    workflow.add_node("treatment_agent", treatment_agent)
    workflow.add_node("report_agent", report_agent)

    # 3. Define Edges (Sequential for now, can be parallelized)
    workflow.set_entry_point("image_agent")
    
    workflow.add_edge("image_agent", "xai_agent")
    workflow.add_edge("xai_agent", "segmentation_agent")
    workflow.add_edge("segmentation_agent", "rag_medical_agent")
    workflow.add_edge("rag_medical_agent", "diagnosis_agent")
    workflow.add_edge("diagnosis_agent", "treatment_agent")
    workflow.add_edge("treatment_agent", "report_agent")
    workflow.add_edge("report_agent", END)

    # 4. Compile
    app = workflow.compile()
    return app

# Expose a compiled instance
agentic_tumor_app = build_graph()
