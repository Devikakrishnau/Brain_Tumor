from typing import TypedDict, Optional, List, Dict, Any

class AgenticTumorState(TypedDict):
    # Inputs
    image_path: str
    clinical_text: str
    patient_age: Optional[int]
    patient_gender: Optional[str]

    # Intermediates
    tumor_type: Optional[str]
    resnet_confidence: Optional[float]
    
    segmentation_mask_path: Optional[str]
    tumor_size_cm: Optional[float]
    tumor_volume_cc: Optional[float]
    
    gradcam_path: Optional[str]
    xai_confidence: Optional[float]
    similar_cases: Optional[List[Dict[str, Any]]]
    
    rag_context: Optional[str]
    guidelines_found: Optional[List[str]]
    
    probable_diagnosis: Optional[str]
    treatment_recommendations: Optional[str]
    
    # Final output
    final_report: Optional[str]
