import React, { useState } from 'react';
import axios from 'axios';
import { Upload, Brain, Activity, FileText } from 'lucide-react';
import ChatAssistant from './ChatAssistant';
import SegmentationViewer from './SegmentationViewer';

const API_URL = 'http://localhost:8000';

export default function Dashboard() {
  const [file, setFile] = useState(null);
  const [patientName, setPatientName] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [patientGender, setPatientGender] = useState('');
  const [patientBlood, setPatientBlood] = useState('');
  const [doctorName, setDoctorName] = useState('');
  const [clinicalText, setClinicalText] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleAnalyze = async () => {
    if (!file) return alert("Please upload an MRI image.");
    setLoading(true);
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('clinical_text', clinicalText);
    formData.append('patient_age', patientAge);
    formData.append('patient_gender', patientGender);

    try {
      const response = await axios.post(`${API_URL}/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Analysis failed. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleGeneratePDF = async () => {
    if (!result) return;
    setPdfLoading(true);
    try {
      const payload = {
        patient: {
          name: patientName || 'Unknown',
          age: patientAge || 'Unknown',
          gender: patientGender || 'Unknown',
          blood_group: patientBlood || 'Unknown',
          doctor: doctorName || 'Unknown'
        },
        analysis: {
          prediction: result.prediction,
          confidence: result.resnet_confidence || 0,
          risk_level: result.resnet_confidence > 0.8 ? "High" : "Medium",
          tumor_size_cm: result.tumor_size_cm || 0
        },
        img_url: result.gradcam_image
      };

      const response = await axios.post(`${API_URL}/generate-report`, payload, {
        responseType: 'blob' // important for file download
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'Radiology_Report.pdf');
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error(error);
      alert("Failed to generate PDF report.");
    } finally {
      setPdfLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Left Column: Input Form & Results */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        <div className="glass-panel">
          <h2><Activity size={20} style={{marginRight: '8px', verticalAlign: 'text-bottom'}} />Patient Data Input</h2>
          
          <div style={{ marginBottom: '16px' }}>
            <input 
              type="file" 
              accept="image/*" 
              onChange={e => setFile(e.target.files[0])} 
              style={{ marginBottom: '8px' }}
            />
            {file && (
              <div style={{ marginTop: '8px', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '8px', background: 'rgba(0,0,0,0.2)', textAlign: 'center' }}>
                <img 
                  src={URL.createObjectURL(file)} 
                  alt="Uploaded MRI Preview" 
                  style={{ maxHeight: '150px', maxWidth: '100%', borderRadius: '4px', objectFit: 'contain' }}
                />
              </div>
            )}
          </div>

          <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px'}}>
            <input 
              type="text" 
              placeholder="Patient Name" 
              value={patientName} 
              onChange={e => setPatientName(e.target.value)} 
            />
            <input 
              type="text" 
              placeholder="Doctor Name" 
              value={doctorName} 
              onChange={e => setDoctorName(e.target.value)} 
            />
            <input 
              type="number" 
              placeholder="Age" 
              value={patientAge} 
              onChange={e => setPatientAge(e.target.value)} 
            />
            <select value={patientGender} onChange={e => setPatientGender(e.target.value)}>
              <option value="">Select Gender</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
            <select value={patientBlood} onChange={e => setPatientBlood(e.target.value)}>
              <option value="">Blood Group</option>
              <option value="A+">A+</option>
              <option value="A-">A-</option>
              <option value="B+">B+</option>
              <option value="B-">B-</option>
              <option value="O+">O+</option>
              <option value="O-">O-</option>
              <option value="AB+">AB+</option>
              <option value="AB-">AB-</option>
            </select>
          </div>
          <textarea 
            placeholder="Clinical Notes (e.g., headache, seizures)" 
            rows="3" 
            value={clinicalText} 
            onChange={e => setClinicalText(e.target.value)}
          ></textarea>
          <button onClick={handleAnalyze} disabled={loading} style={{width: '100%'}}>
            {loading ? 'Running Multi-Agent Analysis...' : 'Analyze with Agentic AI'}
          </button>
        </div>

        {result && (
          <div className="glass-panel" style={{ animation: 'fadeIn 0.5s ease' }}>
            <h2><Brain size={20} style={{marginRight: '8px', verticalAlign: 'text-bottom'}} />Agentic Diagnosis</h2>
            
            <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px', marginBottom: '24px'}}>
              <div style={{background: 'rgba(59, 130, 246, 0.1)', padding: '16px', borderRadius: '8px', border: '1px solid var(--primary-color)'}}>
                <h4 style={{color: 'var(--text-secondary)'}}>Tumor Type</h4>
                <div style={{fontSize: '1.5rem', fontWeight: 'bold'}}>{result.prediction}</div>
                <div style={{fontSize: '0.875rem', color: '#10b981'}}>Conf: {(result.resnet_confidence * 100).toFixed(1)}%</div>
              </div>
              <div style={{background: 'rgba(16, 185, 129, 0.1)', padding: '16px', borderRadius: '8px', border: '1px solid var(--secondary-color)'}}>
                <h4 style={{color: 'var(--text-secondary)'}}>Size & Volume</h4>
                <div style={{fontSize: '1.5rem', fontWeight: 'bold'}}>{result.tumor_size_cm} cm</div>
              </div>
              <div style={{background: 'rgba(239, 68, 68, 0.1)', padding: '16px', borderRadius: '8px', border: '1px solid #ef4444'}}>
                <h4 style={{color: 'var(--text-secondary)'}}>Clinical Risk</h4>
                <div style={{fontSize: '1.5rem', fontWeight: 'bold', color: result.resnet_confidence > 0.8 ? '#ef4444' : '#f59e0b'}}>
                  {result.resnet_confidence > 0.8 ? "High" : "Medium"}
                </div>
              </div>
            </div>

            <SegmentationViewer maskUrl={result.segmentation_mask} heatmapUrl={result.gradcam_image} />

            <div style={{marginTop: '24px'}}>
              <h3><FileText size={18} style={{marginRight: '8px'}} />RAG Medical Context</h3>
              <p style={{background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '8px', fontSize: '0.9rem'}}>
                {result.agent_reasoning}
              </p>
            </div>

            <div style={{marginTop: '24px'}}>
              <h3>Recommended Treatment</h3>
              <p style={{color: '#34d399', fontWeight: '500', marginBottom: '16px'}}>{result.treatment_recommendations}</p>
              <button onClick={handleGeneratePDF} disabled={pdfLoading} style={{width: '100%', background: 'var(--secondary-color)'}}>
                {pdfLoading ? 'Generating PDF...' : 'Generate PDF Report'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Right Column: Chat Assistant */}
      <div className="glass-panel chat-container">
            <ChatAssistant 
              patientContext={{
                prediction: result?.prediction,
                tumor_size: result?.tumor_size_cm,
                age: patientAge,
                gender: patientGender,
                notes: clinicalText
              }} 
            />
      </div>
    </div>
  );
}
