import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import Login from './components/Login';

function App() {
  return (
    <Router>
      <header className="header">
        <h1>Agentic AI Multimodal Tumor Diagnosis</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Powered by LangGraph, U-Net, and Explainable RAG</p>
      </header>
      <main style={{ padding: '24px', display: 'flex', justifyContent: 'center' }}>
        <Routes>
          <Route path="/" element={<Login />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </Router>
  );
}

export default App;
