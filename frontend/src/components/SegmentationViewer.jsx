import React, { useState } from 'react';
import { Layers } from 'lucide-react';

export default function SegmentationViewer({ maskUrl, heatmapUrl }) {
  const [view, setView] = useState('heatmap'); // 'heatmap' or 'mask'

  return (
    <div>
      <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px'}}>
        <h3><Layers size={18} style={{marginRight: '8px'}} /> Explainability Viewer</h3>
        <div style={{display: 'flex', gap: '8px'}}>
          <button 
            onClick={() => setView('heatmap')}
            style={{padding: '6px 12px', background: view === 'heatmap' ? 'var(--primary-color)' : 'transparent', border: '1px solid var(--primary-color)'}}
          >
            Grad-CAM Heatmap
          </button>
          <button 
            onClick={() => setView('mask')}
            style={{padding: '6px 12px', background: view === 'mask' ? 'var(--secondary-color)' : 'transparent', border: '1px solid var(--secondary-color)'}}
          >
            U-Net Mask
          </button>
        </div>
      </div>
      
      <div style={{
        width: '100%', 
        height: '250px', 
        background: '#000', 
        borderRadius: '8px', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        overflow: 'hidden',
        border: '1px solid var(--border-color)'
      }}>
        {view === 'heatmap' && heatmapUrl ? (
          <img src={heatmapUrl} alt="Grad-CAM" style={{maxHeight: '100%', maxWidth: '100%', objectFit: 'contain'}} />
        ) : view === 'mask' && maskUrl ? (
          <img src={maskUrl} alt="U-Net Mask" style={{maxHeight: '100%', maxWidth: '100%', objectFit: 'contain'}} />
        ) : (
          <span style={{color: 'var(--text-secondary)'}}>No {view} data available</span>
        )}
      </div>
    </div>
  );
}
