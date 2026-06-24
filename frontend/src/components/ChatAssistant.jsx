import React, { useState } from 'react';
import axios from 'axios';
import { Send, Bot } from 'lucide-react';

const API_URL = 'http://localhost:8000';

export default function ChatAssistant({ patientContext }) {
  const [messages, setMessages] = useState([
    { role: 'bot', content: 'Hello Dr., I am your Llama3-powered clinical assistant. Ask me anything about the diagnosis or treatment guidelines.' }
  ]);
  const [input, setInput] = useState('');

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');

    try {
      const res = await axios.post(`${API_URL}/chat`, {
        message: userMsg.content,
        patient_context: patientContext
      });
      setMessages(prev => [...prev, { role: 'bot', content: res.data.reply }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: 'bot', content: 'Error connecting to LLM server.' }]);
    }
  };

  return (
    <>
      <h2 style={{borderBottom: '1px solid var(--border-color)', paddingBottom: '16px', marginBottom: '16px'}}>
        <Bot size={20} style={{marginRight: '8px', verticalAlign: 'text-bottom'}} />
        Clinical Assistant
      </h2>
      <div className="chat-messages">
        {messages.map((m, idx) => (
          <div key={idx} className={`chat-bubble ${m.role}`}>
            {m.content}
          </div>
        ))}
      </div>
      <div className="chat-input">
        <input 
          value={input} 
          onChange={e => setInput(e.target.value)} 
          onKeyPress={e => e.key === 'Enter' && handleSend()}
          placeholder="Ask why Glioma was predicted..." 
        />
        <button onClick={handleSend} style={{padding: '12px'}}><Send size={18} /></button>
      </div>
    </>
  );
}
