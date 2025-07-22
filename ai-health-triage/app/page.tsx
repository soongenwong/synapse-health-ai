// app/page.tsx
"use client";

import { useState, useRef, useEffect } from "react";
import { Mic, Bot, User, Volume2, Loader2, Square, AlertTriangle } from "lucide-react";
import ReactMarkdown from 'react-markdown';

// --- API Configuration (INSECURE - FOR LOCAL DEVELOPMENT ONLY) ---
const GROQ_API_KEY = process.env.NEXT_PUBLIC_GROQ_API_KEY || '';
const ELEVENLABS_API_KEY = process.env.NEXT_PUBLIC_ELEVENLABS_API_KEY || '';

const GROQ_CHAT_URL = 'https://api.groq.com/openai/v1/chat/completions';
const GROQ_STT_URL = 'https://api.groq.com/openai/v1/audio/transcriptions';

// --- THIS IS THE FIX ---
// The API requires the specific Voice ID, not the name "Rachel".
const ELEVENLABS_VOICE_ID = '21m00Tcm4TlvDq8ikWAM'; 
const ELEVENLABS_TTS_URL = `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}/stream`;

type Message = {
  role: "user" | "assistant";
  content: string;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatContainerRef.current?.scrollTo({
      top: chatContainerRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages]);
  
  // --- Text-to-Speech (TTS) with ElevenLabs ---
  const playAudio = async (text: string) => {
    if (!text.trim() || !ELEVENLABS_API_KEY) {
      if (!ELEVENLABS_API_KEY) setError("ElevenLabs API Key is not configured.");
      return;
    };
    setIsSpeaking(true);
    setError(null);
    try {
      const response = await fetch(ELEVENLABS_TTS_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'xi-api-key': ELEVENLABS_API_KEY },
        body: JSON.stringify({ text, model_id: "eleven_turbo_v2" }),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        const detail = errorData?.detail?.message || response.statusText;
        throw new Error(`ElevenLabs API Error (${response.status}): ${detail}`);
      }
      
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audio.play();
      audio.onended = () => { setIsSpeaking(false); URL.revokeObjectURL(audioUrl); };
      audio.onerror = () => { setIsSpeaking(false); setError("Error playing audio."); };
    } catch (err: any) {
      console.error("ElevenLabs TTS Error:", err);
      setError(err.message);
      setIsSpeaking(false);
    }
  };

  // --- Chat Completion with Groq ---
  const handleChatSubmit = async (transcript: string) => {
    if (!transcript || isLoading) return;
    if (!GROQ_API_KEY) {
      setError("Groq API Key is not configured.");
      return;
    }

    const userMessage: Message = { role: "user", content: transcript };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    const systemMessage = {
      role: 'system',
      content: 'You are a friendly and helpful voice assistant. Keep your responses concise and conversational.'
    };

    try {
      const response = await fetch(GROQ_CHAT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${GROQ_API_KEY}` },
        body: JSON.stringify({
          messages: [systemMessage, { role: "user", content: transcript }],
          model: 'llama3-8b-8192'
        }),
      });
      if (!response.ok) throw new Error(`Groq Chat Error: ${await response.text()}`);
      
      const data = await response.json();
      const aiResponseContent = data.choices[0]?.message?.content || "Sorry, I couldn't get a valid response.";
      const assistantMessage: Message = { role: "assistant", content: aiResponseContent };
      setMessages((prev) => [...prev, assistantMessage]);
      await playAudio(aiResponseContent);
    } catch (err: any) {
      console.error("Chat API Error:", err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // --- Voice Recording and Transcription (STT) ---
  const handleToggleRecording = async () => {
    if (error) { setError(null); return; }
    if (isRecording) {
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
    } else {
      if (!GROQ_API_KEY) { setError("Groq API Key is not configured for transcription."); return; }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        audioChunksRef.current = [];
        mediaRecorderRef.current.ondataavailable = (event) => audioChunksRef.current.push(event.data);
        mediaRecorderRef.current.onstop = async () => {
          setIsLoading(true);
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          stream.getTracks().forEach(track => track.stop());
          const formData = new FormData();
          formData.append('file', audioBlob, 'audio.webm');
          formData.append('model', 'whisper-large-v3');
          try {
            const sttResponse = await fetch(GROQ_STT_URL, { method: 'POST', headers: { 'Authorization': `Bearer ${GROQ_API_KEY}` }, body: formData });
            if (!sttResponse.ok) throw new Error(`Groq STT Error: ${await sttResponse.text()}`);
            const { text: transcript } = await sttResponse.json();
            if (transcript && transcript.trim() !== '') { await handleChatSubmit(transcript); }
          } catch (err: any) {
            console.error("Transcription/Chat Error:", err);
            setError("Sorry, I couldn't process that. Please try again.");
          } finally {
            setIsLoading(false);
          }
        };
        mediaRecorderRef.current.start();
        setIsRecording(true);
      } catch (err) {
        console.error("Microphone Access Error:", err);
        setError("Microphone access denied. Please allow it in browser settings.");
      }
    }
  };
  
  // --- UI Rendering ---
  const renderVoiceButton = () => {
    let icon = <Mic className="w-8 h-8" />; let text = "Click to speak"; let buttonClass = "bg-blue-500 hover:bg-blue-600";
    if (error) { icon = <AlertTriangle className="w-8 h-8" />; text = "Click to try again"; buttonClass = "bg-red-500 hover:bg-red-600";
    } else if (isRecording) { icon = <Square className="w-8 h-8 text-red-500" />; text = "Listening... Tap to stop"; buttonClass = "bg-white ring-4 ring-blue-300 animate-pulse";
    } else if (isLoading) { icon = <Loader2 className="w-8 h-8 animate-spin" />; text = "Thinking..."; buttonClass = "bg-gray-400";
    } else if (isSpeaking) { icon = <Volume2 className="w-8 h-8" />; text = "Speaking..."; buttonClass = "bg-gray-400"; }
    return (
      <div className="flex flex-col items-center gap-2">
        <button onClick={handleToggleRecording} disabled={isSpeaking || isLoading} className={`w-20 h-20 rounded-full flex items-center justify-center text-white transition-all duration-300 shadow-lg focus:outline-none focus:ring-4 focus:ring-blue-300 ${buttonClass} disabled:bg-gray-400 disabled:cursor-not-allowed`} aria-label={text}>{icon}</button>
        <p className="text-sm text-center font-medium text-gray-500 h-10 w-64">{error ? <span className="text-red-600">{error}</span> : text}</p>
      </div>
    );
  }
  return (
    <div className="flex flex-col h-screen bg-white">
      <header className="p-4 border-b border-gray-200 flex-shrink-0"><div className="max-w-3xl mx-auto"><h1 className="text-xl font-semibold text-gray-800">Synapse Health AI</h1></div></header>
      <main ref={chatContainerRef} className="flex-grow overflow-y-auto p-4 md:p-6">
        {messages.length === 0 && !isLoading ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Bot className="w-24 h-24 text-gray-300 mb-4" />
            <h1 className="text-4xl font-semibold text-gray-800">Ready to talk?</h1>
            <p className="text-gray-500 mt-2">Click the microphone below to start.</p>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-6">{messages.map((msg, index) => (
              <div key={index} className={`flex items-start gap-4 animate-fade-in ${msg.role === 'user' ? 'justify-end' : ''}`}>
                {msg.role === 'assistant' && <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center"><Bot className="w-5 h-5 text-gray-600" /></div>}
                <div className={`p-4 rounded-2xl max-w-xl ${msg.role === 'user' ? 'bg-blue-500 text-white rounded-br-none' : 'bg-gray-100 text-gray-800 rounded-bl-none'}`}>
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                  {msg.role === 'assistant' && <button onClick={() => playAudio(msg.content)} disabled={isSpeaking || isLoading} className="mt-2 p-1 text-gray-500 hover:text-gray-800 disabled:opacity-50" aria-label="Replay audio"><Volume2 className="w-4 h-4" /></button>}
                </div>
                {msg.role === 'user' && <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center"><User className="w-5 h-5 text-blue-600" /></div>}
              </div>
            ))}
            {isLoading && !isSpeaking && (
              <div className="flex items-start gap-4 animate-fade-in">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center"><Bot className="w-5 h-5 text-gray-600" /></div>
                <div className="p-4 rounded-2xl bg-gray-100 flex items-center"><Loader2 className="w-5 h-5 animate-spin text-gray-500" /></div>
              </div>
            )}
          </div>
        )}
      </main>
      <footer className="p-6 bg-white/80 backdrop-blur-sm border-t border-gray-200 flex-shrink-0"><div className="flex justify-center">{renderVoiceButton()}</div></footer>
    </div>
  );
}