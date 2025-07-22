// app/page.tsx
"use client";

import { useState, useRef, useEffect } from "react";
import { Mic, Bot, User, Volume2, Loader2, Square, AlertTriangle } from "lucide-react";
import ReactMarkdown from 'react-markdown';

// --- ElevenLabs Configuration ---
// WARNING: Exposing API keys on the client-side is a security risk.
// This should only be used for local development or testing.
// For production, use a server-side API route to protect your key.
const ELEVENLABS_API_KEY = process.env.NEXT_PUBLIC_ELEVENLABS_API_KEY || '';
const ELEVENLABS_VOICE_ID = 'Rachel'; // A popular, expressive voice.
const ELEVENLABS_API_URL = `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}/stream`;

// --- Type Definitions ---
type Message = {
  role: "user" | "assistant";
  content: string;
};

// --- Main Chat Page Component ---
export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [sttError, setSttError] = useState<string | null>(null);
  
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // --- Speech Recognition (STT) Setup using Web Speech API ---
  useEffect(() => {
    // Check for browser support
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setSttError("Speech recognition is not supported in this browser. Please use Chrome or Safari.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false; // Process speech after user stops talking
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      setIsRecording(true);
      setSttError(null);
    };

    recognition.onend = () => {
      setIsRecording(false);
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
      let errorMessage = `An error occurred: ${event.error}.`;
      if (event.error === 'network') {
        errorMessage = "Network error. Please check your internet connection and try again.";
      } else if (event.error === 'not-allowed') {
        errorMessage = "Microphone access denied. Please allow microphone access in your browser settings.";
      }
      setSttError(errorMessage);
      setIsRecording(false);
    };

    recognition.onresult = async (event) => {
      const transcript = event.results[0][0].transcript.trim();
      if (transcript) {
        // Once we have the transcript, submit it to the chat handler
        await handleChatSubmit(transcript);
      }
    };

    recognitionRef.current = recognition;
  }, []);

  // Auto-scroll to the latest message
  useEffect(() => {
    chatContainerRef.current?.scrollTo({
      top: chatContainerRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages]);
  
  // --- Text-to-Speech (TTS) Function using ElevenLabs ---
  const playAudio = async (text: string) => {
    if (!text.trim()) return;
    if (!ELEVENLABS_API_KEY) {
        console.error("ElevenLabs API key is not set.");
        setMessages(prev => [...prev, { role: 'assistant', content: "Audio playback is not configured by the developer."}])
        return;
    }

    setIsSpeaking(true);
    try {
      const response = await fetch(ELEVENLABS_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'xi-api-key': ELEVENLABS_API_KEY,
        },
        body: JSON.stringify({
          text: text,
          model_id: "eleven_turbo_v2", // A fast and high-quality model
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`ElevenLabs API request failed: ${response.statusText}`);
      }
      
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      audio.play();
      audio.onended = () => { setIsSpeaking(false); URL.revokeObjectURL(audioUrl); };
      audio.onerror = () => { setIsSpeaking(false); console.error("Error playing the audio."); };

    } catch (error) {
      console.error("Error playing audio from ElevenLabs:", error);
      setIsSpeaking(false);
    }
  };

  // --- Main Logic to Handle Chat Submission ---
  const handleChatSubmit = async (transcript: string) => {
    if (!transcript || isLoading) return;

    const userMessage: Message = { role: "user", content: transcript };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // This calls your backend API route for the LLM
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: [{ role: "user", content: transcript }] }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `API request failed`);
      }

      const data = await response.json();
      const aiResponseContent = data.choices[0]?.message?.content || "Sorry, I couldn't get a valid response.";
      const assistantMessage: Message = { role: "assistant", content: aiResponseContent };
      setMessages((prev) => [...prev, assistantMessage]);

      // Automatically play the audio for the new AI response
      await playAudio(aiResponseContent);

    } catch (error) {
      console.error("API call error:", error);
      const errorMessageContent = error instanceof Error ? error.message : "An unknown error occurred.";
      const errorMessage: Message = { role: "assistant", content: `Sorry, something went wrong: ${errorMessageContent}` };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleVoiceButtonClick = () => {
    if (sttError) {
        setSttError(null);
        return;
    }
    if (isRecording) {
      recognitionRef.current?.stop();
    } else {
      recognitionRef.current?.start();
    }
  };
  
  // --- Render Voice Button State ---
  const renderVoiceButton = () => {
    const isDisabled = isSpeaking;
    let icon = <Mic className="w-8 h-8" />;
    let text = "Click to speak";
    let buttonClass = "bg-blue-500 hover:bg-blue-600";
    
    if (sttError) {
        icon = <AlertTriangle className="w-8 h-8" />;
        text = "Click to try again";
        buttonClass = "bg-yellow-500 hover:bg-yellow-600";
    } else if (isLoading) {
      icon = <Loader2 className="w-8 h-8 animate-spin" />;
      text = "Thinking...";
      buttonClass = "bg-gray-400";
    } else if (isRecording) {
      icon = <Square className="w-8 h-8 text-red-500" />;
      text = "Listening...";
      buttonClass = "bg-white ring-4 ring-blue-300 animate-pulse";
    } else if (isSpeaking) {
      icon = <Volume2 className="w-8 h-8" />;
      text = "Speaking...";
      buttonClass = "bg-gray-400";
    }

    return (
        <div className="flex flex-col items-center gap-2">
            <button
              onClick={handleVoiceButtonClick}
              disabled={isDisabled && !sttError}
              className={`w-20 h-20 rounded-full flex items-center justify-center text-white transition-all duration-300 shadow-lg focus:outline-none focus:ring-4 focus:ring-blue-300 ${buttonClass} disabled:bg-gray-400 disabled:cursor-not-allowed`}
              aria-label={text}
            >
              {icon}
            </button>
            <p className="text-sm text-center font-medium text-gray-500 h-10 w-64">
                {sttError ? <span className="text-red-600">{sttError}</span> : text}
            </p>
        </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-white">
      <header className="p-4 border-b border-gray-200 flex-shrink-0">
        <div className="max-w-3xl mx-auto">
          <h1 className="text-xl font-semibold text-gray-800">Voice Assistant</h1>
        </div>
      </header>

      <main ref={chatContainerRef} className="flex-grow overflow-y-auto p-4 md:p-6">
        {messages.length === 0 && !isLoading ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Bot className="w-24 h-24 text-gray-300 mb-4" />
            <h1 className="text-4xl font-semibold text-gray-800">Ready to talk?</h1>
            <p className="text-gray-500 mt-2">Click the microphone below to start the conversation.</p>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg, index) => (
              <div key={index} className={`flex items-start gap-4 animate-fade-in ${msg.role === 'user' ? 'justify-end' : ''}`}>
                {msg.role === 'assistant' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                    <Bot className="w-5 h-5 text-gray-600" />
                  </div>
                )}
                <div className={`p-4 rounded-2xl max-w-xl ${
                  msg.role === 'user' 
                  ? 'bg-blue-500 text-white rounded-br-none' 
                  : 'bg-gray-100 text-gray-800 rounded-bl-none'
                }`}>
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                  {msg.role === 'assistant' && (
                    <button 
                      onClick={() => playAudio(msg.content)} 
                      disabled={isSpeaking || isLoading}
                      className="mt-2 p-1 text-gray-500 hover:text-gray-800 disabled:opacity-50"
                      aria-label="Replay audio"
                    >
                      <Volume2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
                 {msg.role === 'user' && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                    <User className="w-5 h-5 text-blue-600" />
                  </div>
                )}
              </div>
            ))}
             {isLoading && (
                <div className="flex items-start gap-4 animate-fade-in">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                      <Bot className="w-5 h-5 text-gray-600" />
                  </div>
                  <div className="p-4 rounded-2xl bg-gray-100 flex items-center">
                      <Loader2 className="w-5 h-5 animate-spin text-gray-500" />
                  </div>
                </div>
            )}
          </div>
        )}
      </main>

      <footer className="p-6 bg-white/80 backdrop-blur-sm border-t border-gray-200 flex-shrink-0">
        <div className="flex justify-center">
           {renderVoiceButton()}
        </div>
      </footer>
    </div>
  );
}