// app/page.tsx
"use client";

import { useState, FormEvent, useRef, useEffect } from "react";
import { Plus, Sparkles, Mic, ArrowUp, Loader2, Bot, User, WandSparkles } from "lucide-react";
import ReactMarkdown from 'react-markdown';

// Define the structure of a message in our chat
type Message = {
  role: "user" | "assistant";
  content: string;
};

// --- UPDATED ChatInputForm component ---
// This is the only part that has changed.
interface ChatInputFormProps {
  input: string;
  setInput: (value: string) => void;
  handleSubmit: (e: FormEvent) => Promise<void>;
  isLoading: boolean;
}

const ChatInputForm = ({ input, setInput, handleSubmit, isLoading }: ChatInputFormProps) => {
  return (
    <form onSubmit={handleSubmit} className="relative w-full max-w-3xl">
      {/* Increased padding here from p-2 to p-3 for a larger height */}
      <div className="relative flex items-center p-3 bg-gray-100 rounded-full shadow-sm border border-gray-200">
        <button type="button" className="p-2 text-gray-500 hover:text-gray-800">
          <Plus className="w-5 h-5" />
        </button>
        <button type="button" className="p-2 flex items-center gap-1 text-gray-500 hover:text-gray-800">
          <WandSparkles className="w-5 h-5" />
          <span className="text-sm font-medium">Tools</span>
        </button>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault(); 
              handleSubmit(e);
            }
          }}
          // Updated placeholder text
          placeholder="Ask anything..."
          // Removed rows={1} as padding now controls the height
          className="flex-grow px-4 py-2 bg-transparent resize-none focus:outline-none text-gray-800 placeholder-gray-500 overflow-y-auto max-h-32"
        />
        <button type="button" className="p-2 text-gray-500 hover:text-gray-800">
          <Mic className="w-5 h-5" />
        </button>
        {/* Improved button styling for better user feedback */}
        <button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="p-2.5 rounded-full transition-colors
                     bg-blue-500 text-white hover:bg-blue-600 
                     disabled:bg-gray-200 disabled:text-gray-400 disabled:cursor-not-allowed"
        >
          {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <ArrowUp className="w-5 h-5" />}
        </button>
      </div>
    </form>
  );
};


// Main component for the chat page (NO CHANGES BELOW THIS LINE)
export default function Home() {

  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatContainerRef.current?.scrollTo({
      top: chatContainerRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    const currentInput = input;
    setInput("");
    setIsLoading(true);

    try {
      const apiKey = process.env.NEXT_PUBLIC_GROQ_API_KEY;
      if (!apiKey) throw new Error("Groq API key is not configured.");
      
      const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: "llama3-8b-8192",
          messages: [
            { role: "system", content: "You are an AI-powered Symptom Triage Tool. Your goal is to help users by suggesting likely conditions, assessing urgency, and recommending next steps. Format your response clearly using markdown. Start with Urgency, then Possible Conditions, then Recommendations." },
            { role: "user", content: currentInput },
          ],
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error.message || `API request failed`);
      }

      const data = await response.json();
      const aiResponseContent = data.choices[0]?.message?.content || "Sorry, I couldn't get a valid response.";
      const assistantMessage: Message = { role: "assistant", content: aiResponseContent };
      setMessages((prev) => [...prev, assistantMessage]);

    } catch (error) {
      console.error("API call error:", error);
      const errorMessageContent = error instanceof Error ? error.message : "An unknown error occurred.";
      const errorMessage: Message = { role: "assistant", content: `Sorry, something went wrong: ${errorMessageContent}` };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  if (messages.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-white p-4">
        <div className="text-center mb-8">
            <h1 className="text-4xl font-semibold text-gray-800">Where should we begin?</h1>
        </div>
        <ChatInputForm
          input={input}
          setInput={setInput}
          handleSubmit={handleSubmit}
          isLoading={isLoading}
        />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-white">
      <main ref={chatContainerRef} className="flex-grow overflow-y-auto p-4 md:p-6">
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
                <ReactMarkdown
                  components={{
                    // This will apply the classes to the root div rendered by react-markdown
                    div: ({ node, ...props }) => <div className="prose prose-sm max-w-none" {...props} />
                  }}
                >
                  {msg.content}
                </ReactMarkdown>
              </div>
               {msg.role === 'user' && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                  <User className="w-5 h-5 text-gray-600" />
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
      </main>

      <footer className="p-4 bg-white/80 backdrop-blur-sm border-t border-gray-200">
        <div className="flex justify-center">
           <ChatInputForm
            input={input}
            setInput={setInput}
            handleSubmit={handleSubmit}
            isLoading={isLoading}
          />
        </div>
      </footer>
    </div>
  );
}