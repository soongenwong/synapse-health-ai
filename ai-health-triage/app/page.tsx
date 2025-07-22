// app/page.tsx

"use client"; // <--- IMPORTANT: Add this for components with state and interactivity

import { useState, FormEvent } from 'react';
import { Stethoscope, BrainCircuit, AlertTriangle, ShieldCheck, UserCheck, HeartPulse, Loader2 } from 'lucide-react';

// Define a type for our results for better type-safety
type TriageResult = {
  urgency: {
    level: 'High' | 'Medium' | 'Low';
    assessment: string;
  };
  conditions: {
    name: string;
    probability: number;
  }[];
  recommendations: {
    title: string;
    details: string;
    icon: React.ElementType;
  }[];
};

// --- Mock API Call Simulation ---
// In a real hackathon, you'd replace this with an actual API call to your backend.
const getMockTriageResult = async (symptoms: string): Promise<TriageResult> => {
  return new Promise(resolve => {
    setTimeout(() => {
      // Simple logic for demonstration
      if (symptoms.toLowerCase().includes('chest pain')) {
        resolve({
          urgency: { level: 'High', assessment: 'Immediate attention recommended.' },
          conditions: [
            { name: 'Myocardial Infarction (Heart Attack)', probability: 75 },
            { name: 'Angina', probability: 60 },
            { name: 'Anxiety Attack', probability: 45 },
          ],
          recommendations: [
            { title: 'Go to A&E / Emergency Room', details: 'Seek immediate medical care. Do not drive yourself.', icon: AlertTriangle },
            { title: 'Call Emergency Services', details: 'Dial 999 or your local emergency number.', icon: HeartPulse },
          ],
        });
      } else {
        resolve({
          urgency: { level: 'Low', assessment: 'Monitor symptoms at home.' },
          conditions: [
            { name: 'Common Cold', probability: 85 },
            { name: 'Allergic Rhinitis', probability: 50 },
            { name: 'Sinusitis', probability: 40 },
          ],
          recommendations: [
            { title: 'Self-Care', details: 'Rest, stay hydrated, and use over-the-counter medication if needed.', icon: ShieldCheck },
            { title: 'Contact a GP if symptoms worsen', details: 'Schedule a non-urgent appointment with your doctor.', icon: UserCheck },
          ],
        });
      }
    }, 1500); // Simulate 1.5 second network delay
  });
};

// Urgency Card Styling
const urgencyStyles = {
  High: {
    bg: 'bg-red-50',
    border: 'border-red-500',
    text: 'text-red-800',
    icon: <AlertTriangle className="h-8 w-8 text-red-500" />
  },
  Medium: {
    bg: 'bg-yellow-50',
    border: 'border-yellow-500',
    text: 'text-yellow-800',
    icon: <AlertTriangle className="h-8 w-8 text-yellow-500" />
  },
  Low: {
    bg: 'bg-green-50',
    border: 'border-green-500',
    text: 'text-green-800',
    icon: <ShieldCheck className="h-8 w-8 text-green-500" />
  },
}

export default function Home() {
  const [symptomInput, setSymptomInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<TriageResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!symptomInput.trim()) {
      setError('Please describe your symptoms.');
      return;
    }
    setError(null);
    setIsLoading(true);
    setResults(null);

    const triageData = await getMockTriageResult(symptomInput);
    setResults(triageData);
    setIsLoading(false);
  };

  return (
    <main className="min-h-screen bg-gray-50 font-sans text-gray-800">
      <div className="container mx-auto max-w-4xl p-4 sm:p-8">
        
        {/* Header */}
        <header className="mb-8 text-center">
          <div className="flex justify-center items-center gap-3">
            <Stethoscope className="h-10 w-10 text-blue-600" />
            <h1 className="text-4xl font-bold tracking-tight text-gray-900">
              Symptom Sentry
            </h1>
          </div>
          <p className="mt-2 text-lg text-gray-600">
            Your AI-powered guide for health next steps.
          </p>
        </header>
        
        {/* Symptom Input Form */}
        <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200">
          <form onSubmit={handleSubmit}>
            <label htmlFor="symptoms" className="block text-lg font-semibold mb-2 text-gray-700">
              Describe your symptoms
            </label>
            <textarea
              id="symptoms"
              name="symptoms"
              rows={4}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-shadow"
              placeholder="e.g., I have a sharp headache, a fever of 38°C, and a sore throat..."
              value={symptomInput}
              onChange={(e) => setSymptomInput(e.target.value)}
            />
            {error && <p className="text-red-500 mt-2">{error}</p>}
            <button
              type="submit"
              disabled={isLoading}
              className="mt-4 w-full flex items-center justify-center gap-2 bg-blue-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-700 disabled:bg-blue-300 transition-all"
            >
              {isLoading ? (
                <>
                  <Loader2 className="animate-spin h-5 w-5" />
                  Analyzing...
                </>
              ) : 'Analyze Symptoms'}
            </button>
          </form>
        </div>

        {/* Results Section */}
        {results && (
          <div className="mt-10 space-y-8 animate-fade-in">
            {/* Urgency Assessment */}
            <div className={`p-6 rounded-xl border-l-4 ${urgencyStyles[results.urgency.level].bg} ${urgencyStyles[results.urgency.level].border}`}>
              <div className="flex items-start gap-4">
                <div>{urgencyStyles[results.urgency.level].icon}</div>
                <div>
                  <h3 className={`text-xl font-bold ${urgencyStyles[results.urgency.level].text}`}>
                    Urgency Level: {results.urgency.level}
                  </h3>
                  <p className="mt-1 text-gray-700">{results.urgency.assessment}</p>
                </div>
              </div>
            </div>

            {/* Likely Conditions & Next Steps */}
            <div className="grid md:grid-cols-2 gap-8">
              {/* Likely Conditions */}
              <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200">
                <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
                  <BrainCircuit className="h-6 w-6 text-blue-500" />
                  Possible Conditions
                </h3>
                <ul className="space-y-3">
                  {results.conditions.map((cond, index) => (
                    <li key={index} className="flex justify-between items-center text-gray-600">
                      <span>{cond.name}</span>
                      <span className="text-sm font-medium bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
                        {cond.probability}%
                      </span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Recommended Next Steps */}
              <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200">
                <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
                  <UserCheck className="h-6 w-6 text-green-500" />
                  Recommended Next Steps
                </h3>
                <ul className="space-y-4">
                  {results.recommendations.map((rec, index) => (
                    <li key={index} className="flex items-start gap-3">
                      <div className="mt-1">
                        <rec.icon className="h-5 w-5 text-gray-500" />
                      </div>
                      <div>
                        <p className="font-semibold text-gray-800">{rec.title}</p>
                        <p className="text-sm text-gray-600">{rec.details}</p>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

      </div>
    </main>
  );
}