import { useState, useRef, useEffect } from 'react';
import Head from 'next/head';
import { motion } from 'framer-motion';
import EmotionTimeline from '../components/EmotionTimeline';
import VideoFeed from '../components/VideoFeed';
import ModalityContributions from '../components/ModalityContributions';
import IntentProbabilities from '../components/IntentProbabilities';

export default function Home() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentAnalysis, setCurrentAnalysis] = useState(null);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const startAnalysis = async () => {
    setIsAnalyzing(true);
    // Initialize webcam
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      // Start analysis loop
      analyzeFrame();
    } catch (error) {
      console.error('Error accessing webcam:', error);
      setIsAnalyzing(false);
    }
  };

  const stopAnalysis = () => {
    setIsAnalyzing(false);
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
    }
  };

  const analyzeFrame = async () => {
    if (!isAnalyzing || !videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const video = videoRef.current;

    // Draw current frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to blob for API
    canvas.toBlob(async (blob) => {
      const formData = new FormData();
      formData.append('image', blob, 'frame.jpg');

      try {
        const response = await fetch('http://localhost:8000/analyze/frame', {
          method: 'POST',
          body: formData
        });

        if (response.ok) {
          const result = await response.json();
          setCurrentAnalysis(result);
          setAnalysisHistory(prev => [...prev.slice(-49), result]); // Keep last 50
        }
      } catch (error) {
        console.error('Analysis error:', error);
      }
    });

    // Continue analysis loop
    if (isAnalyzing) {
      setTimeout(analyzeFrame, 1000); // Analyze every second
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <Head>
        <title>EMOTIA - Multi-Modal Emotion & Intent Intelligence</title>
        <meta name="description" content="Real-time emotion and intent analysis for video calls" />
      </Head>

      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold text-cyan-400">EMOTIA</h1>
          <div className="flex space-x-4">
            <button
              onClick={isAnalyzing ? stopAnalysis : startAnalysis}
              className={`px-6 py-2 rounded-lg font-semibold transition-colors ${
                isAnalyzing
                  ? 'bg-red-600 hover:bg-red-700'
                  : 'bg-cyan-600 hover:bg-cyan-700'
              }`}
            >
              {isAnalyzing ? 'Stop Analysis' : 'Start Analysis'}
            </button>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="max-w-7xl mx-auto p-4 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Panel - Video Feed */}
        <div className="lg:col-span-1">
          <div className="bg-gray-800 rounded-xl p-4 glassmorphism">
            <h2 className="text-xl font-semibold mb-4 text-cyan-400">Live Video Feed</h2>
            <VideoFeed
              videoRef={videoRef}
              canvasRef={canvasRef}
              isAnalyzing={isAnalyzing}
            />
          </div>
        </div>

        {/* Center Panel - Emotion Timeline */}
        <div className="lg:col-span-1">
          <div className="bg-gray-800 rounded-xl p-4 glassmorphism">
            <h2 className="text-xl font-semibold mb-4 text-lime-400">Emotion Timeline</h2>
            <EmotionTimeline history={analysisHistory} />
          </div>
        </div>

        {/* Right Panel - Analysis Results */}
        <div className="lg:col-span-1 space-y-6">
          {/* Current Analysis */}
          <div className="bg-gray-800 rounded-xl p-4 glassmorphism">
            <h2 className="text-xl font-semibold mb-4 text-violet-400">Current Analysis</h2>
            {currentAnalysis ? (
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold text-cyan-300">Dominant Emotion</h3>
                  <p className="text-2xl font-bold text-cyan-400">
                    {currentAnalysis.emotion.dominant}
                  </p>
                </div>
                <div>
                  <h3 className="font-semibold text-lime-300">Intent</h3>
                  <p className="text-xl font-bold text-lime-400">
                    {currentAnalysis.intent.dominant}
                  </p>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h3 className="font-semibold text-violet-300">Engagement</h3>
                    <p className="text-lg font-bold text-violet-400">
                      {(currentAnalysis.engagement * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <h3 className="font-semibold text-pink-300">Confidence</h3>
                    <p className="text-lg font-bold text-pink-400">
                      {(currentAnalysis.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-gray-400">No analysis available</p>
            )}
          </div>

          {/* Modality Contributions */}
          <ModalityContributions contributions={currentAnalysis?.modality_contributions} />

          {/* Intent Probabilities */}
          <IntentProbabilities probabilities={currentAnalysis?.intent.predictions} />
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700 p-4 mt-8">
        <div className="max-w-7xl mx-auto text-center text-gray-400">
          <p>EMOTIA - Ethical AI for Human-Centric Video Analysis</p>
        </div>
      </footer>
    </div>
  );
}