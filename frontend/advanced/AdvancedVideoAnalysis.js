import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { Mic, MicOff, Video, VideoOff, Settings, Zap, Shield, BarChart3 } from 'lucide-react';

const AdvancedVideoAnalysis = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentAnalysis, setCurrentAnalysis] = useState(null);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionQuality, setConnectionQuality] = useState('good');
  const [modelVersion, setModelVersion] = useState('v2.0.0');
  const [privacyMode, setPrivacyMode] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const sessionIdRef = useRef(`session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);

  // WebRTC and WebSocket setup
  const initializeWebRTC = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        },
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      // Initialize WebSocket for real-time analysis
      initializeWebSocket();

      setIsConnected(true);
    } catch (error) {
      console.error('WebRTC initialization failed:', error);
      setConnectionQuality('error');
    }
  }, []);

  const initializeWebSocket = () => {
    const ws = new WebSocket(`ws://localhost:8000/ws/analyze/${sessionIdRef.current}`);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnectionQuality('good');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.error) {
        console.error('Analysis error:', data.error);
        setConnectionQuality('error');
      } else {
        setCurrentAnalysis(data);
        setAnalysisHistory(prev => [...prev.slice(-99), data]); // Keep last 100
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      setConnectionQuality('disconnected');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionQuality('error');
    };

    wsRef.current = ws;
  };

  const startAnalysis = async () => {
    setIsAnalyzing(true);
    await initializeWebRTC();
  };

  const stopAnalysis = () => {
    setIsAnalyzing(false);

    // Stop WebRTC stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close();
    }

    setIsConnected(false);
    setCurrentAnalysis(null);
  };

  // Real-time frame capture and streaming
  useEffect(() => {
    if (!isAnalyzing || !videoRef.current || !wsRef.current) return;

    const captureFrame = () => {
      if (!isAnalyzing) return;

      const canvas = canvasRef.current;
      const video = videoRef.current;
      const ctx = canvas.getContext('2d');

      if (video.videoWidth > 0 && video.videoHeight > 0) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        // Convert to blob and send via WebSocket
        canvas.toBlob((blob) => {
          if (blob && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            const reader = new FileReader();
            reader.onload = () => {
              const data = {
                image: reader.result,
                timestamp: Date.now(),
                sessionId: sessionIdRef.current
              };
              wsRef.current.send(JSON.stringify(data));
            };
            reader.readAsDataURL(blob);
          }
        }, 'image/jpeg', 0.8);
      }

      // Continue capturing at ~10 FPS
      setTimeout(captureFrame, 100);
    };

    captureFrame();

    return () => {
      // Cleanup
    };
  }, [isAnalyzing]);

  // Connection quality monitoring
  useEffect(() => {
    const checkConnection = () => {
      if (wsRef.current) {
        const state = wsRef.current.readyState;
        if (state === WebSocket.CLOSED || state === WebSocket.CLOSING) {
          setConnectionQuality('disconnected');
        } else if (state === WebSocket.CONNECTING) {
          setConnectionQuality('connecting');
        }
      }
    };

    const interval = setInterval(checkConnection, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      {/* Advanced Header */}
      <header className="bg-black/20 backdrop-blur-xl border-b border-white/10 p-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="text-3xl"
            >
              🚀
            </motion.div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-violet-400 bg-clip-text text-transparent">
                EMOTIA Advanced
              </h1>
              <p className="text-sm text-gray-400">Real-time Multi-Modal Intelligence</p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            {/* Connection Status */}
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${
                connectionQuality === 'good' ? 'bg-green-400' :
                connectionQuality === 'connecting' ? 'bg-yellow-400 animate-pulse' :
                'bg-red-400'
              }`} />
              <span className="text-sm capitalize">{connectionQuality}</span>
            </div>

            {/* Model Version */}
            <div className="text-sm text-gray-400">
              Model: {modelVersion}
            </div>

            {/* Privacy Mode */}
            <button
              onClick={() => setPrivacyMode(!privacyMode)}
              className={`p-2 rounded-lg transition-colors ${
                privacyMode ? 'bg-red-600 hover:bg-red-700' : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              <Shield className={`w-5 h-5 ${privacyMode ? 'text-white' : 'text-gray-400'}`} />
            </button>

            {/* Control Buttons */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={isAnalyzing ? stopAnalysis : startAnalysis}
              className={`px-6 py-3 rounded-xl font-semibold transition-all duration-300 ${
                isAnalyzing
                  ? 'bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 shadow-red-500/25'
                  : 'bg-gradient-to-r from-cyan-600 to-violet-600 hover:from-cyan-700 hover:to-violet-700 shadow-cyan-500/25'
              } shadow-lg`}
            >
              <div className="flex items-center space-x-2">
                <Zap className="w-5 h-5" />
                <span>{isAnalyzing ? 'Stop Analysis' : 'Start Advanced Analysis'}</span>
              </div>
            </motion.button>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="max-w-7xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Video Feed Panel */}
        <div className="lg:col-span-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-white/10"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-cyan-400">Live Video Feed</h2>
              <div className="flex space-x-2">
                <Video className="w-5 h-5 text-green-400" />
                <Mic className="w-5 h-5 text-blue-400" />
              </div>
            </div>

            <div className="relative aspect-video bg-black/50 rounded-xl overflow-hidden">
              <video
                ref={videoRef}
                autoPlay
                muted
                className="w-full h-full object-cover"
                style={{ display: isAnalyzing ? 'block' : 'none' }}
              />
              <canvas
                ref={canvasRef}
                className="w-full h-full"
                style={{ display: isAnalyzing ? 'none' : 'block' }}
              />

              {!isAnalyzing && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <div className="text-6xl mb-4">🎥</div>
                    <p className="text-gray-400">Advanced analysis ready</p>
                    <p className="text-sm text-gray-500 mt-2">WebRTC + WebSocket streaming</p>
                  </div>
                </div>
              )}

              {/* Real-time overlay */}
              {isAnalyzing && currentAnalysis && (
                <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm rounded-lg p-3">
                  <div className="text-sm">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                      <span>Processing: {currentAnalysis.processing_time?.toFixed(2)}s</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        </div>

        {/* Real-time Analytics */}
        <div className="lg:col-span-8 space-y-6">
          {/* Emotion Timeline */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-white/10"
          >
            <h2 className="text-xl font-semibold mb-4 text-lime-400">Real-time Emotion Timeline</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={analysisHistory.slice(-20)}>
                  <XAxis dataKey="timestamp" />
                  <YAxis domain={[0, 1]} />
                  <Area
                    type="monotone"
                    dataKey="engagement.mean"
                    stroke="#10B981"
                    fill="#10B981"
                    fillOpacity={0.3}
                  />
                  <Area
                    type="monotone"
                    dataKey="confidence.mean"
                    stroke="#3B82F6"
                    fill="#3B82F6"
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </motion.div>

          {/* Current Analysis */}
          <AnimatePresence>
            {currentAnalysis && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="grid grid-cols-1 md:grid-cols-2 gap-6"
              >
                {/* Emotion Analysis */}
                <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-white/10">
                  <h3 className="text-lg font-semibold mb-4 text-cyan-400">Emotion Analysis</h3>
                  <div className="space-y-3">
                    {currentAnalysis.emotion?.probabilities?.map((prob, idx) => (
                      <div key={idx} className="flex items-center justify-between">
                        <span className="capitalize text-sm">{['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'][idx]}</span>
                        <div className="flex items-center space-x-2">
                          <div className="w-20 bg-gray-700 rounded-full h-2">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${prob * 100}%` }}
                              className="bg-gradient-to-r from-cyan-500 to-violet-500 h-2 rounded-full"
                            />
                          </div>
                          <span className="text-sm w-12 text-right">{(prob * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Intent Analysis */}
                <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-white/10">
                  <h3 className="text-lg font-semibold mb-4 text-violet-400">Intent Analysis</h3>
                  <div className="space-y-3">
                    {currentAnalysis.intent?.probabilities?.map((prob, idx) => (
                      <div key={idx} className="flex items-center justify-between">
                        <span className="capitalize text-sm">{['agreement', 'confusion', 'hesitation', 'confidence', 'neutral'][idx]}</span>
                        <div className="flex items-center space-x-2">
                          <div className="w-20 bg-gray-700 rounded-full h-2">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${prob * 100}%` }}
                              className="bg-gradient-to-r from-violet-500 to-pink-500 h-2 rounded-full"
                            />
                          </div>
                          <span className="text-sm w-12 text-right">{(prob * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Modality Contributions */}
          {currentAnalysis?.modality_importance && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white/5 backdrop-blur-xl rounded-2xl p-6 border border-white/10"
            >
              <h3 className="text-lg font-semibold mb-4 text-pink-400">AI Decision Factors</h3>
              <div className="grid grid-cols-3 gap-4">
                {['Vision', 'Audio', 'Text'].map((modality, idx) => (
                  <div key={modality} className="text-center">
                    <div className="text-2xl mb-2">
                      {modality === 'Vision' ? '👁️' : modality === 'Audio' ? '🎵' : '💬'}
                    </div>
                    <div className="text-sm text-gray-400 mb-2">{modality}</div>
                    <div className="text-xl font-bold text-pink-400">
                      {(currentAnalysis.modality_importance[idx] * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-black/20 backdrop-blur-xl border-t border-white/10 p-4 mt-8">
        <div className="max-w-7xl mx-auto flex justify-between items-center text-sm text-gray-400">
          <div>EMOTIA Advanced v2.0 - Real-time Multi-Modal Intelligence</div>
          <div className="flex items-center space-x-4">
            <span>Privacy Mode: {privacyMode ? 'ON' : 'OFF'}</span>
            <span>WebRTC Active</span>
            <span>WebSocket Connected</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default AdvancedVideoAnalysis;