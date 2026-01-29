import React, { useRef, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Sphere, Line } from '@react-three/drei';
import * as THREE from 'three';
import { motion } from 'framer-motion';

// Emotion space visualization component
function EmotionSpace({ analysisData, isActive }) {
  const meshRef = useRef();
  const pointsRef = useRef();
  const [emotionHistory, setEmotionHistory] = useState([]);

  useEffect(() => {
    if (analysisData && isActive) {
      setEmotionHistory(prev => [...prev.slice(-50), analysisData]);
    }
  }, [analysisData, isActive]);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.005;
    }
  });

  // Convert emotion probabilities to 3D coordinates
  const getEmotionCoordinates = (emotions) => {
    if (!emotions || emotions.length !== 7) return [0, 0, 0];

    // Map emotions to 3D space: valence (x), arousal (y), dominance (z)
    const valence = emotions[3] - emotions[4]; // happy - sad
    const arousal = (emotions[0] + emotions[2] + emotions[5]) - (emotions[1] + emotions[6]); // angry + fear + surprise - disgust - neutral
    const dominance = emotions[6] - (emotions[1] + emotions[2]); // neutral - (disgust + fear)

    return [valence * 2, arousal * 2, dominance * 2];
  };

  return (
    <group ref={meshRef}>
      {/* Emotion axes */}
      <Line points={[[-3, 0, 0], [3, 0, 0]]} color="red" lineWidth={2} />
      <Line points={[[0, -3, 0], [0, 3, 0]]} color="green" lineWidth={2} />
      <Line points={[[0, 0, -3], [0, 0, 3]]} color="blue" lineWidth={2} />

      {/* Axis labels */}
      <Text position={[3.2, 0, 0]} fontSize={0.3} color="red">Valence</Text>
      <Text position={[0, 3.2, 0]} fontSize={0.3} color="green">Arousal</Text>
      <Text position={[0, 0, 3.2]} fontSize={0.3} color="blue">Dominance</Text>

      {/* Current emotion point */}
      {analysisData && (
        <Sphere
          args={[0.1, 16, 16]}
          position={getEmotionCoordinates(analysisData.emotion?.probabilities)}
        >
          <meshStandardMaterial
            color={new THREE.Color().setHSL(
              analysisData.emotion?.probabilities?.indexOf(Math.max(...analysisData.emotion.probabilities)) / 7,
              0.8,
              0.6
            )}
            emissive={new THREE.Color(0.1, 0.1, 0.1)}
          />
        </Sphere>
      )}

      {/* Emotion trajectory */}
      {emotionHistory.length > 1 && (
        <Line
          points={emotionHistory.map(data => getEmotionCoordinates(data.emotion?.probabilities))}
          color="cyan"
          lineWidth={3}
        />
      )}

      {/* Emotion labels at corners */}
      <Text position={[2, 2, 2]} fontSize={0.2} color="yellow">Happy</Text>
      <Text position={[-2, -2, -2]} fontSize={0.2} color="purple">Sad</Text>
      <Text position={[2, -2, 0]} fontSize={0.2} color="orange">Angry</Text>
      <Text position={[-2, 2, 0]} fontSize={0.2} color="pink">Surprised</Text>
    </group>
  );
}

// Intent visualization component
function IntentVisualization({ analysisData, isActive }) {
  const groupRef = useRef();
  const [intentHistory, setIntentHistory] = useState([]);

  useEffect(() => {
    if (analysisData && isActive) {
      setIntentHistory(prev => [...prev.slice(-30), analysisData]);
    }
  }, [analysisData, isActive]);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.z += 0.01;
    }
  });

  // Convert intent to radial coordinates
  const getIntentPosition = (intent, index) => {
    const angle = (index / 5) * Math.PI * 2;
    const radius = intent * 2;
    return [Math.cos(angle) * radius, Math.sin(angle) * radius, 0];
  };

  return (
    <group ref={groupRef}>
      {/* Intent radar chart */}
      {analysisData?.intent?.probabilities?.map((prob, idx) => (
        <Sphere
          key={idx}
          args={[prob * 0.3, 8, 8]}
          position={getIntentPosition(prob, idx)}
        >
          <meshStandardMaterial
            color={new THREE.Color().setHSL(idx / 5, 0.7, 0.5)}
            emissive={new THREE.Color(0.05, 0.05, 0.05)}
          />
        </Sphere>
      ))}

      {/* Intent labels */}
      {['Agreement', 'Confusion', 'Hesitation', 'Confidence', 'Neutral'].map((intent, idx) => {
        const angle = (idx / 5) * Math.PI * 2;
        const x = Math.cos(angle) * 2.5;
        const y = Math.sin(angle) * 2.5;
        return (
          <Text
            key={intent}
            position={[x, y, 0]}
            fontSize={0.15}
            color="white"
            anchorX="center"
            anchorY="middle"
          >
            {intent}
          </Text>
        );
      })}

      {/* Connecting lines */}
      {analysisData?.intent?.probabilities && (
        <Line
          points={[
            ...analysisData.intent.probabilities.map((prob, idx) => getIntentPosition(prob, idx)),
            getIntentPosition(analysisData.intent.probabilities[0], 0) // Close the shape
          ]}
          color="lime"
          lineWidth={2}
        />
      )}
    </group>
  );
}

// Modality fusion visualization
function ModalityFusion({ analysisData, isActive }) {
  const fusionRef = useRef();

  useFrame((state) => {
    if (fusionRef.current) {
      fusionRef.current.rotation.x += 0.005;
      fusionRef.current.rotation.y += 0.003;
    }
  });

  return (
    <group ref={fusionRef}>
      {/* Vision sphere */}
      <Sphere args={[0.5, 16, 16]} position={[-2, 0, 0]}>
        <meshStandardMaterial
          color="blue"
          emissive={new THREE.Color(0.1, 0.1, 0.3)}
          transparent
          opacity={analysisData?.modality_importance?.[0] || 0.3}
        />
      </Sphere>

      {/* Audio sphere */}
      <Sphere args={[0.5, 16, 16]} position={[0, 2, 0]}>
        <meshStandardMaterial
          color="green"
          emissive={new THREE.Color(0.1, 0.3, 0.1)}
          transparent
          opacity={analysisData?.modality_importance?.[1] || 0.3}
        />
      </Sphere>

      {/* Text sphere */}
      <Sphere args={[0.5, 16, 16]} position={[2, 0, 0]}>
        <meshStandardMaterial
          color="red"
          emissive={new THREE.Color(0.3, 0.1, 0.1)}
          transparent
          opacity={analysisData?.modality_importance?.[2] || 0.3}
        />
      </Sphere>

      {/* Fusion center */}
      <Sphere args={[0.3, 16, 16]} position={[0, 0, 0]}>
        <meshStandardMaterial
          color="white"
          emissive={new THREE.Color(0.2, 0.2, 0.2)}
        />
      </Sphere>

      {/* Connection lines */}
      <Line points={[[-2, 0, 0], [0, 0, 0]]} color="cyan" lineWidth={3} />
      <Line points={[[0, 2, 0], [0, 0, 0]]} color="cyan" lineWidth={3} />
      <Line points={[[2, 0, 0], [0, 0, 0]]} color="cyan" lineWidth={3} />

      {/* Labels */}
      <Text position={[-2, -1, 0]} fontSize={0.2} color="blue">Vision</Text>
      <Text position={[0, 3, 0]} fontSize={0.2} color="green">Audio</Text>
      <Text position={[2, -1, 0]} fontSize={0.2} color="red">Text</Text>
      <Text position={[0, -1.5, 0]} fontSize={0.25} color="white">Fusion</Text>
    </group>
  );
}

// Main 3D visualization component
export default function Advanced3DVisualization({ analysisData, isActive }) {
  const [activeView, setActiveView] = useState('emotion');

  return (
    <div className="w-full h-96 bg-black/50 rounded-2xl overflow-hidden border border-white/10">
      {/* View Controls */}
      <div className="absolute top-4 left-4 z-10 flex space-x-2">
        {[
          { key: 'emotion', label: 'Emotion Space', icon: '🧠' },
          { key: 'intent', label: 'Intent Radar', icon: '🎯' },
          { key: 'fusion', label: 'Modality Fusion', icon: '🔗' }
        ].map(({ key, label, icon }) => (
          <motion.button
            key={key}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setActiveView(key)}
            className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeView === key
                ? 'bg-cyan-600 text-white'
                : 'bg-white/10 text-gray-300 hover:bg-white/20'
            }`}
          >
            {icon} {label}
          </motion.button>
        ))}
      </div>

      {/* 3D Canvas */}
      <Canvas camera={{ position: [5, 5, 5], fov: 60 }}>
        <ambientLight intensity={0.4} />
        <pointLight position={[10, 10, 10]} intensity={0.8} />
        <pointLight position={[-10, -10, -10]} intensity={0.3} />

        <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />

        {activeView === 'emotion' && (
          <EmotionSpace analysisData={analysisData} isActive={isActive} />
        )}
        {activeView === 'intent' && (
          <IntentVisualization analysisData={analysisData} isActive={isActive} />
        )}
        {activeView === 'fusion' && (
          <ModalityFusion analysisData={analysisData} isActive={isActive} />
        )}
      </Canvas>

      {/* Info Panel */}
      <div className="absolute bottom-4 right-4 bg-black/70 backdrop-blur-sm rounded-lg p-3 text-sm">
        <div className="text-cyan-400 font-semibold mb-2">3D Analysis</div>
        <div className="text-gray-300">
          {activeView === 'emotion' && 'Visualizing emotion in 3D valence-arousal-dominance space'}
          {activeView === 'intent' && 'Intent analysis as radar chart with temporal tracking'}
          {activeView === 'fusion' && 'Multi-modal fusion showing contribution weights'}
        </div>
        <div className="text-xs text-gray-400 mt-1">
          Drag to rotate • Scroll to zoom • Right-click to pan
        </div>
      </div>
    </div>
  );
}