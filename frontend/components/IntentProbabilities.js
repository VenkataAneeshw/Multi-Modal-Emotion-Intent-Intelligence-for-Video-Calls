import { motion } from 'framer-motion';

const IntentProbabilities = ({ probabilities }) => {
  if (!probabilities) return null;

  const intents = Object.entries(probabilities).map(([intent, prob]) => ({
    name: intent,
    value: prob,
    color: getIntentColor(intent)
  }));

  function getIntentColor(intent) {
    const colors = {
      agreement: 'bg-green-500',
      confusion: 'bg-red-500',
      hesitation: 'bg-yellow-500',
      confidence: 'bg-blue-500',
      neutral: 'bg-gray-500'
    };
    return colors[intent] || 'bg-gray-500';
  }

  return (
    <div className="bg-gray-800 rounded-xl p-4 glassmorphism">
      <h2 className="text-xl font-semibold mb-4 text-lime-400">Intent Probabilities</h2>
      <div className="space-y-3">
        {intents.map((intent, index) => (
          <div key={intent.name}>
            <div className="flex justify-between text-sm mb-1">
              <span className="capitalize">{intent.name}</span>
              <span>{(intent.value * 100).toFixed(1)}%</span>
            </div>
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${intent.value * 100}%` }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className={`h-3 ${intent.color} rounded-full`}
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default IntentProbabilities;