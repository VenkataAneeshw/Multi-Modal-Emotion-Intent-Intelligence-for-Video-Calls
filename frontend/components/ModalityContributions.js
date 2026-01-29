import { motion } from 'framer-motion';

const ModalityContributions = ({ contributions }) => {
  if (!contributions) return null;

  const modalities = [
    { name: 'Vision', value: contributions.vision, color: 'bg-cyan-500' },
    { name: 'Audio', value: contributions.audio, color: 'bg-lime-500' },
    { name: 'Text', value: contributions.text, color: 'bg-violet-500' }
  ];

  return (
    <div className="bg-gray-800 rounded-xl p-4 glassmorphism">
      <h2 className="text-xl font-semibold mb-4 text-cyan-400">Modality Contributions</h2>
      <div className="space-y-3">
        {modalities.map((modality, index) => (
          <div key={modality.name}>
            <div className="flex justify-between text-sm mb-1">
              <span>{modality.name}</span>
              <span>{(modality.value * 100).toFixed(1)}%</span>
            </div>
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${modality.value * 100}%` }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className={`h-3 ${modality.color} rounded-full`}
            />
          </div>
        ))}
      </div>
      <p className="text-xs text-gray-400 mt-3">
        How much each modality influenced the prediction
      </p>
    </div>
  );
};

export default ModalityContributions;