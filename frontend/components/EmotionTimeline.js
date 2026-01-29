import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer } from 'recharts';

const EmotionTimeline = ({ history }) => {
  // Prepare data for chart
  const chartData = history.map((item, index) => ({
    time: index,
    engagement: item.engagement * 100,
    confidence: item.confidence * 100,
    emotion: Object.entries(item.emotion.predictions).reduce((a, b) => a[1] > b[1] ? a : b)[1] * 100
  }));

  const emotionColors = {
    happy: '#10B981',
    sad: '#3B82F6',
    angry: '#EF4444',
    fear: '#8B5CF6',
    surprise: '#F59E0B',
    disgust: '#6B7280',
    neutral: '#9CA3AF'
  };

  return (
    <div className="space-y-4">
      {/* Current Emotion Display */}
      {history.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center p-4 bg-gray-700 rounded-lg"
        >
          <div className="text-3xl mb-2">
            {history[history.length - 1].emotion.dominant === 'happy' && '😊'}
            {history[history.length - 1].emotion.dominant === 'sad' && '😢'}
            {history[history.length - 1].emotion.dominant === 'angry' && '😠'}
            {history[history.length - 1].emotion.dominant === 'fear' && '😨'}
            {history[history.length - 1].emotion.dominant === 'surprise' && '😲'}
            {history[history.length - 1].emotion.dominant === 'disgust' && '🤢'}
            {history[history.length - 1].emotion.dominant === 'neutral' && '😐'}
          </div>
          <p className="text-lg font-semibold capitalize">
            {history[history.length - 1].emotion.dominant}
          </p>
        </motion.div>
      )}

      {/* Timeline Chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <XAxis dataKey="time" />
            <YAxis domain={[0, 100]} />
            <Line
              type="monotone"
              dataKey="engagement"
              stroke="#10B981"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="confidence"
              stroke="#3B82F6"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="emotion"
              stroke="#EF4444"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex justify-center space-x-6 text-sm">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-green-500 rounded mr-2"></div>
          <span>Engagement</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-blue-500 rounded mr-2"></div>
          <span>Confidence</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-red-500 rounded mr-2"></div>
          <span>Emotion Strength</span>
        </div>
      </div>
    </div>
  );
};

export default EmotionTimeline;