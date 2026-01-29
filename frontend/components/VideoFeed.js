import { useEffect, useRef } from 'react';

const VideoFeed = ({ videoRef, canvasRef, isAnalyzing }) => {
  return (
    <div className="relative">
      <video
        ref={videoRef}
        autoPlay
        muted
        className="w-full rounded-lg bg-black"
        style={{ display: isAnalyzing ? 'block' : 'none' }}
      />
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        className="w-full rounded-lg bg-black"
        style={{ display: isAnalyzing ? 'none' : 'block' }}
      />
      {!isAnalyzing && (
        <div className="absolute inset-0 flex items-center justify-center text-gray-400">
          <div className="text-center">
            <div className="text-6xl mb-4">📹</div>
            <p>Click "Start Analysis" to begin</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoFeed;