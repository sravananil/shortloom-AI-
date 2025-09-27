import React, { useState } from 'react';
import { Play, Settings, Heart, Laugh, Smile, ArrowRight, X } from 'lucide-react';

interface VideoProcessingOptionsProps {
  uploadedFile: File;
  onProcess: (options: ProcessingOptions) => void;
  onCancel: () => void;
  isDark: boolean;
}

export interface ProcessingOptions {
  file: File;
  useSentiment: boolean;
  sentiment: string;
  aspectRatio: string;
  quality: string;
}

const VideoProcessingOptions: React.FC<VideoProcessingOptionsProps> = ({
  uploadedFile,
  onProcess,
  onCancel,
  isDark
}) => {
  const [useSentiment, setUseSentiment] = useState(false);
  const [selectedSentiment, setSelectedSentiment] = useState('happy');
  const [aspectRatio, setAspectRatio] = useState('9:16');
  const [quality, setQuality] = useState('1080p');

  const sentimentOptions = [
    { value: 'happy', label: 'Happy & Joyful', icon: Smile, color: 'text-yellow-500' },
    { value: 'funny', label: 'Funny & Comedy', icon: Laugh, color: 'text-green-500' },
    { value: 'emotional', label: 'Emotional & Touching', icon: Heart, color: 'text-pink-500' }
  ];

  const aspectRatioOptions = [
    { value: '9:16', label: 'Vertical (9:16)', description: 'Perfect for TikTok, Instagram Reels' },
    { value: '16:9', label: 'Horizontal (16:9)', description: 'Great for YouTube, Facebook' },
    { value: '1:1', label: 'Square (1:1)', description: 'Ideal for Instagram Posts' }
  ];

  const qualityOptions = [
    { value: '720p', label: '720p HD', description: 'Good quality, smaller file size' },
    { value: '1080p', label: '1080p Full HD', description: 'High quality, balanced size' },
    { value: '4k', label: '4K Ultra HD', description: 'Best quality, larger file size' }
  ];

  const handleProcess = () => {
    const options: ProcessingOptions = {
      file: uploadedFile,
      useSentiment,
      sentiment: useSentiment ? selectedSentiment : 'all',
      aspectRatio,
      quality
    };
    onProcess(options);
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className={`max-w-2xl w-full rounded-2xl shadow-2xl max-h-[90vh] overflow-y-auto ${
        isDark ? 'bg-gray-800 border border-gray-700' : 'bg-white border border-gray-200'
      }`}>
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div>
            <h2 className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
              Processing Options
            </h2>
            <p className={`text-sm mt-1 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              Customize how you want your video processed
            </p>
          </div>
          <button
            onClick={onCancel}
            className={`p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 ${
              isDark ? 'text-gray-400' : 'text-gray-600'
            }`}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-8">
          {/* File Info */}
          <div className={`p-4 rounded-lg ${isDark ? 'bg-gray-700' : 'bg-gray-50'}`}>
            <div className="flex items-center space-x-3">
              <Play className="w-5 h-5 text-purple-500" />
              <div>
                <p className={`font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  {uploadedFile.name}
                </p>
                <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                  {(uploadedFile.size / (1024 * 1024)).toFixed(1)} MB
                </p>
              </div>
            </div>
          </div>

          {/* Sentiment Analysis Option */}
          <div className="space-y-4">
            <h3 className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
              Content Analysis
            </h3>
            
            <div className="space-y-3">
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="radio"
                  name="sentiment-option"
                  checked={!useSentiment}
                  onChange={() => setUseSentiment(false)}
                  className="w-4 h-4 text-purple-600"
                />
                <div>
                  <span className={`font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                    Proceed without Sentiment Analysis
                  </span>
                  <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                    Find the best moments based on faces, motion, and overall quality
                  </p>
                </div>
              </label>

              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="radio"
                  name="sentiment-option"
                  checked={useSentiment}
                  onChange={() => setUseSentiment(true)}
                  className="w-4 h-4 text-purple-600"
                />
                <div>
                  <span className={`font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                    Process with Sentiment Analysis
                  </span>
                  <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                    Find moments that match specific emotions and feelings
                  </p>
                </div>
              </label>
            </div>

            {/* Sentiment Selection */}
            {useSentiment && (
              <div className={`mt-4 p-4 rounded-lg border-2 border-dashed ${
                isDark ? 'border-gray-600 bg-gray-700/50' : 'border-gray-300 bg-gray-50/50'
              }`}>
                <p className={`text-sm font-medium mb-3 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                  Select the type of moments you want to find:
                </p>
                <div className="grid grid-cols-1 gap-3">
                  {sentimentOptions.map((option) => (
                    <label key={option.value} className="flex items-center space-x-3 cursor-pointer">
                      <input
                        type="radio"
                        name="sentiment-type"
                        value={option.value}
                        checked={selectedSentiment === option.value}
                        onChange={(e) => setSelectedSentiment(e.target.value)}
                        className="w-4 h-4 text-purple-600"
                      />
                      <option.icon className={`w-5 h-5 ${option.color}`} />
                      <span className={`font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                        {option.label}
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Aspect Ratio Selection */}
          <div className="space-y-4">
            <h3 className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
              Aspect Ratio
            </h3>
            <div className="grid grid-cols-1 gap-3">
              {aspectRatioOptions.map((option) => (
                <label key={option.value} className="flex items-center space-x-3 cursor-pointer">
                  <input
                    type="radio"
                    name="aspect-ratio"
                    value={option.value}
                    checked={aspectRatio === option.value}
                    onChange={(e) => setAspectRatio(e.target.value)}
                    className="w-4 h-4 text-purple-600"
                  />
                  <div>
                    <span className={`font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                      {option.label}
                    </span>
                    <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                      {option.description}
                    </p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Quality Selection */}
          <div className="space-y-4">
            <h3 className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
              Export Quality
            </h3>
            <div className="grid grid-cols-1 gap-3">
              {qualityOptions.map((option) => (
                <label key={option.value} className="flex items-center space-x-3 cursor-pointer">
                  <input
                    type="radio"
                    name="quality"
                    value={option.value}
                    checked={quality === option.value}
                    onChange={(e) => setQuality(e.target.value)}
                    className="w-4 h-4 text-purple-600"
                  />
                  <div>
                    <span className={`font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                      {option.label}
                    </span>
                    <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                      {option.description}
                    </p>
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={onCancel}
            className={`px-4 py-2 rounded-lg font-medium ${
              isDark 
                ? 'text-gray-400 hover:text-gray-300 hover:bg-gray-700' 
                : 'text-gray-600 hover:text-gray-700 hover:bg-gray-100'
            }`}
          >
            Cancel
          </button>
          <button
            onClick={handleProcess}
            className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-6 py-3 rounded-lg font-semibold hover:shadow-lg transition-all duration-300 flex items-center space-x-2"
          >
            <span>Generate Shorts</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default VideoProcessingOptions;
