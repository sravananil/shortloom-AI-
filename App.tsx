import React, { useState, useRef, useCallback, useEffect } from 'react';
import { isLoggedIn, getCurrentUser, logout, getTokens, makeAuthenticatedRequest } from './auth';
import { Upload, Play, Download, Check, X, Moon, Sun, Zap, Clock, Sparkles, Star, ArrowRight, Menu, ChevronDown, Settings, Heart, Laugh, Smile } from 'lucide-react';
import { FaWhatsapp, FaTelegram, FaTwitter, FaLink } from "react-icons/fa";
import shortloomLogo from './assets/shortloom_logo_blue.png';
import { trackEvent, EventType, initAnalytics } from './analytics';

import AuthSplitScreen from "./components/AuthSplitScreen";
import EnhancedGeneratedShorts from './components/EnhancedGeneratedShorts';

// TypeScript interfaces
interface VideoClip {
  url: string;
  filename: string;
  start_time: number;
  end_time: number;
  duration: number;
  score: number;
  srtUrl?: string;
  quality?: string;
  aspect_ratio?: string;
  emotion_filter?: string;
  title?: string;
  description?: string;
  hashtags?: string[];
  engagement?: number;
}

interface UploadResponse {
  clips: VideoClip[];
  message?: string;
}

interface ProcessingOptions {
  file: File;
  useSentiment: boolean;
  sentiment: string;
  aspectRatio: string;
  quality: string;
}

// Theme context
const ThemeContext = React.createContext<{
  isDark: boolean;
  toggleTheme: () => void;
}>({
  isDark: false,
  toggleTheme: () => {},
});

// VideoProcessingOptions Component
const VideoProcessingOptions: React.FC<{
  uploadedFile?: File;
  youtubeUrl?: string;
  onProcess: (options: ProcessingOptions) => void;
  onCancel: () => void;
  isDark: boolean;
  hideSentiment?: boolean;
}> = ({ uploadedFile, youtubeUrl, onProcess, onCancel, isDark, hideSentiment }) => {
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
      file: uploadedFile!,
      useSentiment,
      sentiment: useSentiment ? selectedSentiment : 'all',
      aspectRatio,
      quality
    };
    onProcess(options);
  };

  const handleProcessYoutube = () => {
    const options: ProcessingOptions = {
      file: undefined as any, // not used for YouTube
      useSentiment: false,
      sentiment: 'all',
      aspectRatio,
      quality
    };
    onProcess(options);
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className={`max-w-2xl w-full rounded-2xl shadow-2xl max-h-[90vh] overflow-y-auto  ${
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
          {/* File Info or YouTube Info */}
          {uploadedFile && (
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
          )}
          {youtubeUrl && (
            <div className={`p-4 rounded-lg ${isDark ? 'bg-gray-700' : 'bg-gray-50'}`}> 
              <div className="flex items-center space-x-3">
                <Play className="w-5 h-5 text-purple-500" />
                <div>
                  <p className={`font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}> 
                    YouTube Import
                  </p>
                  <p className={`text-sm break-all ${isDark ? 'text-gray-400' : 'text-gray-600'}`}> 
                    {youtubeUrl}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Sentiment Analysis Option (hide for YouTube) */}
          {!hideSentiment && (
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
          )}

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
          {youtubeUrl ? (
            <button
              onClick={handleProcessYoutube}
              className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-6 py-3 rounded-lg font-semibold hover:shadow-lg transition-all duration-300 flex items-center space-x-2"
            >
              <span>Import & Generate Shorts</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          ) : (
            <button
              onClick={handleProcess}
              className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-6 py-3 rounded-lg font-semibold hover:shadow-lg transition-all duration-300 flex items-center space-x-2"
            >
              <span>Generate Shorts</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

// Navigation Component (unchanged from your original)
const Navigation: React.FC = () => {
  const { isDark, toggleTheme } = React.useContext(ThemeContext);
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <nav className={`fixed top-0 w-full z-50 transition-all duration-300 ${
      isDark ? 'bg-gray-900/95 border-gray-800' : 'bg-white/95 border-gray-200'
    } backdrop-blur-md border-b`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-3 mb-4">
              <img
                src={shortloomLogo}
                alt="ShortLoom Logo"
                className="w-8 h-8 rounded-full shadow-lg border-2 border-cyan-400 bg-black object-contain"
              />
              <span className="text-xl font-bold bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 bg-clip-text text-transparent drop-shadow-lg"
                style={{ fontFamily: 'Saira Stencil One, Audiowide, sans-serif' }}>
                ShortLoom
              </span>
            </div>
          </div>

          {/* Desktop Menu */}
          <div className="hidden md:flex items-center space-x-8">
            <a href="#home" className={`hover:text-purple-500 transition-colors ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
              Home
            </a>
            <a href="#how-it-works" className={`hover:text-purple-500 transition-colors ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
              How It Works
            </a>
            <a href="#demo" className={`hover:text-purple-500 transition-colors ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
              Demo
            </a>
            <a href="#pricing" className={`hover:text-purple-500 transition-colors ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
              Pricing
            </a>
            <button
              onClick={toggleTheme}
              className={`p-2 rounded-lg transition-colors ${
                isDark ? 'hover:bg-gray-800 text-gray-300' : 'hover:bg-gray-100 text-gray-600'
              }`}
            >
              {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden flex items-center space-x-2">
            <button
              onClick={toggleTheme}
              className={`p-2 rounded-lg transition-colors ${
                isDark ? 'hover:bg-gray-800 text-gray-300' : 'hover:bg-gray-100 text-gray-600'
              }`}
            >
              {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className={`p-2 rounded-lg transition-colors ${
                isDark ? 'hover:bg-gray-800 text-gray-300' : 'hover:bg-gray-100 text-gray-600'
              }`}
            >
              <Menu className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMenuOpen && (
          <div className={`md:hidden border-t ${isDark ? 'border-gray-800 bg-gray-900' : 'border-gray-200 bg-white'}`}>
            <div className="px-2 pt-2 pb-3 space-y-1">
              <a href="#home" className={`block px-3 py-2 rounded-md hover:bg-purple-500/10 ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
                Home
              </a>
              <a href="#how-it-works" className={`block px-3 py-2 rounded-md hover:bg-purple-500/10 ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
                How It Works
              </a>
              <a href="#demo" className={`block px-3 py-2 rounded-md hover:bg-purple-500/10 ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
                Demo
              </a>
              <a href="#pricing" className={`block px-3 py-2 rounded-md hover:bg-purple-500/10 ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
                Pricing
              </a>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

// Hero Section (unchanged from your original)
const Hero: React.FC<{ onScrollToUpload: () => void }> = ({ onScrollToUpload }) => {
  const { isDark } = React.useContext(ThemeContext);

  return (
    <section id="home" className={`pt-20 pb-16 px-4 ${isDark ? 'bg-gray-900' : 'bg-white'}`}>
      <div className="max-w-7xl mx-auto text-center">
        <div className="max-w-4xl mx-auto">
          <h1 className={`text-4xl sm:text-5xl md:text-7xl font-bold mb-6 ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Turn Your Videos Into
            <span className="bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-transparent">
              {' '}Viral Shorts
            </span>
          </h1>
          <p className={`text-xl md:text-2xl mb-8 leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            AI-powered video editing that automatically finds the best moments from your long-form content and creates engaging short clips ready for social media.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <button
              onClick={onScrollToUpload}
              className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-8 py-4 rounded-xl font-semibold text-lg hover:shadow-lg hover:scale-105 transition-all duration-300 flex items-center space-x-2"
            >
              <Upload className="w-5 h-5" />
              <span>Start Creating Shorts</span>
            </button>
            <button className={`px-8 py-4 rounded-xl font-semibold text-lg border-2 transition-all duration-300 flex items-center space-x-2 ${
              isDark 
                ? 'border-gray-600 text-gray-300 hover:border-purple-500 hover:text-purple-400' 
                : 'border-gray-300 text-gray-700 hover:border-purple-500 hover:text-purple-600'
            }`}>
              <Play className="w-5 h-5" />
              <span>Watch Demo</span>
            </button>
          </div>
        </div>
        
        {/* Stats */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="text-center">
            <div className={`text-3xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>10M+</div>
            <div className={`${isDark ? 'text-gray-400' : 'text-gray-600'}`}>Videos Processed</div>
          </div>
          <div className="text-center">
            <div className={`text-3xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>50K+</div>
            <div className={`${isDark ? 'text-gray-400' : 'text-gray-600'}`}>Happy Creators</div>
          </div>
          <div className="text-center">
            <div className={`text-3xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>99.9%</div>
            <div className={`${isDark ? 'text-gray-400' : 'text-gray-600'}`}>Uptime</div>
          </div>
        </div>
      </div>
    </section>
  );
};

// Updated Upload Form Component (simplified to show options modal)
const UploadForm: React.FC<{
  onUpload: (file: File) => void;
  isProcessing: boolean;
  progress: number;
  error: string | null;
  youtubeUrl: string;
  setYoutubeUrl: (url: string) => void;
  handleYoutubeImport: () => void;
  isImporting: boolean;
  importError: string | null;
  token: string | null;
}> = ({ onUpload, isProcessing, progress, error, youtubeUrl, setYoutubeUrl, handleYoutubeImport, isImporting, importError, token }) => {
  const { isDark } = React.useContext(ThemeContext);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      onUpload(files[0]);
    }
  }, [onUpload]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      onUpload(files[0]);
    }
  }, [onUpload]);

  return (
    <section id="demo" className={`py-16 px-4 ${isDark ? 'bg-gray-800' : 'bg-gray-50'}`}>
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <h2 className={`text-4xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Upload Your Video
          </h2>
          <p className={`text-xl ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Drop your video file and let our AI work its magic
          </p>
        </div>

        <div
          className={`relative border-2 border-dashed rounded-2xl p-6 sm:p-12 text-center transition-all duration-300 ${
            dragActive
              ? 'border-purple-500 bg-purple-500/10'
              : error
              ? 'border-red-500'
              : isDark
              ? 'border-gray-600 hover:border-purple-500'
              : 'border-gray-300 hover:border-purple-500'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileSelect}
            className="hidden"
            disabled={isProcessing}
          />

          {!isProcessing ? (
            <>
              <div className={`w-16 h-16 mx-auto mb-6 rounded-full flex items-center justify-center ${
                isDark ? 'bg-gray-700' : 'bg-gray-100'
              }`}>
                <Upload className={`w-8 h-8 ${isDark ? 'text-gray-400' : 'text-gray-500'}`} />
              </div>
              <h3 className={`text-xl font-semibold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Drop your video here
              </h3>
              <p className={`mb-6 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                Or click to browse files (MP4, MOV, AVI supported)
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-6 py-3 rounded-lg font-semibold hover:shadow-lg transition-all duration-300"
              >
                Choose File
              </button>
            </>
          ) : (
            <div className="space-y-4">
              <div className={`w-16 h-16 mx-auto rounded-full flex items-center justify-center ${
                isDark ? 'bg-purple-900' : 'bg-purple-100'
              }`}>
                <Zap className="w-8 h-8 text-purple-500 animate-pulse" />
              </div>
              <h3 className={`text-xl font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                AI is processing your video...
              </h3>
              <div className={`w-full bg-gray-200 rounded-full h-3 ${isDark ? 'bg-gray-700' : ''}`}>
                <div
                  className="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                {progress}% complete
              </p>
            </div>
          )}

          {error && (
            <div className="mt-4 p-4 bg-red-100 border border-red-300 rounded-lg">
              <div className="flex items-center space-x-2 text-red-700">
                <X className="w-5 h-5" />
                <span>{error}</span>
              </div>
            </div>
          )}
        </div>

        {/* YouTube URL import section */}
        <div className="mt-8">
          <h3 className={`text-xl font-semibold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Or import from YouTube
          </h3>
          <p className={`mb-4 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Paste the link to your YouTube video and we'll do the rest.
          </p>
          <div className="flex flex-col sm:flex-row gap-4">
            <input
              type="text"
              value={youtubeUrl}
              onChange={(e) => setYoutubeUrl(e.target.value)}
              placeholder="Enter YouTube video URL"
              className={`flex-1 px-4 py-3 rounded-lg border transition-colors mb-2 sm:mb-0 ${
                isDark 
                  ? 'bg-gray-800 border-gray-600 text-white placeholder-gray-400' 
                  : 'bg-white border-gray-300 text-gray-900'
              }`}
            />
            <button
              onClick={handleYoutubeImport}
              className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-6 py-3 rounded-lg font-semibold hover:shadow-lg transition-all duration-300 flex items-center justify-center space-x-2"
              disabled={isImporting}
            >
              {isImporting ? (
                <>
                  <svg className="animate-spin h-5 w-5 mr-3 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4.293 12.293a1 1 0 011.414 0L12 18.586l6.293-6.293a1 1 0 011.414 1.414l-7 7a1 1 0 01-1.414 0l-7-7a1 1 0 010-1.414z"></path>
                  </svg>
                  <span>Importing...</span>
                </>
              ) : (
                <>
                  <Upload className="w-5 h-5" />
                  <span>Import Video</span>
                </>
              )}
            </button>
          </div>

          {importError && (
            <div className="mt-4 p-4 bg-red-100 border border-red-300 rounded-lg">
              <div className="flex items-center space-x-2 text-red-700">
                <X className="w-5 h-5" />
                <span>{importError}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

// Clip Card Component (unchanged from your original with fullscreen fix)
const ClipCard: React.FC<{ clip: VideoClip }> = ({ clip }) => {
  const { isDark } = React.useContext(ThemeContext);
  const [isHovered, setIsHovered] = useState(false);
  const [copied, setCopied] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Fullscreen aspect ratio handler
  const handleFullscreenChange = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (document.fullscreenElement === video) {
      video.style.width = 'auto';
      video.style.height = '100vh';
      video.style.maxWidth = 'calc(100vh * 9/16)';
      video.style.margin = '0 auto';
      video.style.objectFit = 'contain';
    } else {
      video.style.width = '100%';
      video.style.height = '100%';
      video.style.maxWidth = 'none';
      video.style.margin = '0';
      video.style.objectFit = 'cover';
    }
  }, []);

  useEffect(() => {
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('mozfullscreenchange', handleFullscreenChange);
    document.addEventListener('MSFullscreenChange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
      document.removeEventListener('mozfullscreenchange', handleFullscreenChange);
      document.removeEventListener('MSFullscreenChange', handleFullscreenChange);
    };
  }, [handleFullscreenChange]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-500';
    if (score >= 60) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <div
      className={`rounded-xl overflow-hidden transition-all duration-300 hover:scale-105 ${
        isDark ? 'bg-gray-800 border border-gray-700' : 'bg-white border border-gray-200'
      } shadow-lg hover:shadow-xl`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div 
        className="relative bg-black"
        style={{
          aspectRatio: clip.aspect_ratio || '9 / 16',
          width: '100%',
          maxWidth: '360px',
          margin: 'auto',
          borderRadius: '12px',
          overflow: 'hidden',
        }}
      >
        <video
          ref={videoRef}
          src={`http://localhost:5000${clip.url}`}
          className="w-full h-full"
          muted
          preload="metadata"
          controls
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'contain',
            background: '#000',
            aspectRatio: clip.aspect_ratio || '9/16',
          }}
        >
          {clip.srtUrl && (
            <track
              label="English"
              kind="subtitles"
              srcLang="en"
              src={`http://localhost:5000${clip.srtUrl}`}
              default
            />
          )}
        </video>
        {/* Watermark visual overlay */}
        <div className="absolute bottom-2 right-2 bg-white/70 px-2 py-1 rounded text-xs font-bold text-purple-600 pointer-events-none select-none">
          ShortLoom
        </div>
        <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-sm px-2 py-1 rounded text-white text-sm">
          {formatTime(clip.duration)}
        </div>
        {/* Show processing options on video */}
        <div className="absolute top-3 left-3 flex space-x-1">
          {clip.quality && (
            <div className="bg-blue-500/80 px-1 py-0.5 rounded text-xs text-white">
              {clip.quality}
            </div>
          )}
          {clip.emotion_filter && clip.emotion_filter !== 'all' && (
            <div className="bg-pink-500/80 px-1 py-0.5 rounded text-xs text-white">
              {clip.emotion_filter}
            </div>
          )}
        </div>
      </div>
      
      <div className="p-3 sm:p-4">
        <div className="flex items-center justify-between mb-2 sm:mb-3">
          <h4 className={`font-semibold truncate text-sm sm:text-base ${isDark ? 'text-white' : 'text-gray-900'}`}>
            {clip.filename}
          </h4>
          <div className={`text-xs sm:text-sm font-bold ${getScoreColor(clip.score)}`}>
            {clip.score?.toFixed(1) || '0.0'}%
          </div>
        </div>
        
        <div className={`text-xs sm:text-sm mb-3 sm:mb-4 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          <div className="flex items-center space-x-2 sm:space-x-4">
            <span>Start: {formatTime(clip.start_time)}</span>
            <span>End: {formatTime(clip.end_time)}</span>
          </div>
          {/* Show processing info */}
          <div className="mt-1 text-xs opacity-75">
            {clip.aspect_ratio} • {clip.quality} {clip.emotion_filter && clip.emotion_filter !== 'all' && `• ${clip.emotion_filter}`}
          </div>
        </div>
        
        <button
          onClick={() => {
            const link = document.createElement('a');
            link.href = `http://localhost:5000${clip.url}`;
            link.download = clip.filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
          }}
          className="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white py-1.5 sm:py-2 px-3 sm:px-4 rounded-lg font-semibold hover:shadow-lg transition-all duration-300 flex items-center justify-center space-x-2 text-sm sm:text-base"
        >
          <Download className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
          <span>Download</span>
        </button>

        {/* Share buttons */}
        <div className="flex gap-1 sm:gap-2 mt-2">
          <a href={`https://wa.me/?text=${encodeURIComponent(`http://localhost:5000${clip.url}`)}`} target="_blank" rel="noopener" className="p-1.5 sm:p-2 rounded bg-green-500 text-white text-xs sm:text-sm"><FaWhatsapp /></a>
          <a href={`https://t.me/share/url?url=${encodeURIComponent(`http://localhost:5000${clip.url}`)}`} target="_blank" rel="noopener" className="p-1.5 sm:p-2 rounded bg-blue-500 text-white text-xs sm:text-sm"><FaTelegram /></a>
          <a href={`https://twitter.com/intent/tweet?url=${encodeURIComponent(`http://localhost:5000${clip.url}`)}`} target="_blank" rel="noopener" className="p-1.5 sm:p-2 rounded bg-blue-400 text-white text-xs sm:text-sm"><FaTwitter /></a>
          <button
            onClick={() => {
              navigator.clipboard.writeText(`http://localhost:5000${clip.url}`);
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            }}
            className="p-1.5 sm:p-2 rounded bg-gray-300 text-gray-700 text-xs sm:text-sm"
          >
            <FaLink />
            {copied && <span className="ml-1 sm:ml-2 text-xs text-green-600">Copied!</span>}
          </button>
        </div>
      </div>
    </div>
  );
};

// Generated Shorts Section (unchanged from your original)
const GeneratedShorts: React.FC<{ clips: VideoClip[] }> = ({ clips }) => {
  const { isDark } = React.useContext(ThemeContext);

  if (clips.length === 0) return null;

  return (
    <section className={`py-16 px-4 ${isDark ? 'bg-gray-900' : 'bg-white'}`}>
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h2 className={`text-4xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Your Generated Shorts
          </h2>
          <p className={`text-xl ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            AI found {clips.length} amazing moments in your video
          </p>
        </div>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-8">
          {clips.map((clip, index) => (
            <ClipCard key={index} clip={clip} />
          ))}
        </div>
      </div>
    </section>
  );
};

// How It Works Section (unchanged from your original)
const HowItWorks: React.FC = () => {
  const { isDark } = React.useContext(ThemeContext);

  const steps = [
    {
      icon: Upload,
      title: "Upload Your Video",
      description: "Simply drag and drop your long-form video content into our platform."
    },
    {
      icon: Settings,
      title: "Choose Your Options",
      description: "Select sentiment analysis, aspect ratio, and quality settings."
    },
    {
      icon: Zap,
      title: "AI Processing",
      description: "Our advanced AI analyzes your video to identify the most engaging moments."
    },
    {
      icon: Download,
      title: "Download Shorts",
      description: "Get your optimized short clips ready for social media platforms."
    }
  ];

  return (
    <section id="how-it-works" className={`py-12 sm:py-16 px-4 ${isDark ? 'bg-gray-800' : 'bg-gray-50'}`}>
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-10 sm:mb-16">
          <h2 className={`text-3xl sm:text-4xl font-bold mb-3 sm:mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
            How It Works
          </h2>
          <p className={`text-lg sm:text-xl max-w-2xl mx-auto ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Transform your content in four simple steps
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6 sm:gap-8">
          {steps.map((step, index) => (
            <div key={index} className="text-center">
              <div className={`w-16 h-16 sm:w-20 sm:h-20 mx-auto mb-4 sm:mb-6 rounded-full flex items-center justify-center ${
                isDark ? 'bg-gray-700' : 'bg-white'
              } shadow-lg`}>
                <step.icon className="w-8 h-8 sm:w-10 sm:h-10 text-purple-500" />
              </div>
              <h3 className={`text-lg sm:text-xl font-semibold mb-2 sm:mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                {step.title}
              </h3>
              <p className={`text-sm sm:text-base ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                {step.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

// Features Section (unchanged from your original)
const Features: React.FC = () => {
  const { isDark } = React.useContext(ThemeContext);

  const features = [
    {
      icon: Zap,
      title: "AI-Powered Detection",
      description: "Advanced algorithms identify the most engaging moments automatically"
    },
    {
      icon: Heart,
      title: "Sentiment Analysis",
      description: "Find specific emotional moments like happy, funny, or touching scenes"
    },
    {
      icon: Settings,
      title: "Multiple Formats",
      description: "Export in 9:16, 16:9, or 1:1 aspect ratios with quality options"
    },
    {
      icon: Clock,
      title: "Lightning Fast",
      description: "Process hours of content in minutes with our optimized AI pipeline"
    },
    {
      icon: Star,
      title: "High Quality Output",
      description: "Maintain original video quality while optimizing for social platforms"
    },
    {
      icon: Sparkles,
      title: "Smart Cropping",
      description: "Automatically adjusts aspect ratios for different social media platforms"
    }
  ];

  return (
    <section className={`py-16 px-4 ${isDark ? 'bg-gray-900' : 'bg-white'}`}>
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className={`text-4xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Powerful Features
          </h2>
          <p className={`text-xl max-w-2xl mx-auto ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Everything you need to create viral content
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div key={index} className={`p-6 rounded-xl transition-all duration-300 hover:scale-105 ${
              isDark ? 'bg-gray-800 border border-gray-700' : 'bg-gray-50 hover:bg-white hover:shadow-lg'
            }`}>
              <feature.icon className="w-12 h-12 text-purple-500 mb-4" />
              <h3 className={`text-lg font-semibold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                {feature.title}
              </h3>
              <p className={`${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

// Pricing, Newsletter, Footer sections (unchanged from your original)
const Pricing: React.FC = () => {
  const { isDark } = React.useContext(ThemeContext);

  const plans = [
    {
      name: "Free",
      price: "$0",
      period: "/month",
      features: [
        "5 videos per month",
        "Basic AI processing",
        "720p output quality",
        "Email support"
      ],
      popular: false
    },
    {
      name: "Pro",
      price: "$29",
      period: "/month",
      features: [
        "Unlimited videos",
        "Advanced sentiment analysis",
        "4K output quality",
        "Priority support",
        "Multiple aspect ratios",
        "API access"
      ],
      popular: true
    }
  ];

  return (
    <section id="pricing" className={`py-16 px-4 ${isDark ? 'bg-gray-800' : 'bg-gray-50'}`}>
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-16">
          <h2 className={`text-4xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Choose Your Plan
          </h2>
          <p className={`text-xl ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Start free, upgrade when you need more
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {plans.map((plan, index) => (
            <div
              key={index}
              className={`relative p-8 rounded-2xl transition-all duration-300 hover:scale-105 ${
                plan.popular
                  ? 'bg-gradient-to-br from-purple-500 to-pink-500 text-white'
                  : isDark
                  ? 'bg-gray-900 border border-gray-700'
                  : 'bg-white border border-gray-200'
              } shadow-xl`}
            >
              {plan.popular && (
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <div className="bg-yellow-400 text-black px-4 py-1 rounded-full text-sm font-semibold">
                    Most Popular
                  </div>
                </div>
              )}

              <div className="text-center mb-8">
                <h3 className={`text-2xl font-bold mb-2 ${
                  plan.popular ? 'text-white' : isDark ? 'text-white' : 'text-gray-900'
                }`}>
                  {plan.name}
                </h3>
                <div className="flex items-center justify-center">
                  <span className={`text-5xl font-bold ${
                    plan.popular ? 'text-white' : isDark ? 'text-white' : 'text-gray-900'
                  }`}>
                    {plan.price}
                  </span>
                  <span className={`ml-2 ${
                    plan.popular ? 'text-white/80' : isDark ? 'text-gray-400' : 'text-gray-600'
                  }`}>
                    {plan.period}
                  </span>
                </div>
              </div>

              <ul className="space-y-4 mb-8">
                {plan.features.map((feature, featureIndex) => (
                  <li key={featureIndex} className="flex items-center space-x-3">
                    <Check className={`w-5 h-5 ${
                      plan.popular ? 'text-white' : 'text-green-500'
                    }`} />
                    <span className={
                      plan.popular ? 'text-white' : isDark ? 'text-gray-300' : 'text-gray-700'
                    }>
                      {feature}
                    </span>
                  </li>
                ))}
              </ul>

              <button className={`w-full py-3 px-6 rounded-lg font-semibold transition-all duration-300 ${
                plan.popular
                  ? 'bg-white text-purple-600 hover:bg-gray-100'
                  : isDark
                  ? 'bg-purple-600 text-white hover:bg-purple-700'
                  : 'bg-gray-900 text-white hover:bg-gray-800'
              }`}>
                Get Started
              </button>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

const Newsletter: React.FC = () => {
  const { isDark } = React.useContext(ThemeContext);
  const [email, setEmail] = useState('');

  return (
    <section className={`py-16 px-4 ${isDark ? 'bg-gray-900' : 'bg-white'}`}>
      <div className="max-w-4xl mx-auto text-center">
        <h2 className={`text-4xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
          Stay Updated
        </h2>
        <p className={`text-xl mb-8 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          Get the latest updates on new features and AI improvements
        </p>
        
        <div className="flex flex-col sm:flex-row gap-4 max-w-md mx-auto">
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter your email"
            className={`flex-1 px-4 py-3 rounded-lg border transition-colors ${
              isDark 
                ? 'bg-gray-800 border-gray-600 text-white placeholder-gray-400' 
                : 'bg-white border-gray-300 text-gray-900'
            }`}
          />
          <button className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-6 py-3 rounded-lg font-semibold hover:shadow-lg transition-all duration-300 flex items-center space-x-2">
            <span>Subscribe</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </section>
  );
};

const Footer: React.FC = () => {
  const { isDark } = React.useContext(ThemeContext);

  return (
    <footer className={`py-12 px-4 border-t ${isDark ? 'bg-gray-900 border-gray-800' : 'bg-gray-50 border-gray-200'}`}>
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-4 gap-8">
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <span className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                ShortLoom
              </span>
            </div>
            <p className={`${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              AI-powered video editing for the modern creator.
            </p>
          </div>
          
          <div>
            <h4 className={`font-semibold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
              Product
            </h4>
            <ul className="space-y-2">
              <li><a href="#" className={`hover:text-purple-500 transition-colors ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>Features</a></li>
              <li><a href="#" className={`hover:text-purple-500 transition-colors ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>Pricing</a></li>
              <li><a href="#" className={`hover:text-purple-500 transition-colors ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>API</a></li>
            </ul>
          </div>
          
          <div>
            <h4 className={`font-semibold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
              Company
            </h4>
            <ul className="space-y-2">
              <li><a href="#" className={`hover:text-purple-500 transition-colors ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>About</a></li>
              <li><a href="#" className={`hover:text-purple-500 transition-colors ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>Blog</a></li>
              <li><a href="#" className={`hover:text-purple-500 transition-colors ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>Careers</a></li>
            </ul>
          </div>
          
          <div>
            <h4 className={`font-semibold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
              Support
            </h4>
            <ul className="space-y-2">
              <li><a href="#" className={`hover:text-purple-500 transition-colors ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>Help Center</a></li>
              <li><a href="#" className={`hover:text-purple-500 transition-colors ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>Contact</a></li>
              <li><a href="#" className={`hover:text-purple-500 transition-colors ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>Status</a></li>
            </ul>
          </div>
        </div>
        
        <div className={`mt-8 pt-8 border-t text-center ${isDark ? 'border-gray-800 text-gray-400' : 'border-gray-200 text-gray-600'}`}>
          <p>&copy; 2025 ShortLoom. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

// Main App Component with Processing Options Integration
const ShortLoomApp: React.FC = () => {
  const [isDark, setIsDark] = useState(false);
  const [clips, setClips] = useState<VideoClip[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  // AI content state
  const [aiTitle, setAiTitle] = useState<string | null>(null);
  const [aiDescription, setAiDescription] = useState<string | null>(null);
  const [aiHashtags, setAiHashtags] = useState<string[]>([]);

  // NEW: Processing options modal state
  const [showProcessingOptions, setShowProcessingOptions] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  // For YouTube modal
  const [showYoutubeOptions, setShowYoutubeOptions] = useState(false);
  const [pendingYoutubeUrl, setPendingYoutubeUrl] = useState<string | null>(null);

  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [isImporting, setIsImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);

  const toggleTheme = () => setIsDark(!isDark);

  const scrollToUpload = () => {
    document.getElementById('demo')?.scrollIntoView({ behavior: 'smooth' });
  };

  // UPDATED: Upload now shows options modal instead of processing immediately
  const handleUpload = async (file: File) => {
    setError(null);
    setUploadedFile(file);
    setShowProcessingOptions(true); // Show options modal instead of processing
  };

  // NEW: Show modal for YouTube import options
  const handleYoutubeImportModal = () => {
    setPendingYoutubeUrl(youtubeUrl.trim());
    setShowYoutubeOptions(true);
  };

  // NEW: Process with selected options
  // For YouTube import with options
  const handleProcessYoutubeWithOptions = async (options: ProcessingOptions) => {
    setShowYoutubeOptions(false);
    setIsImporting(true);
    setImportError(null);
    try {
      const response = await fetch('http://localhost:5000/youtube-upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: pendingYoutubeUrl,
          aspect_ratio: options.aspectRatio,
          quality: options.quality
        }),
      });
      if (!response.ok) {
        throw new Error('Failed to import video. Please check the link or try again.');
      }
      const data = await response.json();
      setClips(data.clips || []);
      // Set AI content if present
      setAiTitle(data.title || null);
      setAiDescription(data.description || null);
      setAiHashtags(Array.isArray(data.hashtags) ? data.hashtags : []);
      localStorage.setItem('shortloom_clips', JSON.stringify(data.clips || []));
      trackEvent(EventType.CLIP_GENERATED, { 
        clipCount: (data.clips || []).length,
        source: 'youtube',
        aspectRatio: options.aspectRatio,
        quality: options.quality
      });
      setTimeout(() => {
        const resultsSection = document.querySelector('#generated-shorts');
        resultsSection?.scrollIntoView({ behavior: 'smooth' });
      }, 500);
    } catch (err) {
      setImportError(err instanceof Error ? err.message : 'Import failed. Please try again.');
      trackEvent(EventType.ERROR, { 
        type: 'youtube_import_failed',
        message: err instanceof Error ? err.message : 'Unknown error',
        context: 'youtube_import_with_options'
      });
    } finally {
      setIsImporting(false);
      setPendingYoutubeUrl(null);
    }
  };

  // For file upload
  const handleProcessWithOptions = async (options: ProcessingOptions) => {
    setShowProcessingOptions(false);
    setIsProcessing(true);
    setProgress(0);
    setError(null);
    setClips([]);

    // Track upload event
    trackEvent(EventType.VIDEO_UPLOAD, { 
      fileName: options.file.name,
      fileSize: options.file.size,
      fileType: options.file.type,
      sentiment: options.sentiment,
      aspectRatio: options.aspectRatio,
      quality: options.quality
    });

    try {
      const formData = new FormData();
      formData.append('video', options.file);
      formData.append('aspect_ratio', options.aspectRatio);
      formData.append('quality', options.quality);
      formData.append('emotion_filter', options.sentiment);

      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + Math.random() * 10;
        });
      }, 500);

      const response = await makeAuthenticatedRequest('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setProgress(100);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Server error details:', errorText);
        throw new Error(`Upload failed: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      console.log('Backend response:', data);

      let processedClips: VideoClip[] = [];

      if (Array.isArray(data)) {
        processedClips = data;
      } else if (data.clips && Array.isArray(data.clips)) {
        processedClips = data.clips;
      } else if (data.data && Array.isArray(data.data)) {
        processedClips = data.data;
      } else {
        // Fallback with processing options info
        processedClips = [
          {
            url: "/static/clips/demo_clip_1.mp4",
            filename: "demo_clip_1.mp4",
            start_time: 10.5,
            end_time: 18.3,
            duration: 7.8,
            score: 88.2,
            quality: options.quality,
            aspect_ratio: options.aspectRatio,
            emotion_filter: options.sentiment
          },
          {
            url: "/static/clips/demo_clip_2.mp4",
            filename: "demo_clip_2.mp4",
            start_time: 25.1,
            end_time: 32.4,
            duration: 7.3,
            score: 92.5,
            quality: options.quality,
            aspect_ratio: options.aspectRatio,
            emotion_filter: options.sentiment
          }
        ];

        console.log('Using demo clips with processing options:', options);
        trackEvent(EventType.ERROR, { 
          type: 'unexpected_response_format',
          context: 'upload_with_options'
        });
      }

      setClips(processedClips);
      // Set AI content if present
      setAiTitle(data.title || null);
      setAiDescription(data.description || null);
      setAiHashtags(Array.isArray(data.hashtags) ? data.hashtags : []);
      localStorage.setItem('shortloom_clips', JSON.stringify(processedClips));
      
      trackEvent(EventType.CLIP_GENERATED, { 
        clipCount: processedClips.length,
        source: 'upload',
        sentiment: options.sentiment,
        aspectRatio: options.aspectRatio,
        quality: options.quality
      });
      
      setTimeout(() => {
        const resultsSection = document.querySelector('#generated-shorts');
        resultsSection?.scrollIntoView({ behavior: 'smooth' });
      }, 500);

    } catch (err) {
      console.error('Upload error:', err);
      setError(err instanceof Error ? err.message : 'Upload failed. Please try again.');
      
      trackEvent(EventType.ERROR, { 
        type: 'upload_failed',
        message: err instanceof Error ? err.message : 'Unknown error',
        context: 'upload_with_options'
      });

      // Show demo clips on error for testing UI
      const demoClips: VideoClip[] = [
        {
          url: "/static/clips/demo_clip_1.mp4",
          filename: "demo_clip_1.mp4",
          start_time: 10.5,
          end_time: 18.3,
          duration: 7.8,
          score: 88.2,
          quality: options.quality,
          aspect_ratio: options.aspectRatio,
          emotion_filter: options.sentiment
        }
      ];
      setClips(demoClips);
      setAiTitle(null);
      setAiDescription(null);
      setAiHashtags([]);
    } finally {
      setIsProcessing(false);
    }
  };

  // NEW: Cancel processing options
  const handleCancelProcessing = () => {
    setShowProcessingOptions(false);
    setUploadedFile(null);
  };

  const handleYoutubeImport = async () => {
    setIsImporting(true);
    setImportError(null);

    trackEvent(EventType.VIDEO_UPLOAD, { 
      source: 'youtube',
      url: youtubeUrl.trim()
    });

    const ytRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+$/;
    if (!ytRegex.test(youtubeUrl.trim())) {
      setImportError('Please enter a valid YouTube link.');
      setIsImporting(false);
      
      trackEvent(EventType.ERROR, { 
        type: 'youtube_validation_failed',
        url: youtubeUrl.trim(),
        context: 'youtube_import'
      });
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/youtube-upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: youtubeUrl.trim() }),
      });
      if (!response.ok) {
        throw new Error('Failed to import video. Please check the link or try again.');
      }
      const data = await response.json();
      setClips(data.clips || []);
      localStorage.setItem('shortloom_clips', JSON.stringify(data.clips || []));
      
      trackEvent(EventType.CLIP_GENERATED, { 
        clipCount: (data.clips || []).length,
        source: 'youtube'
      });
      
      setTimeout(() => {
        const resultsSection = document.querySelector('#generated-shorts');
        resultsSection?.scrollIntoView({ behavior: 'smooth' });
      }, 500);
    } catch (err) {
      setImportError(err instanceof Error ? err.message : 'Import failed. Please try again.');
      
      trackEvent(EventType.ERROR, { 
        type: 'youtube_import_failed',
        message: err instanceof Error ? err.message : 'Unknown error',
        context: 'youtube_import'
      });
    } finally {
      setIsImporting(false);
    }
  };

  // Load saved clips from localStorage on initial render
  useEffect(() => {
    const savedClips = localStorage.getItem('shortloom_clips');
    if (savedClips) {
      setClips(JSON.parse(savedClips));
    }
  }, []);

  // Authentication state using new auth utility
  const [authChecked, setAuthChecked] = useState(false);
  const [user, setUser] = useState(getCurrentUser());
  const [token, setToken] = useState<string | null>(getTokens()?.access_token || null);

  useEffect(() => {
    if (isLoggedIn()) {
      setUser(getCurrentUser());
      setToken(getTokens()?.access_token || null);
    } else {
      setUser(null);
      setToken(null);
    }
    setAuthChecked(true);
  }, []);

  const handleLoginSuccess = () => {
    setUser(getCurrentUser());
    setToken(getTokens()?.access_token || null);
  trackEvent(EventType.USER_LOGIN, { 
      action: 'login',
      success: true
    });
  };

  // Set up body classes for proper theming
  React.useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
      document.body.style.backgroundColor = '#111827';
    } else {
      document.documentElement.classList.remove('dark');
      document.body.style.backgroundColor = '#ffffff';
    }
  }, [isDark]);

  React.useEffect(() => {
    initAnalytics();
  }, []);

  if (!authChecked) {
    return null;
  }
  if (!isLoggedIn()) {
    return <AuthSplitScreen onLogin={handleLoginSuccess} />;
  }

  return (
    <ThemeContext.Provider value={{ isDark, toggleTheme }}>
      <div className={`min-h-screen transition-colors duration-300 overflow-x-hidden ${
        isDark ? 'bg-gray-900 text-white' : 'bg-white text-gray-900'
      }`}>
        <Navigation />
        <Hero onScrollToUpload={scrollToUpload} />
        <UploadForm 
          onUpload={handleUpload}
          isProcessing={isProcessing}
          progress={progress}
          error={error}
          youtubeUrl={youtubeUrl}
          setYoutubeUrl={setYoutubeUrl}
          // Instead of direct import, show modal for options
          handleYoutubeImport={handleYoutubeImportModal}
          isImporting={isImporting}
          importError={importError}
          token={token}
        />

        {/* NEW: Processing Options Modal for file upload */}
        {showProcessingOptions && uploadedFile && (
          <VideoProcessingOptions
            uploadedFile={uploadedFile}
            onProcess={handleProcessWithOptions}
            onCancel={handleCancelProcessing}
            isDark={isDark}
          />
        )}
        {/* NEW: Processing Options Modal for YouTube import (sentiment hidden) */}
        {showYoutubeOptions && pendingYoutubeUrl && (
          <VideoProcessingOptions
            youtubeUrl={pendingYoutubeUrl}
            hideSentiment={true}
            onProcess={handleProcessYoutubeWithOptions}
            onCancel={() => { setShowYoutubeOptions(false); setPendingYoutubeUrl(null); }}
            isDark={isDark}
          />
        )}

        <div id="generated-shorts">
          <EnhancedGeneratedShorts
            clips={clips.map((clip) => ({
              ...clip,
              title: clip.title || aiTitle || 'ShortLoom Short',
              description: clip.description || aiDescription || 'Enjoy this moment!',
              hashtags: clip.hashtags || aiHashtags || ['#ShortLoom', '#Shorts'],
              engagement: typeof clip.engagement === 'object' && clip.engagement !== null
                ? clip.engagement
                : {
                    predictedViews: Math.floor((clip.score || 80) * 100),
                    predictedLikes: Math.floor((clip.score || 80) * 10),
                    predictedShares: Math.floor((clip.score || 80)),
                    viralPotential: Math.round(clip.score || 80),
                  },
            }))}
          />
        </div>
        <HowItWorks />
        <Features />
        <Pricing />
        <Newsletter />
        <Footer />
      </div>
    </ThemeContext.Provider>
  );
};

export default ShortLoomApp;
