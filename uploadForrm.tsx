// src/components/UploadForm.jsx

import React, { useState } from 'react';
import { Upload, Sparkles, Download, ArrowRight, Play } from 'lucide-react'; // Import necessary icons
import { makeAuthenticatedRequest } from '../auth';

// You might also want to import other icons from your App.tsx if UploadForm needs them for other sections
// For simplicity, I've only included ones directly related to the upload/display.


interface UploadFormProps {
  youtubeUrl: string;
  setYoutubeUrl: (v: string) => void;
  handleYoutubeImport: () => void;
  onUpload: (file: File) => void;
}

const UploadForm: React.FC<UploadFormProps> = ({
  youtubeUrl,
  setYoutubeUrl,
  handleYoutubeImport,
  onUpload,
}) => {
  const [isImporting, setIsImporting] = useState<boolean>(false);
  const [importError, setImportError] = useState<string | null>(null);

  // Helper function to extract YouTube thumbnail from URL
  function getYoutubeThumbnail(url: string): string | null {
    // Extract the video ID from the URL using regex
    const match = url.match(
      /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/
    );
    if (match && match[1]) {
      return `https://img.youtube.com/vi/${match[1]}/hqdefault.jpg`;
    }
    return null;
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      // Pass file up to parent for processing options modal
      if (file) {
        // Reset the input so user can re-upload the same file if needed
        e.target.value = '';
        // Call parent handler
  if (typeof onUpload === 'function') onUpload(file);
      }
    }
  };

  // No handleGenerateShorts here; upload is handled in parent

  return (
    // This is the content that was previously inside the <div className="min-h-screen bg-dark-300">
    // in the conceptual App.tsx. You will need to adapt the styling and sections
    // to fit how you want your UploadForm to look within your existing App.jsx structure.

    // I'm assuming you want to include the upload section and the display of shorts
    // within this UploadForm component.
    <div className="bg-black text-white min-h-screen font-sans p-8"> {/* Added some basic padding */}
      {/* Upload Section from the conceptual App.tsx */}
      <section id="upload-section" className="py-20 text-center">
        <h2 className="text-3xl md:text-4xl font-bold mb-8">
          Upload Your <span className="gradient-text">Video</span>
        </h2>
        <div className="max-w-xl mx-auto p-8 card-gradient rounded-2xl">
          <div className="mb-8">
            <div className="flex flex-col sm:flex-row gap-2 items-center justify-center">
              <input
                type="text"
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
                placeholder="Enter YouTube video URL"
                className={`flex-1 px-4 py-3 rounded-lg border transition-colors ${
                  'bg-gray-800 border-gray-600 text-white placeholder-gray-400'
                }`}
              />
              <button
                onClick={handleYoutubeImport}
                className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-6 py-3 rounded-lg font-semibold hover:shadow-lg transition-all duration-300 flex items-center space-x-2"
              >
                {isImporting ? 'Importing...' : 'Import from YouTube'}
              </button>
            </div>
            {importError && <div className="text-red-500 mt-2 text-sm">{importError}</div>}
          </div>
          {/* No sentiment modal here; handled in parent */}
          <input
            type="file"
            accept="video/mp4"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-400
                       file:mr-4 file:py-2 file:px-4
                       file:rounded-full file:border-0
                       file:text-sm file:font-semibold
                       file:bg-primary file:text-black
                       hover:file:bg-primary-dark cursor-pointer mb-6"
          />
          {/* No selected file display here; handled in parent/modal */}
          {/* No aspect ratio selection here; handled in modal */}
          {/* No generate button here; handled in modal */}

          {youtubeUrl && getYoutubeThumbnail(youtubeUrl) && (
            <div className="flex justify-center mt-4">
              <img
                src={getYoutubeThumbnail(youtubeUrl) || undefined}
                alt="YouTube thumbnail"
                className="w-48 h-28 rounded-lg object-cover border"
              />
            </div>
          )}
        </div>
      </section>

      {/* You can add other sections of your original App.tsx here if they are part of the form's layout,
          or keep them in App.jsx if they are part of the main page structure.
          For instance, if the "How it Works" or "Features" sections are generic informational,
          they could stay in App.jsx. But if they're closely tied to the upload process,
          you might want them here. */}

      {/* Example: Hero section might remain in App.jsx, but if UploadForm is your main view,
          you could integrate it here too. */}

      {/* This is a placeholder for where you might put other content from your original App.tsx */}
      {/* <section className="py-20">
        <h2 className="text-3xl text-center">Other Content Here</h2>
      </section> */}
    </div>
  );
};

export default UploadForm;

// from moviepy.editor import VideoFileClip

// def crop_aspect_ratio(clip, aspect_ratio='9:16'):
//     # aspect_ratio: string like '9:16', '16:9', '1:1'
//     w, h = clip.size
//     if aspect_ratio == '9:16':
//         target_ratio = 9 / 16
//     elif aspect_ratio == '16:9':
