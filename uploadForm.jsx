import React, { useState } from 'react';

const UploadComponent = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [clips, setClips] = useState([]);
  const [title, setTitle] = useState('');
  const [hashtags, setHashtags] = useState([]);

  const handleUpload = async (formData) => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setClips(data.clips);
      setTitle(data.title);         // <-- Show this in your UI
      setHashtags(data.hashtags);   // <-- Show this in your UI
      setLoading(false);
    } catch (err) {
      setError('Upload failed');
      setLoading(false);
    }
  };

  return (
    <div>
      {/* Your existing UI code */}
      {title && (
        <div className="mt-4">
          <h3 className="font-bold">Suggested Title:</h3>
          <p>{title}</p>
        </div>
      )}
      {hashtags && hashtags.length > 0 && (
        <div className="mt-2">
          <h3 className="font-bold">Suggested Hashtags:</h3>
          <p>{hashtags.join(' ')}</p>
        </div>
      )}
    </div>
  );
};

export default UploadComponent;
