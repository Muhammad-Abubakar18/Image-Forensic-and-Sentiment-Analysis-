
// App.js
import React, { useState } from 'react';
import './App.css';
import MetadataPage from './components/MetadataPage';
import Loader from './components/Loader';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageData, setImageData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showResultPage, setShowResultPage] = useState(false);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
    }
  };

  const handleAnalyzeClick = async () => {
    if (selectedImage) {
      setLoading(true);
      const formData = new FormData();
      formData.append('files', selectedImage);

      try {
        const response = await fetch('http://127.0.0.1:8000/process-images', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) throw new Error('Server error');

        const json = await response.json();
        setImageData(json.results[0]);
        setShowResultPage(true);
      } catch (err) {
        alert('Failed to analyze image.');
      } finally {
        setLoading(false);
      }
    }
  };

  if (loading) return <Loader />;

  if (showResultPage && imageData) {
    return <MetadataPage imageData={imageData} />;
  }

  return (
    <div className="app-container">
      <div className="main-content">
        {/* Left Side - Title and Description */}
        <div className="left-section">
          <div className="title-section">
            <h1 className="main-title">
              <span className="title-icon">🔍</span>
              ForensiCam AI Powered Image Detector
            </h1>
            <div className="title-decoration"></div>
          </div>

          <div className="description-section">
            <h2 className="description-title">Advanced Image Analysis Platform</h2>
            <p className="description-text">
              Our innovative application leverages advanced AI technology to deliver comprehensive insights into your images. Detect manipulation,
              analyze emotional content, and extract detailed metadata with precision and efficiency.
            </p>

            <div className="features-list">
              <div className="feature-item">
                <span className="feature-icon">🕵</span>
                <span>Forensic Analysis</span>
              </div>
              <div className="feature-item">
                <span className="feature-icon">😊</span>
                <span>Sentiment Detection</span>
              </div>
              <div className="feature-item">
                <span className="feature-icon">📊</span>
                <span>Metadata Extraction</span>
              </div>
              <div className="feature-item">
                <span className="feature-icon">🔒</span>
                <span>Security Verification</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Side - Upload Section */}
        <div className="right-section">
          {/* Floating particles for background animation */}
          <div className="particle"></div>
          <div className="particle"></div>
          <div className="particle"></div>

          <div className="upload-card">
            <div className="upload-header">
              <h3 className="upload-title">Upload Your Image</h3>
              <p className="upload-subtitle">Get instant analysis results</p>
            </div>

            <div className="upload-area">
              <label htmlFor="file-upload" className="custom-file-upload">
                <div className="upload-icon">📤</div>
                <div className="upload-text">
                  <span className="upload-primary">Choose Image File</span>
                  <span className="upload-secondary">or drag and drop here</span>
                </div>
              </label>
              <input
                id="file-upload"
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
              />
            </div>

            {selectedImage && (
              <div className="image-preview">
                <div className="preview-header">
                  <span className="preview-title">Image Preview</span>
                </div>
                <div className="preview-container">
                  <img
                    src={URL.createObjectURL(selectedImage)}
                    alt="Preview"
                    className="preview-image"
                  />
                </div>
                <button className="analyze-button" onClick={handleAnalyzeClick}>
                  <span className="button-icon">🔍</span>
                  <span className="button-text">Analyze Image</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;