import React from 'react';
import './ImagePreviewModal.css';
import { FaTimes } from 'react-icons/fa';

const ImagePreviewModal = ({ src, onClose }) => {
  if (!src) return null;

  return (
    <div className="image-preview-overlay" onClick={onClose}>
      <div className="image-preview-content" onClick={(e) => e.stopPropagation()}>
        <button className="close-btn" onClick={onClose}>
          <FaTimes />
        </button>
        <img src={src} alt="Preview" className="preview-image" />
      </div>
    </div>
  );
};

export default ImagePreviewModal;
