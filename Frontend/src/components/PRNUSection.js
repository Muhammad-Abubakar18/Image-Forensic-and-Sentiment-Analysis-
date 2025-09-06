import { useState } from "react";
import JsonTable from "./JsonTable";
import { mapKeysToLabels } from '../utils/MetadataMapper';

const PRNUSection = ({ prnuData, prnuMaps }) => {
  const [selectedImage, setSelectedImage] = useState(null);

  if (!prnuData && (!prnuMaps || prnuMaps.length === 0)) {
    return (
      <div className="no-data-message">
        <span>No PRNU results available.</span>
      </div>
    );
  }

  return (
    <div className="prnu-section">
      {prnuData && (
        <JsonTable
          title="PRNU Results"
          data={mapKeysToLabels(prnuData)}
          columnHeaders={["Attribute", "Value"]}
        />
      )}

      {prnuMaps && prnuMaps.length > 0 && (
        <div className="recent-uploads-section">
          <h3>PRNU Maps</h3>
          <div className="prnu-grid">
            {prnuMaps.map((mapUrl, index) => (
              <div
                className="prnu-card"
                key={index}
                onClick={() => setSelectedImage(`http://localhost:8000${mapUrl}`)}
              >
                <img
                  src={`http://localhost:8000${mapUrl}`}
                  alt={`PRNU Map ${index + 1}`}
                />
                <p>Map {index + 1}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 🖼 Modal Preview */}
      {selectedImage && (
        <div className="image-modal-overlay" onClick={() => setSelectedImage(null)}>
          <div className="image-modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedImage(null)}>×</button>
            <img src={selectedImage} alt="Full Preview" />
          </div>
        </div>
      )}
    </div>
  );
};

export default PRNUSection;
