import React from 'react';
import { FaCheckCircle, FaTimesCircle } from 'react-icons/fa';
import './MetadataPage.css';
import handleImageClick from './MetadataPage'

const ThumbnailComparison = ({ imageData }) => {
    const thumbnailData = imageData?.thumbnail_analysis;

    const renderCell = (label, value) => {
        const isMissing = !value || value === '' || value === null;
        const isScore = label === 'Similarity Score';

        return (
            <tr>
                <td>{label}</td>
                <td style={{ color: isMissing ? 'red' : 'green', display: 'flex', alignItems: 'center' }}>
                    {isMissing ? (
                        <>
                            <FaTimesCircle color="red" style={{ marginRight: '6px' }} />
                            <span>No data</span>
                        </>
                    ) : (
                        <>
                            <FaCheckCircle color="green" style={{ marginRight: '6px' }} />
                            <span>{isScore ? Number(value).toFixed(4) : value}</span>
                        </>
                    )}
                </td>
            </tr>
        );
    };

    if (!thumbnailData || thumbnailData.error) {
        return (
            <div className="no-data-message">
                <FaTimesCircle color="red" style={{ marginRight: '6px' }} />
                <span>{thumbnailData?.error || 'No thumbnail data available.'}</span>
            </div>
        );
    }

    return (
        <div className="thumbnail-results">
            {/* Thumbnail Heading + Image Centered */}
            <div className="thumbnail-wrapper">
                <h3 className="thumbnail-heading">Thumbnail</h3>
                <img
                    src={`http://localhost:8000${thumbnailData.thumbnail_path}`}
                    alt="Extracted Thumbnail"
                    className="thumbnail-image"
                    onClick={() => handleImageClick(thumbnailData.thumbnail_path)}
                  style={{ cursor: "pointer" }}
                />
            </div>

            {/* Data Table */}
            <table className="data-table">
                <thead>
                    <tr>
                        <th>Attribute</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {renderCell('Similarity Score', thumbnailData.similarity_score)}
                    {renderCell('Main Image Hash', thumbnailData.main_hash)}
                    {renderCell('Thumbnail Hash', thumbnailData.thumbnail_hash)}
                </tbody>
            </table>
        </div>
    );
};

export default ThumbnailComparison;
