import React, { useState, useEffect } from 'react';
import ResultSection from './ResultSection';
import './MetadataPage.css';
import JsonTable from './JsonTable';
import MatrixTable from './MatrixTable';
import { FaCheckCircle, FaFolderOpen, FaTimesCircle } from 'react-icons/fa';
import { FaImage, FaSearch, FaSun, FaWaveSquare, FaCopy, FaCut, FaLayerGroup, FaFingerprint, FaCamera, FaUserCheck } from 'react-icons/fa';
import { mapKeysToLabels } from '../utils/MetadataMapper';
import HashTable from './HashTable';
import findBaseFolder from '../utils/FindBaseFolders';
import ImagePreviewModal from './ImagePreviewModal';
import PRNUSection from './PRNUSection';
import ThumbnailComparison from './Thumbnail';

const MetadataPage = ({ imageData }) => {
  const sections = [
    { name: 'Dashboard', icon: <FaLayerGroup /> },
    { name: 'Metadata', icon: <FaSearch /> },
    { name: 'ELA', icon: <FaLayerGroup /> },
    { name: 'Lighting Heatmap', icon: <FaSun /> },
    { name: 'Noise Map', icon: <FaWaveSquare /> },
    { name: 'Copy-Move', icon: <FaCopy /> },
    { name: 'Splicing', icon: <FaCut /> },
    { name: 'JPEG Structure', icon: <FaImage /> },
    { name: 'Digest Info', icon: <FaFingerprint /> },
    { name: 'JPEG Quality', icon: <FaCamera /> },
    { name: 'Morphing', icon: <FaUserCheck /> },
    { name: 'CFA', icon: <FaWaveSquare /> },
    { name: 'Thumbnail', icon: <FaImage /> },
    { name: 'Camera ID', icon: <FaCamera /> },
    { name: 'Emotion Analysis', icon: <FaUserCheck /> },
    { name: 'Object Detection', icon: <FaSearch /> },
    { name: 'PRNU', icon: <FaFingerprint /> },
  ];

  const [active, setActive] = useState('Metadata');
  const [previewSrc, setPreviewSrc] = useState(null);
  const [selectedFolder, setSelectedFolder] = useState(null);
  const [dashboardStats, setDashboardStats] = useState({});
  const [dashboardFiles, setDashboardFiles] = useState({});
  const [filteredImages, setFilteredImages] = useState([]);
  const [recentUploads, setRecentUploads] = useState([]);

  const folderData = [
    { key: "ela_images", label: "ELA" },
    { key: "forgery_techniques", label: "Forgery Techniques" },
    { key: "noise_map_images", label: "Noise Maps" },
    { key: "lighting_heatmaps", label: "Lighting Heatmaps" },
  ];

  const elaBaseUrl = 'http://127.0.0.1:8000/';

  useEffect(() => {
    fetch('http://localhost:8000/analyzed-results')
      .then(res => res.json())
      .then(data => {
        setDashboardStats(data.stats);
        setDashboardFiles(data.files);
      });
  }, []);

  useEffect(() => {
    fetch('http://localhost:8000/recent-uploads')
      .then(res => res.json())
      .then(data => {
        console.log('Recent uploads:', data.recent_uploads);
        setRecentUploads(data.recent_uploads.slice(0, 5));
      })
      .catch(err => console.error('Error fetching recent uploads:', err));
  }, []);

  const formatLabel = (label) => {
    if (!label || typeof label !== "string") return label;
    return label.charAt(0).toUpperCase() + label.slice(1);
  };

  const rawPrnuData = imageData?.prnu?.fingerprint || {};
  const formattedPrnuData = Object.keys(rawPrnuData)
    .filter((key) => key !== "saved_as")
    .reduce((obj, key) => {
      obj[formatLabel(key)] = rawPrnuData[key];
      return obj;
    }, {});

  const handleImageClick = (src) => {
    setPreviewSrc(src);
  };

  return (
    <>
      {/*<header className="title-bar">
      <img src="/camera.png" alt="Logo" className="logo" />
      <h1>Image Forensics Tool</h1>
    </header>*/}
      <div className="metadata-page">
        <aside className="sidebar">
          <h2>Results</h2>
          <ul>
            {sections.map(({ name, icon }) => (
              <li
                key={name}
                onClick={() => setActive(name)}
                className={active === name ? 'active' : ''}
              >
                <span className="icon">{icon}</span>
                {name}
              </li>
            ))}
          </ul>
        </aside>

        <main className="results-content">
          {active === "Dashboard" && (
            <>
              {!selectedFolder && (
                <h2 className="dashboard-title">Dashboard</h2>
              )}

              {/* Folder View */}
              {!selectedFolder ? (
                <>
                  <div className="dashboard-grid">
                    {folderData.map(({ key, label }) => (
                      <div
                        key={key}
                        className="dashboard-card"
                        onClick={() => {
                          setSelectedFolder(label);
                          if (key === "forgery_techniques") {
                            setFilteredImages({
                              splicing: dashboardFiles["splicing_images"] || [],
                              copyMove: dashboardFiles["copy_move_images"] || []
                            });
                          } else {
                            setFilteredImages(dashboardFiles[key] || []);
                          }
                        }}
                      >
                        <FaFolderOpen size={50} color="#6D83F2" />
                        <h3>{label}</h3>
                        <p>
                          {key === 'forgery_techniques'
                            ? (dashboardStats["splicing_images"] || 0) + (dashboardStats["copy_move_images"] || 0)
                            : dashboardStats[key] || 0} files
                        </p>
                      </div>
                    ))}
                  </div>

                  {/* Recent Uploads Section */}
                  {recentUploads.length > 0 && (
                    <div className="recent-uploads-section">
                      <h3>Recent Uploads</h3>
                      <div className="recent-grid">
                        {recentUploads.map((item, idx) => (
                          <div
                            key={idx}
                            className="recent-card"
                            onClick={() => setPreviewSrc(`http://localhost:8000${item.url}`)}
                          >
                            <img
                              src={`http://localhost:8000${item.url}`}
                              alt={`Upload ${idx + 1}`}
                            />
                            <div className="recent-info">
                              <p><strong>Name:</strong> {item.filename}</p>
                              <p><strong>Uploaded:</strong> {item.timestamp}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <>
                  {/* Back Button */}
                  <button
                    onClick={() => setSelectedFolder(null)}
                    className="dashboard-back-btn"
                  >
                    Back
                  </button>

                  {/* Folder Heading */}
                  <h2 className="dashboard-title">{selectedFolder}</h2>

                  {/* Image Grid */}
                  {selectedFolder === "Forgery Techniques" ? (
                    <>
                      <h3>Splicing</h3>
                      <div className="image-grid">
                        {filteredImages.splicing?.map((file, idx) => (
                          <div
                            key={`splicing-${idx}`}
                            className="image-card"
                            onClick={() => setPreviewSrc(`http://localhost:8000/${findBaseFolder(file)}/${file}`)}
                          >
                            <img
                              src={`http://localhost:8000/${findBaseFolder(file)}/${file}`}
                              alt={file}
                            />
                            <p>{file}</p>
                          </div>
                        ))}
                      </div>

                      <h3>Copy-Move</h3>
                      <div className="image-grid">
                        {filteredImages.copyMove?.map((file, idx) => (
                          <div
                            key={`copymove-${idx}`}
                            className="image-card"
                            onClick={() => setPreviewSrc(`http://localhost:8000/${findBaseFolder(file)}/${file}`)}
                          >
                            <img
                              src={`http://localhost:8000/${findBaseFolder(file)}/${file}`}
                              alt={file}
                            />
                            <p>{file}</p>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="image-grid">
                      {filteredImages.map((file, idx) => (
                        <div
                          key={idx}
                          className="image-card"
                          onClick={() => setPreviewSrc(`http://localhost:8000/${findBaseFolder(file)}/${file}`)}
                        >
                          <img
                            src={`http://localhost:8000/${findBaseFolder(file)}/${file}`}
                            alt={file}
                          />
                          <p>{file}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </>
          )}
          {active === 'Metadata' && (
            <ResultSection title="Metadata" >
              {console.log("Source Image Path:", imageData.source_image_path)}
              {imageData.source_image_path && (
                <div className="centered-image">
                  <img src={`http://localhost:8000${imageData.source_image_path}`} alt="Uploaded" className="source-image" onClick={() =>
                    handleImageClick(`http://localhost:8000${imageData.source_image_path}`)}
                    style={{ cursor: "pointer" }} />
                </div>
              )}
              {imageData.image && (
                <div style={{ marginBottom: '1rem' }}>
                  <h3>Image Name: <span style={{ fontWeight: 'normal' }}>{imageData.image}</span></h3>
                </div>
              )}

              <h3>EXIF Metadata - PIL</h3>
              {imageData.metadata_pil &&
                Object.keys(imageData.metadata_pil).filter((key) => key.toLowerCase() !== 'info').length > 0 ? (
                <JsonTable
                  data={mapKeysToLabels(
                    Object.fromEntries(
                      Object.entries(imageData.metadata_pil).filter(([key]) => key.toLowerCase() !== 'info')
                    )
                  )}
                  columnHeaders={['Attribute', 'Value']}
                />
              ) : (
                <p style={{ display: 'flex', alignItems: 'center' }}>
                  <FaTimesCircle color="red" style={{ marginRight: '6px' }} />
                  No image metadata found.
                </p>
              )}

              <h3>EXIF Metadata - ExifRead</h3>
              {imageData.metadata_exifread &&
                Object.keys(imageData.metadata_exifread).filter((key) => key.toLowerCase() !== 'info').length > 0 ? (
                <JsonTable
                  data={mapKeysToLabels(
                    Object.fromEntries(
                      Object.entries(imageData.metadata_exifread).filter(([key]) => key.toLowerCase() !== 'info')
                    )
                  )}
                  columnHeaders={['Attribute', 'Value']}
                />
              ) : (
                <p style={{ display: 'flex', alignItems: 'center' }}>
                  <FaTimesCircle color="red" style={{ marginRight: '6px' }} />
                  No image metadata found.
                </p>
              )}

              <h3>Brightness Histogram (first 10 values)</h3>
              {imageData.lighting_inconsistencies?.brightness_histogram ? (
                <JsonTable data={Object.fromEntries(
                  imageData.lighting_inconsistencies.brightness_histogram
                    .slice(0, 10)
                    .map((val, idx) => [idx, val])
                )} columnHeaders={['Index', 'Value']} />
              ) : (
                <p>No brightness histogram available.</p>
              )}

              <h3>Hashes</h3>
              <HashTable data={imageData.hashes} />

              <h3>Lighting Inconsistencies</h3>
              <JsonTable
                data={mapKeysToLabels({
                  mean_local_variance: imageData.lighting_inconsistencies.mean_local_variance.toFixed(2),
                  std_local_variance: imageData.lighting_inconsistencies.std_local_variance.toFixed(2)
                })}
                showColumnHeaders={false}
              />

              <h3>Regional Noise Variation</h3>
              {Array.isArray(imageData.noise_analysis?.regional_variation) &&
                imageData.noise_analysis.regional_variation.length > 0 ? (
                <JsonTable
                  data={Object.fromEntries(
                    mapKeysToLabels(imageData.noise_analysis.regional_variation).map(item => Object.entries(item)[0])
                  )}
                  columnHeaders={['Region', 'Value']}
                />

              ) : (
                <p>No regional noise variation data available.</p>
              )}

              <h3>Lighting Histogram (first 10 values)</h3>
              {Array.isArray(imageData.lighting_histogram) ? (
                <JsonTable data={Object.fromEntries(
                  imageData.lighting_histogram.slice(0, 10).map((val, idx) => [idx, val])
                )} columnHeaders={['Index', 'Value']} />
              ) : (
                <p>No lighting histogram available.</p>
              )}
            </ResultSection>
          )}

          {active === 'ELA' && (
            <ResultSection title="Error Level Analysis (ELA)">
              {imageData.source_image_path && (
                <div className="centered-image">
                  <img src={`http://localhost:8000${imageData.source_image_path}`} alt="Uploaded" className="source-image" onClick={() =>
                    handleImageClick(`http://localhost:8000${imageData.source_image_path}`)}
                    style={{ cursor: "pointer" }} />
                </div>
              )}
              <h3>ELA Image</h3>
              <div className="centered-image">
                <img src={elaBaseUrl + imageData.ela_image_path} alt="ELA"
                  onClick={() => handleImageClick(elaBaseUrl + imageData.ela_image_path)}
                  style={{ cursor: "pointer" }} />
              </div>
            </ResultSection>
          )}

          {active === 'Lighting Heatmap' && imageData.lighting_inconsistencies.heatmap_path && (
            <ResultSection title="Lighting Heatmap">
              {imageData.source_image_path && (
                <div className="centered-image">
                  <img src={`http://localhost:8000${imageData.source_image_path}`} alt="Uploaded" className="source-image" onClick={() =>
                    handleImageClick(`http://localhost:8000${imageData.source_image_path}`)}
                    style={{ cursor: "pointer" }} />
                </div>
              )}
              <h3>Lighting Heatmap Image</h3>
              <div className="centered-image">
                <img src={elaBaseUrl + imageData.lighting_inconsistencies.heatmap_path} alt="Heatmap" onClick={() => handleImageClick(elaBaseUrl + imageData.lighting_inconsistencies.heatmap_path)}
                  style={{ cursor: "pointer" }} />
              </div>
            </ResultSection>
          )}

          {active === 'Noise Map' && imageData.noise_analysis.noise_map_path && (
            <ResultSection title="Noise Map">
              {imageData.source_image_path && (
                <div className="centered-image">
                  <img src={`http://localhost:8000${imageData.source_image_path}`} alt="Uploaded" className="source-image" onClick={() =>
                    handleImageClick(`http://localhost:8000${imageData.source_image_path}`)}
                    style={{ cursor: "pointer" }} />
                </div>
              )}
              <h3>Noise Map Image</h3>
              <div className="centered-image">
                <img src={elaBaseUrl + imageData.noise_analysis.noise_map_path} alt="Noise Map" onClick={() => handleImageClick(elaBaseUrl + imageData.noise_analysis.noise_map_path)}
                  style={{ cursor: "pointer" }}/>
              </div>
            </ResultSection>
          )}

          {active === 'Copy-Move' && imageData.copy_move_forgery.map_path && (
            <ResultSection title="Copy-Move Forgery">
              {imageData.source_image_path && (
                <div className="centered-image">
                  <img src={`http://localhost:8000${imageData.source_image_path}`} alt="Uploaded" className="source-image" onClick={() =>
                    handleImageClick(`http://localhost:8000${imageData.source_image_path}`)}
                    style={{ cursor: "pointer" }} />
                </div>
              )}
              <h3>Copy-Move Image</h3>
              <div className="centered-image">
                <img src={elaBaseUrl + imageData.copy_move_forgery.map_path} alt="Copy Move Detection" onClick={() => handleImageClick(elaBaseUrl + imageData.copy_move_forgery.map_path)}
                  style={{ cursor: "pointer" }} />
              </div>
            </ResultSection>
          )}

          {active === 'Splicing' && imageData.splicing_analysis.ela_splicing_image && (
            <ResultSection title="Splicing Detection">
              {imageData.source_image_path && (
                <div className="centered-image">
                  <img src={`http://localhost:8000${imageData.source_image_path}`} alt="Uploaded" className="source-image" onClick={() =>
                    handleImageClick(`http://localhost:8000${imageData.source_image_path}`)}
                    style={{ cursor: "pointer" }}/>
                </div>
              )}
              <h3>Splicing Image</h3>
              <div className="centered-image">
                <img src={elaBaseUrl + imageData.splicing_analysis.ela_splicing_image} alt="Splicing" onClick={() => handleImageClick(elaBaseUrl + imageData.splicing_analysis.ela_splicing_image)}
                  style={{ cursor: "pointer" }} />
              </div>
            </ResultSection>
          )}

          {active === 'JPEG Structure' && (
            <ResultSection title="JPEG Structure Metadata">
              {imageData.source_image_path && (
                <div className="centered-image">
                  <img src={`http://localhost:8000${imageData.source_image_path}`} alt="Uploaded" className="source-image" onClick={() =>
                    handleImageClick(`http://localhost:8000${imageData.source_image_path}`)}
                    style={{ cursor: "pointer" }} />
                </div>
              )}
              <h3>JPEG Metadata</h3>
              <JsonTable data={imageData.jpeg_structure_metadata} />
            </ResultSection>
          )}


          {active === 'Digest Info' && (
            <ResultSection title="Digest Information">
              {imageData.source_image_path && (
                <div className="centered-image">
                  <img src={`http://localhost:8000${imageData.source_image_path}`} alt="Uploaded" className="source-image" onClick={() =>
                    handleImageClick(`http://localhost:8000${imageData.source_image_path}`)}
                    style={{ cursor: "pointer" }} />
                </div>
              )}
              <h3>Digest Information</h3>
              <JsonTable
                data={imageData.digest_info}
                columnHeaders={['Attribute', 'Value']}
              />
            </ResultSection>
          )}


          {active === 'JPEG Quality' && (
            <ResultSection title="JPEG Quality Details">
              {imageData.source_image_path && (
                <div className="centered-image">
                  <img src={`http://localhost:8000${imageData.source_image_path}`} alt="Uploaded" className="source-image" onClick={() =>
                    handleImageClick(`http://localhost:8000${imageData.source_image_path}`)}
                    style={{ cursor: "pointer" }} />
                </div>
              )}

              <div className="jpeg-quality">
                <h3>JPEG Quality Estimate</h3>
                <p style={{ display: 'flex', alignItems: 'center' }}>
                  {imageData?.jpeg_quality_details?.quality_estimate !== undefined &&
                    imageData?.jpeg_quality_details?.quality_estimate !== null ? (
                    <>
                      <FaCheckCircle color="green" style={{ marginRight: '6px' }} />
                      {imageData.jpeg_quality_details.quality_estimate}
                    </>
                  ) : (
                    <>
                      <FaTimesCircle color="red" style={{ marginRight: '6px' }} />
                      <span>This image is not in JPEG format.</span>
                    </>
                  )}
                </p>
              </div>

              <div className="matrix-container">
                <div>
                  <h3>Luminance Quantization Table</h3>
                  <MatrixTable matrix={imageData.jpeg_quality_details?.quantization_tables?.Luminance} />
                </div>

                <div>
                  <h3>Chrominance Quantization Table</h3>
                  <MatrixTable matrix={imageData.jpeg_quality_details?.quantization_tables?.["Chrominance 1"]} />
                </div>
              </div>
            </ResultSection>
          )}

          {active === 'Morphing' && (
            <ResultSection title="Morphing Detection">
              {imageData.source_image_path && (
                <div className="centered-image">
                  <img
                    src={`http://localhost:8000${imageData.source_image_path}`}
                    alt="Uploaded"
                    className="source-image"
                    onClick={() =>
                    handleImageClick(`http://localhost:8000${imageData.source_image_path}`)}
                    style={{ cursor: "pointer" }}
                  />
                </div>
              )}
              <h3>Morphing Results</h3>
              {imageData.face_region_analysis ? (
                <>
                  <JsonTable
                    data={mapKeysToLabels(
                      Object.fromEntries(
                        Object.entries(imageData.face_region_analysis).filter(
                          ([key]) => key !== 'error'
                        )
                      )
                    )}
                    showColumnHeaders={true}
                    columnHeaders={['Attribute', 'Value']}
                  />

                  {/* ✅ Show error message separately if present */}
                  {imageData.face_region_analysis.error && (
                    <div className="error-message">
                      <FaTimesCircle color="red" style={{ marginRight: '6px' }} />
                      <span>{imageData.face_region_analysis.error}</span>
                    </div>
                  )}
                </>
              ) : (
                <p>No morphing data available.</p>
              )}
            </ResultSection>
          )}

          {active === 'CFA' && (
            <ResultSection title="CFA (Color Filter Array) Analysis">
              {imageData.source_image_path && (
                <div className="centered-image">
                  <img src={`http://localhost:8000${imageData.source_image_path}`} alt="Uploaded" className="source-image" onClick={() =>
                    handleImageClick(`http://localhost:8000${imageData.source_image_path}`)}
                    style={{ cursor: "pointer" }}/>
                </div>
              )}
              <h3>CFA Analysis Results</h3>
              {imageData.cfa_analysis && !imageData.cfa_analysis.error ? (
                <JsonTable
                  data={mapKeysToLabels(imageData.cfa_analysis)}
                  title={null}
                />
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'red' }}>
                  <FaTimesCircle />
                  <span>{imageData.cfa_analysis?.error || 'No CFA analysis available.'}</span>
                </div>
              )}
            </ResultSection>
          )}

          {active === 'Thumbnail' && (
            <ResultSection title="Thumbnail Comparison">
              {imageData.source_image_path && (
                <div className="centered-image">
                  <img
                    src={`http://localhost:8000${imageData.source_image_path}`}
                    alt="Uploaded"
                    className="source-image"
                    onClick={() =>
                    handleImageClick(`http://localhost:8000${imageData.source_image_path}`)}
                    style={{ cursor: "pointer" }}
                  />
                </div>
              )}

              <ThumbnailComparison imageData={imageData} />
            </ResultSection>
          )}

          {active === 'Camera ID' && (
            <ResultSection title="Camera Model Identification">
              {imageData.source_image_path && (
                <div className="centered-image">
                  <img src={`http://localhost:8000${imageData.source_image_path}`} alt="Uploaded" className="source-image" onClick={() =>
                    handleImageClick(`http://localhost:8000${imageData.source_image_path}`)}
                    style={{ cursor: "pointer" }} />
                </div>
              )}
              {imageData.camera_model && !imageData.camera_model.error ? (
                <JsonTable
                  data={{
                    Make: imageData.camera_model.Make,
                    Model: imageData.camera_model.Model
                  }}
                  title=""
                  showColumnHeaders={true}
                  columnHeaders={['Attribute', 'Value']}
                />
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <FaTimesCircle color="red" />
                  <span>
                    {imageData.camera_model?.error || 'No camera identification available.'}
                  </span>
                </div>
              )}
            </ResultSection>
          )}

          {active === 'Emotion Analysis' && (
            <ResultSection title="Emotion Analysis">
              {imageData.source_image_path && (
                <div className="centered-image">
                  <img
                    src={`http://localhost:8000${imageData.source_image_path}`}
                    alt="Uploaded"
                    className="source-image"
                    onClick={() =>
                    handleImageClick(`http://localhost:8000${imageData.source_image_path}`)}
                    style={{ cursor: "pointer" }}
                  />
                </div>
              )}
              {/* ✅ If emotion data exists */}
              {imageData.emotion_analysis && !imageData.emotion_analysis.error ? (
                <>
                  {/* Show dominant emotion */}
                  <h3>Dominant Emotion</h3>
                  <p style={{ display: 'flex', alignItems: 'center' }}>
                    <FaCheckCircle color="green" style={{ marginRight: '6px' }} />
                    {imageData.emotion_analysis.dominant_emotion}
                  </p>

                  {/* Show emotion scores */}
                  <h3>Emotion Scores</h3>
                  <JsonTable
                    data={mapKeysToLabels(imageData.emotion_analysis.emotion_scores)}
                    columnHeaders={['Emotion', 'Score']}
                  />

                  {/* Show face confidence */}
                  <h3>Face Confidence</h3>
                  <p>
                    {imageData.emotion_analysis.face_confidence
                      ? `${imageData.emotion_analysis.face_confidence} confidence`
                      : "Not available"}
                  </p>

                  {/* Show face region */}
                  <h3>Face Region</h3>
                  <JsonTable
                    data={mapKeysToLabels(imageData.emotion_analysis.face_region)}
                    columnHeaders={['Attribute', 'Value']}
                  />
                </>
              ) : (
                // ✅ Error message case
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'red' }}>
                  <FaTimesCircle />
                  <span>{imageData.emotion_analysis?.error || 'No emotion analysis available.'}</span>
                </div>
              )}
            </ResultSection>
          )}

          {active === 'Object Detection' && (
            <ResultSection title="Object Detection">
              {imageData.source_image_path && (
                <div className="centered-image">
                  <img
                    src={`http://localhost:8000${imageData.source_image_path}`}
                    alt="Uploaded"
                    className="source-image"
                    onClick={() =>
                    handleImageClick(`http://localhost:8000${imageData.source_image_path}`)}
                    style={{ cursor: "pointer" }}
                  />
                </div>
              )}

              {/* ✅ If YOLO results exist */}
              {imageData.object_detection && imageData.object_detection.length > 0 ? (
                <div>
                  <h3>Detected Objects</h3>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Object</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {imageData.object_detection.map((obj, idx) => (
                        <tr key={idx}>
                          <td>{formatLabel(obj.label)}</td>
                          <td style={{ color: obj.confidence > 0.5 ? 'green' : 'red' }}>
                            {obj.confidence.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'red' }}>
                  <FaTimesCircle />
                  <span>No objects detected in this image.</span>
                </div>
              )}
            </ResultSection>
          )}

          {active === 'PRNU' && (
            <ResultSection title="Photo Response Non-Uniformity (PRNU)">
              {imageData.source_image_path && (
                <div className="centered-image">
                  <img
                    src={`http://localhost:8000${imageData.source_image_path}`}
                    alt="Uploaded"
                    className="source-image"
                    onClick={() =>
                    handleImageClick(`http://localhost:8000${imageData.source_image_path}`)}
                    style={{ cursor: "pointer" }}
                  />
                </div>
              )}
              <PRNUSection
                prnuData={formattedPrnuData}
                prnuMaps={[
                  imageData?.prnu?.localization?.heatmap_path,
                  imageData?.prnu?.localization?.mask_path,
                  imageData?.prnu?.localization?.overlay_path,
                ].filter(Boolean)}
              />
            </ResultSection>
          )}

          <ImagePreviewModal
            src={previewSrc}
            onClose={() => setPreviewSrc(null)}
          />
        </main>
      </div>
    </>
  );
};

export default MetadataPage;
