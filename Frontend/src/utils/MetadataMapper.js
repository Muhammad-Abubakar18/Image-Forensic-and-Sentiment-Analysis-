

// Dictionary for mapping technical keys to user-friendly labels
export const keyMapping = {
  // Lighting inconsistencies
  mean_local_variance: 'Mean Local Variance',
  std_local_variance: 'Standard Local Variance',

  // Regional Noise Variations
  region_0_0: 'Top-Left Region',
  region_0_1: 'Top-Right Region',
  region_1_0: 'Bottom-Left Region',
  region_1_1: 'Bottom-Right Region',

  // EXIF Metadata
  ResolutionUnit: 'Resolution Unit',
  YResolution: 'Vertical Resolution (Y-axis)',
  XResolution: 'Horizontal Resolution (X-axis)',
  YCbCrPositioning: 'Chrominance Subsampling Positioning',
  ImageDescription: 'Image Description',
  'Image XResolution': 'Image Horizontal Resolution (X-axis)',
  'Image YResolution': 'Image Vertical Resolution (Y-axis)',
  'Image ResolutionUnit': 'Image Resolution Unit',
  'Image YCbCrPositioning': 'Image Chrominance Subsampling Positioning',
  'Image ImageDescription': 'Image Description',

  // CFA Analysis
  mean_frequency_magnitude: 'Mean Frequency Magnitude',
  std_frequency_magnitude: 'Standard Frequency Magnitude',
  possible_cfa_pattern: 'Possible CFA Pattern',

  // 🆕 Morphing Detection
  face_detected: 'Face Detected',
  morphing_suspected: 'Morphing Suspected',
  asymmetry_score: 'Asymmetry Score',
  blend_artifact_score: 'Blend Artifact Score',
  error: 'Error',

   // Emotion labels
  angry: 'Angry',
  disgust: 'Disgust',
  fear: 'Fear',
  happy: 'Happy',
  sad: 'Sad',
  surprise: 'Surprise',
  neutral: 'Neutral',

    // Face region attributes
  x: 'X Coordinate',
  y: 'Y Coordinate',
  w: 'Width',
  h: 'Height',
  left_eye: 'Left Eye Position',
  right_eye: 'Right Eye Position',

};

// Utility function to apply mapping to an object or array of objects
export const mapKeysToLabels = (data) => {
  if (Array.isArray(data)) {
    return data.map(item => {
      const key = Object.keys(item)[0];
      const value = item[key];
      const newKey = keyMapping[key] || key;
      return { [newKey]: value };
    });
  } else if (typeof data === 'object' && data !== null) {
    return Object.fromEntries(
      Object.entries(data).map(([key, value]) => [
        keyMapping[key] || key,
        value
      ])
    );
  }
  return data;
};

