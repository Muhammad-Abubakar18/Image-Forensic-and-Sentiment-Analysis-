import React from 'react';

const ResultSection = ({ title, children }) => (
  <div className="result-section">
    <h2>{title}</h2>
    <div className="section-content">{children}</div>
  </div>
);

export default ResultSection;
