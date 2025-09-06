import React, { useEffect, useState } from 'react';
import './Loader.css';

const messages = [
  "Extracting metadata...",
  "Detecting manipulations...",
  "Analyzing sentiment...",
  "Finalizing report..."
];

const Loader = () => {
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % messages.length);
    }, 2500); // cycle every 2.5s
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="loader-container">
      <div className="loader-orbit">
        <div className="orbit-ring"></div>
        <div className="orbit-dot"></div>
      </div>

      <p className="loader-text">
        Analyzing image<span className="dots"></span>
      </p>
      <p className="loader-subtext">{messages[messageIndex]}</p>
    </div>
  );
};

export default Loader;
