import React, { useState } from 'react';
import './HashTable.css';
import { FaCopy } from 'react-icons/fa';

const HashTable = ({ data = {} }) => {
  const [comparisonHash, setComparisonHash] = useState('');
  const [copiedKey, setCopiedKey] = useState('');

  const copyToClipboard = (key, value) => {
    navigator.clipboard.writeText(value);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(''), 1500);
  };

  const isHashMatch = (hash) =>
    comparisonHash && hash?.toLowerCase() === comparisonHash.toLowerCase();

  return (
    <div className="hash-table">
      <div className="comparison-input">
        <label>Compare with your hash:</label>
        <input
          type="text"
          value={comparisonHash}
          onChange={(e) => setComparisonHash(e.target.value)}
          placeholder="Enter hash to compare"
        />
      </div>

      <table className="data-table">
        <thead>
          <tr>
            <th>Hash Type</th>
            <th>Value</th>
            <th>Copy</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(data).map(([key, value]) => {
            const match = isHashMatch(value);
            return (
              <tr key={key} className={match ? 'match' : ''}>
                <td>{key}</td>
                <td style={{ color: match ? 'green' : 'black' }}>
                  {value}
                </td>
                <td>
                  <button onClick={() => copyToClipboard(key, value)}>
                    <FaCopy /> {copiedKey === key ? 'Copied!' : 'Copy'}
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default HashTable;
