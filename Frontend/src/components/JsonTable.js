import React from 'react';
import './JsonTable.css';
import { FaCheckCircle, FaTimesCircle } from 'react-icons/fa';

// Helper function to detect missing/invalid values
const isMissingValue = (val) => {
  if (val === null || val === undefined) return true;

  if (typeof val === 'string') {
    const lower = val.toLowerCase();
    return (
      lower === '' ||
      lower === 'n/a' ||
      lower === 'not available' ||
      lower.includes('no exif metadata found') ||
      lower === 'null'
    );
  }

  if (Array.isArray(val)) return val.length === 0;

  if (typeof val === 'object') return Object.keys(val).length === 0;

  return false;
};

const JsonTable = ({ data, title, showColumnHeaders = true, columnHeaders = ['Attribute', 'Value'] }) => {
  if (!data || Object.keys(data).length === 0) {
    return (
      <div className="no-data-message">
        <FaTimesCircle color="red" style={{ marginRight: '6px' }} />
        <span>No data available.</span>
      </div>
    );
  }

  // Filter out entries with only missing values
  const validEntries = Object.entries(data).filter(([_, val]) => !isMissingValue(val));

  if (validEntries.length === 0) {
    return (
      <div className="no-data-message">
        <FaTimesCircle color="red" style={{ marginRight: '6px' }} />
        <span>No valid metadata found.</span>
      </div>
    );
  }

  return (
    <div style={{ marginBottom: '1rem' }}>
      {title && <h3 style={{ marginBottom: '0.5rem' }}>{title}</h3>}
      <div className="table-container">
        <table className="data-table">
          {showColumnHeaders && (
            <thead>
              <tr>
                <th>{columnHeaders[0]}</th>
                <th>{columnHeaders[1]}</th>
              </tr>
            </thead>
          )}
          <tbody>
            {Object.entries(data).map(([key, value]) => {
              const isMissing = isMissingValue(value);

              // ✅ Boolean handling
              if (typeof value === 'boolean') {
                return (
                  <tr key={key} className="data-row">
                    <td className="key-cell">{key}</td>
                    <td className="value-cell">
                      {value ? (
                        <>
                          <FaCheckCircle color="green" style={{ marginRight: '6px' }} />
                          <span style={{ color: 'green' }}>True</span>
                        </>
                      ) : (
                        <>
                          <FaTimesCircle color="red" style={{ marginRight: '6px' }} />
                          <span style={{ color: 'red' }}>False</span>
                        </>
                      )}
                    </td>
                  </tr>
                );
              }

              return (
                <tr key={key} className="data-row">
                  <td className="key-cell">{key}</td>
                  <td className="value-cell">
                    {isMissing ? (
                      <>
                        <FaTimesCircle color="red" style={{ marginRight: '6px' }} />
                        <span className="missing-value">
                          {typeof value === 'string' && value !== '' ? value : 'No value'}
                        </span>
                      </>
                    ) : (
                    <>
                      <FaCheckCircle color="green" style={{ marginRight: '6px' }} />
                      <span className="value-text">
                        {typeof value === 'object' ? (
                          <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                            {JSON.stringify(value, null, 2)}
                          </pre>
                        ) : (
                          <span className="value-content">{value}</span>
                        )}
                      </span>
                    </>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default JsonTable;
