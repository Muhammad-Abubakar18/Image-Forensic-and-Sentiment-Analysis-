import React from 'react';
import './MatrixTable.css';
import { FaTimesCircle } from 'react-icons/fa';

const MatrixTable = ({ matrix }) => {
  if (!matrix || !Array.isArray(matrix) || matrix.length === 0) {
    return (
      <div className="no-data-message">
        <FaTimesCircle color="red" style={{ marginRight: '6px' }} />
        <span>Matrix data could not be extracted because the image is not in JPEG format.</span>
      </div>
    );
  }

  return (
    <table className="matrix-table">
      <tbody>
        {matrix.map((row, i) => (
          <tr key={i}>
            {row.map((value, j) => (
              <td key={j}>{value}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default MatrixTable;
