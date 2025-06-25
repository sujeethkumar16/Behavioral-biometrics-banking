import React, { useEffect, useState } from 'react';
import axios from 'axios';

function Dashboard() {
  const [attempts, setAttempts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const itemsPerPage = 10; // Number of items to display per page

  useEffect(() => {
    const fetchAttempts = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await axios.get(`http://localhost:4000/api/login-attempts?page=${page}&limit=${itemsPerPage}`);
        setAttempts(res.data.attempts);
        setTotalPages(res.data.totalPages);
      } catch (err) {
        console.error('Error fetching login attempts:', err);
        setError('Failed to fetch login attempts. Please try again later.');
      } finally {
        setLoading(false);
      }
    };
    fetchAttempts();
  }, [page]);

  const handleNextPage = () => {
    if (page < totalPages) {
      setPage(prevPage => prevPage + 1);
    }
  };

  const handlePreviousPage = () => {
    if (page > 1) {
      setPage(prevPage => prevPage - 1);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Login Attempts</h1>
      {loading && <p>Loading...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {!loading && !error && (
        <>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ border: '1px solid #ddd', padding: '8px' }}>User </th>
                <th style={{ border: '1px solid #ddd', padding: '8px' }}>Time</th>
                <th style={{ border: '1px solid #ddd', padding: '8px' }}>Fraud</th>
              </tr>
            </thead>
            <tbody>
              {attempts.map((a, i) => (
                <tr key={i}>
                  <td style={{ border: '1px solid #ddd', padding: '8px' }}>{a.userId}</td>
                  <td style={{ border: '1px solid #ddd', padding: '8px' }}>{new Date(a.timestamp).toLocaleString()}</td>
                  <td style={{ border: '1px solid #ddd', padding: '8px' }}>{a.prediction.fraud ? 'Yes' : 'No'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: '20px' }}>
            <button onClick={handlePreviousPage} disabled={page === 1} style={{ marginRight: '10px' }}>
              Previous
            </button>
            <button onClick={handleNextPage} disabled={page === totalPages}>
              Next
            </button>
          </div>
          <p>Page {page} of {totalPages}</p>
        </>
      )}
    </div>
  );
}

export default Dashboard;
