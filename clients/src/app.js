import React, { useEffect, useState } from 'react';
import axios from 'axios';

function App() {
  const [keystrokes, setKeystrokes] = useState([]);
  const [mouseMoves, setMouseMoves] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [userId, setUserId] = useState('123');

  useEffect(() => {
    const handleKeyDown = (e) => {
      const time = Date.now();
      setKeystrokes(prev => [...prev, { key: e.key, downTime: time }]);
    };

    const handleMouseMove = (e) => {
      setMouseMoves(prev => [...prev, { x: e.clientX, y: e.clientY, time: Date.now() }]);
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('mousemove', handleMouseMove);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  const submitData = async () => {
    setLoading(true);
    setError(null);

    const payload = { userId, keystrokeData: keystrokes, mouseData: mouseMoves };

    try {
      const res = await axios.post('http://localhost:4000/api/collect-data', payload);
      alert(`Fraud Prediction: ${res.data.prediction.fraud}`);
    } catch (err) {
      console.error('Error submitting data:', err);
      setError('Failed to submit data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Online Banking Login</h1>
      <input
        type="text"
        placeholder="Type here..."
        onChange={(e) => setUserId(e.target.value)}
        style={{ marginBottom: '10px', padding: '8px', width: '100%' }}
      />
      <button onClick={submitData} disabled={loading} style={{ padding: '10px 15px' }}>
        {loading ? 'Submitting...' : 'Submit'}
      </button>
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
}

export default App;
