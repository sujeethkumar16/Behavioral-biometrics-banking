const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const axios = require('axios');
const { body, validationResult } = require('express-validator');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');

const app = express();
app.use(cors());
app.use(bodyParser.json());
app.use(morgan('combined'));

const limiter = rateLimit({
  windowMs: 1 * 60 * 1000,
  max: 100,
  message: 'Too many requests, please try again later.'
});
app.use('/api/collect-data', limiter);

const validateBehaviorData = [
  body('sessionId').isString().withMessage('Session ID must be a string'),
  body('deviceInfo').isObject().withMessage('Device info must be an object'),
  body('behaviorData').isArray().withMessage('Behavior data must be an array'),
  body('behaviorData.*.event').isString().withMessage('Event type must be a string'),
  body('behaviorData.*.timestamp').isNumeric().withMessage('Timestamp must be a number'),
];

app.post('/api/collect-data', validateBehaviorData, async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }

  const behaviorData = req.body;

  try {
    const response = await axios.post('http://localhost:5000/predict', behaviorData);
    res.json({ prediction: response.data });
  } catch (err) {
    console.error('Error communicating with prediction service:', err.message);
    res.status(500).json({ error: 'Prediction service error', details: err.message });
  }
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date() });
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`Backend running on port ${PORT}`));
