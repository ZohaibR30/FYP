import React from 'react';
import axios from 'axios';
import './WelcomePage.css';

function WelcomePage() {
  const handleButtonClick = (apiEndpoint, modelName) => {
    const newTab = window.open('', '_blank');
    newTab.document.write(`
      <html>
        <head>
          <title>${modelName} Video Stream</title>
          <style>
            body {
              background-color: black;
              margin: 0;
              overflow: hidden;
            }
            h1 {
              color: white;
              padding: 10px;
              text-align: center;
            }
            iframe {
              border: none;
              width: 100%;
              height: 100vh;
            }
          </style>
        </head>
        <body>
          <h1>${modelName}</h1>
          <iframe src="${apiEndpoint}" allow="autoplay" allowfullscreen></iframe>
        </body>
      </html>
    `);
    newTab.document.close();

    newTab.onbeforeunload = () => {
      axios
        .post('/api/stop', { apiEndpoint })
        .then((response) => {
          console.log(response.data);
        })
        .catch((error) => {
          console.error(error);
        });
    };
  };

  return (
    <div className="welcome-container">
      <h1 className="welcome-title">Autonomous DigiWatch</h1>
      <p className="welcome-description">
        Select The Model You Want To Test:
      </p>
      <div className="button-container">
        <button onClick={() => handleButtonClick('http://127.0.0.1:5000/video_feed/LRCN', 'LRCN')}>
          LRCN
        </button>
        <button onClick={() => handleButtonClick('http://127.0.0.1:5000', 'GRU')}>
          GRU
        </button>
        <button onClick={() => handleButtonClick('http://127.0.0.1:5000', 'Visual Transformer')}>
          Visual Transformer
        </button>
      </div>
    </div>
  );
}

export default WelcomePage;
