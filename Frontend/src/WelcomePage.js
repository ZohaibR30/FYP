import React from 'react';
import axios from 'axios';
import './WelcomePage.css';

function WelcomePage() {
  const handleButtonClick = (apiEndpoint) => {
    const newTab = window.open(apiEndpoint, '_blank');
    newTab.onbeforeunload = () => {
      // Send a request to stop the API when the tab is closed
      axios.post('/api/stop', { apiEndpoint })
        .then(response => {
          console.log(response.data);  // Handle the response from the backend
        })
        .catch(error => {
          console.error(error);  // Handle any errors that occurred during the API call
        });
    };
  };

  return (
    <div className="welcome-container">
      <video autoPlay muted loop className="welcome-video">
        <source src="https://thumbs.gfycat.com/ImmaculateHollowAnemonecrab-mobile.mp4" type="video/mp4" />
      </video>
      <div className="content-container">
        <h1 className="welcome-title">Autonomous DigiWatch</h1>
        <p className="welcome-description">
          Select The Model You Want To Test:
        </p>
        <div className="button-container">
          <button onClick={() => handleButtonClick('http://127.0.0.1:5000/video_feed/LRCN')}>
            LRCN
          </button>
          <button onClick={() => handleButtonClick('http://127.0.0.1:5000/video_feed/GRU')}>
            GRU
          </button>
          <button onClick={() => handleButtonClick('http://127.0.0.1:5000/video_feed/ViT')}>
            Visual Transformer
          </button>
        </div>
      </div>
    </div>
  );
}

export default WelcomePage;
