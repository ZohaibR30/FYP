import React from 'react';
import { Link } from 'react-router-dom';
import './NextPage.css';

function NextPage() {
  return (
    <div className="next-page">
      <h1>Welcome to the Next Page!</h1>
      <Link to="/">Go back to Welcome Page</Link>
    </div>
  );
}

export default NextPage;
