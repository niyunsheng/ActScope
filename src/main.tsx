import React from 'react';
import { createRoot } from 'react-dom/client';
import Root from './Root';
import '@xyflow/react/dist/style.css';
import './style.css';
createRoot(document.getElementById('root')!).render(<React.StrictMode><Root /></React.StrictMode>);
