import { useEffect, useState } from 'react';
import ActivationPage from './activation/Page';
import { models } from './activation/model';
export default function Root() {
  const [hash, setHash] = useState(location.hash);
  useEffect(() => {
    const update = () => setHash(location.hash);
    window.addEventListener('hashchange', update);
    return () => window.removeEventListener('hashchange', update);
  }, []);
  const model = models.find(m => hash === `#/models/${m.id}` || (!hash && m === models[0]));
  if (!model) return <main style={{padding:40}}>Model page not found. <a href="#/models/minimax-h3-dit">Open MiniMax H3 DiT</a></main>;
  return <ActivationPage key={model.id} model={model} />;
}
