import { lazy, Suspense, useEffect, useState } from 'react';
const App = lazy(() => import('./App'));
import Workbench from './memory/Workbench';
export default function Root() {
  const [explorer,setExplorer]=useState(location.hash==='#explorer');
  useEffect(()=>{const sync=()=>setExplorer(location.hash==='#explorer');addEventListener('hashchange',sync);return()=>removeEventListener('hashchange',sync);},[]);
  return explorer?<><a className="back-workbench" href="#moe">← 显存工作台</a><Suspense fallback={<p>加载激活关系图…</p>}><App/></Suspense></>:<Workbench/>;
}
