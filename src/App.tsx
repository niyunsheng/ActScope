import { useEffect, useMemo, useState } from 'react';
import { ReactFlow, Background, Controls, MiniMap, Handle, Position, type NodeProps, type Node, type Edge, MarkerType } from '@xyflow/react';
import { parseGraph, logicalBytes, savedStorageBytes, formatBytes, type Graph, type Tensor } from './core/graph';
import example from './data/qwen3-layer.json';

const KEY = 'actscope.graph.v1';
const evidenceNames = { hypothesis: '待验证', source_verified: '源码确认', runtime_observed: '运行观测' };
const initial = () => {
  try { const saved = localStorage.getItem(KEY); return { graph: parseGraph(saved ? JSON.parse(saved) : example), error: '' }; }
  catch { return { graph: parseGraph(example), error: '本地草稿无效或不可读取，已加载示例；原草稿未覆盖。' }; }
};
function GraphNode({ data }: NodeProps) {
  return <div className={`graph-node ${data.kind}`}>
    <Handle type="target" position={Position.Left} />
    <small>{String(data.caption)}</small><strong>{String(data.label)}</strong>
    {data.detail ? <span>{String(data.detail)}</span> : null}
    <Handle type="source" position={Position.Right} />
  </div>;
}
const nodeTypes = { item: GraphNode };

export default function App() {
  const [loaded] = useState(initial);
  const [graph, setGraph] = useState<Graph>(loaded.graph);
  const [message, setMessage] = useState(loaded.error);
  const [dirty, setDirty] = useState(false);
  const [selected, setSelected] = useState('normed');
  const [scope, setScope] = useState('All');
  const [backward, setBackward] = useState(true);
  const [query, setQuery] = useState('');
  const [draft, setDraft] = useState('');
  const tensor = graph.tensors.find(t => t.id === selected);
  const operator = graph.operators.find(o => o.id === selected);
  useEffect(() => { setDraft(tensor ? JSON.stringify(tensor, null, 2) : ''); }, [tensor]);
  useEffect(() => {
    if (!dirty) return;
    try { localStorage.setItem(KEY, JSON.stringify(graph)); setMessage('已保存到此浏览器'); }
    catch { setMessage('浏览器保存失败，请导出 JSON 保存修改。'); }
  }, [graph, dirty]);
  const update = (g: Graph) => { setGraph(g); setDirty(true); };
  const { nodes, edges } = useMemo(() => {
    const nodes: Node[] = [
      ...graph.operators.filter(o => backward || o.phase === 'forward').map(o => ({ id: o.id, type: 'item', position: o.position, data: { label: o.name, caption: o.module, kind: o.phase, detail: o.phase === 'backward' ? '保存依赖目标' : 'Forward operator' } })),
      ...graph.tensors.map(t => ({ id: t.id, type: 'item', position: t.position, data: { label: t.name, caption: `${t.id} · ${evidenceNames[t.evidence]}`, kind: 'tensor', detail: `[${t.shape.join(' × ')}] · ${formatBytes(logicalBytes(t))}` } })),
    ].map(n => ({ ...n, selected: n.id === selected, hidden: scope !== 'All' && [...graph.operators, ...graph.tensors].find(x => x.id === n.id)?.module !== scope, style: { opacity: query && !`${n.id} ${n.data.label}`.toLowerCase().includes(query.toLowerCase()) ? 0.25 : 1 } }));
    const edges: Edge[] = [];
    const link = (source: string, target: string, saved = false) => edges.push({ id: `${source}:${target}:${saved}`, source, target, style: { stroke: saved ? '#d99851' : '#648caa', strokeWidth: saved ? 2 : 1.5, strokeDasharray: saved ? '6 5' : undefined }, markerEnd: { type: MarkerType.ArrowClosed, color: saved ? '#d99851' : '#648caa' } });
    graph.tensors.forEach(t => {
      if (t.producer) link(t.producer, t.id);
      [...new Set(t.consumers)].forEach(c => link(t.id, c));
      if (backward) [...new Set(t.savedBy)].forEach(b => link(t.id, b, true));
    });
    const visible = new Set(nodes.filter(n => !n.hidden).map(n => n.id));
    return { nodes, edges: edges.filter(e => visible.has(e.source) && visible.has(e.target)) };
  }, [graph, backward, selected, scope, query]);
  const exported = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(graph, null, 2)], { type: 'application/json' }));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'actscope-graph.json'; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const applyDraft = () => {
    try {
      const changed = JSON.parse(draft) as Tensor;
      if (changed.id !== selected) throw new Error('此面板不支持修改 ID；请通过完整图 JSON 修改引用。');
      update(parseGraph({ ...graph, tensors: graph.tensors.map(t => t.id === selected ? changed : t) }));
    } catch (e) { setMessage(`修改未应用：${e instanceof Error ? e.message : String(e)}`); }
  };
  return <div className="app">
    <header><div className="brand"><b>◈</b><h1>ActScope</h1><span>ACTIVATION EXPLORER</span></div>
      <div className="actions"><span className="stage">{graph.tensors.some(t => t.evidence === 'runtime_observed') ? 'GPU · 实测保存关系' : 'PHASE 01 · 理论草图'}</span>
        <button onClick={async () => {
          try { const response = await fetch(`${import.meta.env.BASE_URL}examples/qwen3-small-observed.json`); if (!response.ok) throw new Error('实测示例未找到'); update(parseGraph(await response.json())); setSelected('observed_47'); setScope('All'); }
          catch (error) { setMessage(`加载失败：${String(error)}`); }
        }}>实测小模型</button><label className="button">导入 JSON<input type="file" accept=".json,application/json" onChange={async e => {
          const file = e.target.files?.[0]; e.target.value = ''; if (!file) return;
          try { update(parseGraph(JSON.parse(await file.text()))); setSelected(''); setScope('All'); }
          catch (err) { setMessage(`导入失败，当前图未改变：${String(err)}`); }
        }}/></label><button onClick={exported}>导出 JSON</button></div></header>
    <section className="intro"><div><div className="eyebrow">MODEL WORKSPACE / LAYER 00</div><h2>{graph.name}</h2><p>追踪激活从哪里来、被谁保存，以及它背后的 storage。</p></div>
      <div className="baseline"><span>{[...new Set(graph.tensors.map(t => t.dtype.toUpperCase()))].join(' / ')} · B={graph.baseline.batch} · S={graph.baseline.sequence}</span><span>TP {graph.baseline.tp} / CP {graph.baseline.cp} / EP {graph.baseline.ep}</span><code>Megatron {graph.baseline.megatronCommit.slice(0, 10)}</code></div></section>
    <section className="metrics"><div><small>图中激活</small><strong>{graph.tensors.length}<em> tensors</em></strong></div><div><small>已声明保存关系对应的 storage · 去重</small><strong>{formatBytes(savedStorageBytes(graph))}</strong></div><div><small>理论峰值</small><strong className="muted">待生命周期建模</strong></div><div><small>证据覆盖</small><strong>{graph.tensors.filter(t => t.evidence !== 'hypothesis').length} / {graph.tensors.length}<em> 已验证</em></strong></div></section>
    <div className="notice">{graph.baseline.note} 当前数值仅覆盖已声明保存关系，不等于完整前向结束保留量或实测显存。</div>
    <main><section className="workspace"><div className="toolbar"><div className="tabs">{['All', ...new Set([...graph.operators, ...graph.tensors].map(x => x.module))].map(s => <button key={s} className={scope === s ? 'active' : ''} onClick={() => setScope(s)}>{s === 'All' ? '整层' : s}</button>)}</div><input aria-label="搜索激活" placeholder="搜索激活 / 算子…" value={query} onChange={e => setQuery(e.target.value)} /><label><input type="checkbox" checked={backward} onChange={e => setBackward(e.target.checked)} />保存依赖</label></div>
      <div className="canvas"><ReactFlow key={`${scope}:${backward}`} nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView minZoom={0.08} maxZoom={2.5} nodesConnectable={false} deleteKeyCode={null} onNodeClick={(_, n) => setSelected(n.id)} onNodesChange={changes => { const positions = new Map<string, { x: number; y: number }>(); for (const c of changes) if (c.type === 'position' && c.position) positions.set(c.id, c.position); if (positions.size) update({ ...graph, tensors: graph.tensors.map(t => positions.has(t.id) ? { ...t, position: positions.get(t.id)! } : t), operators: graph.operators.map(o => positions.has(o.id) ? { ...o, position: positions.get(o.id)! } : o) }); }}><Background gap={24} color="#253646"/><Controls/><MiniMap nodeColor={n => n.data.kind === 'tensor' ? '#53b4a0' : '#587fa2'} maskColor="#0b121b99"/></ReactFlow></div>
      <footer><span>● 算子　<span className="green">● 激活</span>　<span className="orange">┄ 反向保存</span></span><span>滚轮缩放 · 拖拽节点 · 点击查看详情{scope !== 'All' ? ' · 模块筛选隐藏跨模块连线' : ''}</span></footer></section>
      <aside><div className="eyebrow">INSPECTOR</div>{tensor ? <><h3>{tensor.name}</h3><span className="badge">{evidenceNames[tensor.evidence]}</span><dl><dt>Tensor</dt><dd>{tensor.id}</dd><dt>Shape</dt><dd>[{tensor.shape.join(', ')}]</dd><dt>逻辑大小 / dtype</dt><dd>{formatBytes(logicalBytes(tensor))} / {tensor.dtype}</dd><dt>Stride / offset</dt><dd>{tensor.stride ? `[${tensor.stride.join(', ')}]` : '连续布局'} / {tensor.storageOffset ?? 0}</dd><dt>Storage</dt><dd>{tensor.storageId} · {formatBytes(graph.storages.find(s => s.id === tensor.storageId)!.bytes)}</dd><dt>生产者</dt><dd>{tensor.producer || '层外输入 / 未定位'}</dd><dt>前向消费者</dt><dd>{tensor.consumers.join(', ') || '未声明'}</dd><dt>反向保存者</dt><dd className="orange">{tensor.savedBy.join(', ') || '未声明（不代表无需保存）'}</dd><dt>共享 storage 的激活</dt><dd>{graph.tensors.filter(t => t.storageId === tensor.storageId).map(t => t.id).join(', ')}</dd></dl><p className="note">{tensor.note}</p><details><summary>编辑激活属性（JSON）</summary><p className="note">可修改 shape、dtype、保存者等；扩大 storage 请导出完整图编辑。</p><textarea aria-label="激活属性 JSON" spellCheck={false} value={draft} onChange={e => setDraft(e.target.value)}/><button onClick={applyDraft}>校验并应用</button></details></> : operator ? <><h3>{operator.name}</h3><p>{operator.module} / {operator.phase}</p><p>本阶段通过完整图 JSON 编辑算子；画布支持拖动保存位置。</p></> : <p>点击激活查看保存关系和共享存储。</p>}<div className="next"><small>NEXT MILESTONES</small><p>02 源码核对与完整编辑</p><p>03 生命周期与显存曲线</p><p>04 Recompute / Offload</p></div></aside></main>
    <div role="status" className="status">{message || '示例已加载 · 修改后自动保存至此浏览器'}<button onClick={() => { update(parseGraph(example)); setSelected('normed'); setScope('All'); }}>恢复示例</button></div>
  </div>;
}
