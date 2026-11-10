import { useEffect, useRef, useState } from 'react';
import type { HTMLAttributes } from 'react';

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

export function useGraphViewport(onInteraction: () => void, width = 520, height = 1490) {
  const viewport = useRef<HTMLDivElement>(null);
  const [view, setView] = useState({ scale: 1, x: 0, y: 16 });
  const [dragging, setDragging] = useState(false);
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const suppressClick = useRef(false);
  // Native wheel and resize listeners always invoke the current callbacks.
  const callback = useRef(onInteraction);
  callback.current = onInteraction;

  const center = (scale: number) => {
    setView({ scale, x: (viewport.current!.clientWidth - width * scale) / 2, y: 16 });
    callback.current();
  };
  const fit = () => {
    const el = viewport.current!;
    center(clamp(Math.min((el.clientWidth - 32) / width, (el.clientHeight - 32) / height), .2, 1));
  };
  const zoom = (factor: number, x: number, y: number) => {
    setView(v => {
      const scale = clamp(v.scale * factor, .2, 3);
      const ratio = scale / v.scale;
      return { scale, x: x - (x - v.x) * ratio, y: y - (y - v.y) * ratio };
    });
    callback.current();
  };
  useEffect(() => {
    const el = viewport.current!;
    fit();
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const delta = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? el.clientHeight : 1);
      zoom(Math.exp(-clamp(delta, -200, 200) * .003), e.clientX - rect.left, e.clientY - rect.top);
    };
    el.addEventListener('wheel', wheel, { passive: false });
    const observer = new ResizeObserver(fit);
    observer.observe(el);
    return () => { el.removeEventListener('wheel', wheel); observer.disconnect(); };
  }, [width, height]);

  const endDrag = () => { drag.current = null; setDragging(false); };
  const panHandlers: HTMLAttributes<HTMLDivElement> = {
    onPointerDown(e) {
      if (e.button !== 0) return;
      suppressClick.current = false;
      drag.current = { x: e.clientX, y: e.clientY, moved: false };
    },
    onPointerMove(e) {
      const current = drag.current;
      if (!current) return;
      const dx = e.clientX - current.x, dy = e.clientY - current.y;
      if (!current.moved && Math.hypot(dx, dy) < 4) return;
      if (!current.moved) {
        e.currentTarget.setPointerCapture(e.pointerId);
        suppressClick.current = true;
        setDragging(true);
        callback.current();
      }
      drag.current = { x: e.clientX, y: e.clientY, moved: true };
      setView(v => ({ ...v, x: v.x + dx, y: v.y + dy }));
    },
    onClickCapture(e) {
      if (suppressClick.current) {
        e.preventDefault(); e.stopPropagation(); suppressClick.current = false;
      } else if (!(e.target as HTMLElement).closest('button')) {
        callback.current();
      }
    },
    onPointerUp: endDrag,
    onPointerCancel: endDrag,
    onLostPointerCapture: endDrag,
    onPointerLeave() { if (!drag.current?.moved) drag.current = null; },
  };
  return {
    viewport, view, dragging, fit, panHandlers,
    reset: () => center(1),
    zoomCenter: (factor: number) => zoom(factor, viewport.current!.clientWidth / 2, viewport.current!.clientHeight / 2),
  };
}
