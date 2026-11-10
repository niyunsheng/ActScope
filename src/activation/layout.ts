// Stable model-authored rows and columns; recomputation never changes positions.
export function createGrid<C extends string, S extends string>(
  columns: Record<C, number>, sections: Record<S, number>, rowGap = 50, top = 35,
) {
  const y = (row: number) => top + row * rowGap;
  return {
    columns,
    y,
    point: (column: C, row: number, section: S) => ({ x: columns[column], y: y(sections[section] + row) }),
    size: (width: number, lastRow: number) => ({ width, height: y(lastRow) + rowGap }),
  };
}
