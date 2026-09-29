"""Graphviz drawing of the ordered local facet occurrences (not of the word trie).

Nothing is computed unless a closure is requested. Parallel arrows are
expected: do not apply transitive reduction or merge duplicate arrows.
"""
from __future__ import annotations
from pathlib import Path
import os
import subprocess


def visualize_dag(tree, filename=None, *, format='svg', root_cells=None,
                  max_nodes=500, show_source=False, show_words=False,
                  view=False, engine='dot'):
    """Return DOT text, or write DOT + render and return the rendered Path.

    `root_cells` selects a closure, without changing the stored object.
    A dashed edge denotes a facet collapsed to a component point. Each edge
    carries its original local facet index, even when targets coincide.
    `format='dot'` only writes the source and needs no Graphviz executable.
    The default node limit is a safety limit, never silent truncation.
    """
    if engine not in {'dot', 'neato', 'fdp', 'sfdp', 'circo', 'twopi'}:
        raise ValueError('unsupported Graphviz layout engine')
    if format not in {'svg', 'png', 'pdf', 'dot'}:
        raise ValueError('format must be svg, png, pdf or dot')
    ids = tuple(tree.cell_ids()) if root_cells is None else tree.closure(root_cells)
    if max_nodes is not None and len(ids) > max_nodes:
        raise ValueError(f'{len(ids)} cells exceed max_nodes={max_nodes}; select root_cells or raise the limit explicitly')
    def esc(text):
        return str(text).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    lines = ['digraph QF {', '  graph [rankdir=TB, splines=true];',
             '  node [shape=box, fontname="Helvetica"];',
             '  edge [fontname="Helvetica", fontsize=10];']
    levels = {}
    chosen = set(ids)
    for i in ids:
        cell = tree.cell(i)
        label = f'{"D" if cell.is_component else "cell"} {i} | dim {cell.dimension}'
        if show_source:
            label += '\n' + (f'class {cell.component_vertices}' if cell.is_component else f'source {cell.source_vertices}')
        if show_words:
            label += f'\nword {tree.word(i)}'
        shape = 'doublecircle' if cell.is_component else 'box'
        lines.append(f'  c{i} [label="{esc(label)}", shape={shape}];')
        levels.setdefault(cell.dimension, []).append(i)
    for d in sorted(levels, reverse=True):
        lines.append('  { rank=same; ' + '; '.join(f'c{i}' for i in levels[d]) + '; }')
    for i in ids:
        for f in tree.facets(i):
            if f.target not in chosen:
                raise RuntimeError('selected cells are not face closed')
            # Do not identify repeated targets or erase a zero algebraic boundary.
            style = 'dashed' if f.is_constant else 'solid'
            label = f'd{f.local_index}' + (' constant' if f.is_constant else '')
            lines.append(f'  c{i} -> c{f.target} [label="{label}", style={style}];')
    lines.append('}')
    dot = '\n'.join(lines) + '\n'
    if filename is None:
        return dot
    target = Path(os.fspath(filename))
    if target.suffix.lower() in {'.svg', '.png', '.pdf', '.dot'}:
        target = target.with_suffix('')
    target.parent.mkdir(parents=True, exist_ok=True)
    source = target.with_suffix('.dot')
    source.write_text(dot, encoding='utf-8')
    if format == 'dot':
        return source
    rendered = target.with_suffix('.' + format)
    try:
        subprocess.run([engine, '-T' + format, str(source), '-o', str(rendered)],
                       check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f'Graphviz executable {engine!r} is missing; DOT was saved to {source}') from exc
    if view:
        import webbrowser
        webbrowser.open(rendered.resolve().as_uri())
    return rendered
