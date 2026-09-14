import { editMatrix, optimalEditPaths, type EditCoordinate } from "./algorithms.ts";
import type { Topic } from "./data/topics.ts";
import { bindMethodWorkspace, escapeHtml, math, methodPage, type MethodTrace } from "./methodWorkspace.ts";

export function editDistancePage(topic: Topic): string {
  return methodPage(topic, {
    title: "Turn one string into another.",
    description: "Build the distance matrix, compare each candidate, and follow any minimum path through the exact calculation.",
    inputs: `<label>Source<input id="edit-left" name="edit-left" value="BIO" spellcheck="false" /></label><label>Target<input id="edit-right" name="edit-right" value="BIFE" spellcheck="false" /></label>`,
    notes: "Each Unicode code point occupies one cell. Insertions, deletions and substitutions cost 1; matches cost 0. Empty strings are supported. Up to 48 characters per string.",
    examples: [
      { label: "Two minimum paths", values: { "edit-left": "BIO", "edit-right": "BIFE" } },
      { label: "Classic example", values: { "edit-left": "kitten", "edit-right": "sitting" } },
      { label: "Empty source", values: { "edit-left": "", "edit-right": "cat" } },
    ],
  });
}

export function createEditDistanceTrace(left: string, right: string): MethodTrace {
  const source = Array.from(left);
  const target = Array.from(right);
  if (source.length > 48 || target.length > 48) throw new Error("Use up to 48 characters per string so every matrix cell remains inspectable.");
  const matrix = editMatrix(left, right);
  const enumerated = optimalEditPaths(matrix, 65);
  const paths = enumerated.slice(0, 64);
  const order: EditCoordinate[] = [[0, 0]];
  for (let column = 1; column <= source.length; column += 1) order.push([0, column]);
  for (let row = 1; row <= target.length; row += 1) order.push([row, 0]);
  for (let row = 1; row <= target.length; row += 1) {
    for (let column = 1; column <= source.length; column += 1) order.push([row, column]);
  }
  const fillIndexes = new Map(order.map((coordinate, index) => [coordinate.join(","), index]));
  const distance = matrix[target.length][source.length].value;
  const operation = (from: EditCoordinate, to: EditCoordinate): string => {
    if (to[0] > from[0] && to[1] > from[1]) return source[from[1]] === target[from[0]]
      ? `Match “${source[from[1]]}”` : `Replace “${source[from[1]]}” with “${target[from[0]]}”`;
    return to[0] > from[0] ? `Insert “${target[from[0]]}”` : `Delete “${source[from[1]]}”`;
  };
  const trace = (selectedPath: number): MethodTrace => {
    const path = paths[selectedPath];
    const pathIndexes = new Map(path.map((coordinate, index) => [coordinate.join(","), index]));
    return {
      result: `Minimum edit distance: ${distance}`,
      summary: `${source.length} source characters → ${target.length} target characters · ${matrix.length} × ${matrix[0].length} matrix · ${enumerated.length > 64 ? "More than 64 minimum paths; first 64 available" : `${paths.length} minimum ${paths.length === 1 ? "path" : "paths"}`}`,
      steps: [
        ...order.map(([row, column]) => ({ title: `D(${row}, ${column})`, stage: row === 0 || column === 0 ? "Initialize" : "Fill matrix" })),
        ...path.map((coordinate, index) => ({ title: index === 0 ? "Start at D(0, 0)" : operation(path[index - 1], coordinate), stage: "Minimum path" })),
      ],
      choose: (name, value) => name === "path" ? trace(Math.max(0, Math.min(paths.length - 1, Number(value) || 0))) : trace(selectedPath),
      render(index) {
        const followingPath = index >= order.length;
        const pathStep = index - order.length;
        const [row, column] = followingPath ? path[pathStep] : order[index];
        const value = matrix[row][column].value;
        const arrows = { diagonal: "↖", up: "↑", left: "←" };
        const visual = `<div class="ml-visual-heading"><h2>Distance matrix</h2><small>Rows: target · columns: source</small></div>
          <div class="ml-inline-controls"><label>Minimum path<select data-choice="path" aria-label="Minimum path">${paths.map((_, i) => `<option value="${i}" ${i === selectedPath ? "selected" : ""}>Path ${i + 1}</option>`).join("")}</select></label><button type="button" data-jump="${followingPath ? 0 : order.length}">${followingPath ? "Inspect matrix construction" : "Inspect minimum path"}</button></div>
          <p class="ml-hint">Distance between target prefixes (rows) and source prefixes (columns).</p><div class="ml-table-scroll"><table class="ml-matrix" aria-label="Distance between target and source prefixes"><thead><tr><th scope="col">target / source</th><th scope="col">∅</th>${source.map((character) => `<th scope="col">${escapeHtml(character)}</th>`).join("")}</tr></thead><tbody>${matrix.map((cells, r) => `<tr><th scope="row">${r === 0 ? "∅" : escapeHtml(target[r - 1])}</th>${cells.map((cell, c) => {
            const key = `${r},${c}`;
            const fillIndex = fillIndexes.get(key)!;
            const routeIndex = pathIndexes.get(key);
            const visible = followingPath || fillIndex <= index;
            const current = r === row && c === column;
            const dependency = !followingPath && row > 0 && column > 0 && ((r === row - 1 && c === column) || (r === row && c === column - 1) || (r === row - 1 && c === column - 1));
            return `<td class="${current ? "current" : dependency ? "dependency" : followingPath && routeIndex !== undefined ? "on-path" : ""} ${visible ? "" : "pending"}"><button type="button" data-jump="${followingPath && routeIndex !== undefined ? order.length + routeIndex : fillIndex}" aria-label="Inspect D(${r}, ${c})${visible ? `, distance ${cell.value}` : ""}" aria-pressed="${current}"><strong>${visible ? cell.value : "·"}</strong>${visible ? `<small>${cell.from.map((direction) => arrows[direction]).join(" ") || "start"}</small>` : ""}</button></td>`;
          }).join("")}</tr>`).join("")}</tbody></table></div><p class="ml-legend">Select a cell to inspect it. Arrows point to every minimum predecessor.${followingPath ? " Highlighted cells form the selected minimum path." : " Dots mark cells that have not yet been filled at this step."}</p>
          ${followingPath ? `<h3>String transformation</h3><div class="ml-token-row" aria-label="Selected minimum path">${path.map((coordinate, i) => `${i ? '<span aria-hidden="true">→</span>' : ""}<button type="button" data-jump="${order.length + i}" aria-pressed="${i === pathStep}" class="${i === pathStep ? "current" : ""}"><small>${i === 0 ? "Start" : escapeHtml(operation(path[i - 1], coordinate))}</small><strong>${escapeHtml(target.slice(0, coordinate[0]).join("") + source.slice(coordinate[1]).join("")) || "∅"}</strong><small>D(${coordinate[0]}, ${coordinate[1]}) = ${matrix[coordinate[0]][coordinate[1]].value}</small></button>`).join("")}</div>` : ""}`;
        const prefixes = `<dl class="ml-facts"><div><dt>Source prefix</dt><dd>${escapeHtml(source.slice(0, column).join("")) || "∅"}</dd></div><div><dt>Target prefix</dt><dd>${escapeHtml(target.slice(0, row).join("")) || "∅"}</dd></div><div><dt>Minimum cost</dt><dd>${value}</dd></div></dl>`;
        let detail: string;
        if (followingPath) {
          const previous = path[pathStep - 1];
          const cost = previous ? value - matrix[previous[0]][previous[1]].value : 0;
          detail = `<h3>${previous ? escapeHtml(operation(previous, [row, column])) : "Begin with two empty prefixes"}</h3>${previous ? math(`D_{${row},${column}}=D_{${previous[0]},${previous[1]}}+${cost}=${value}`) : math("D_{0,0}=0")}${prefixes}<p>${pathStep === path.length - 1 ? `The path is complete. Its operations cost ${distance}, the minimum for these strings.` : "Each move follows a recorded minimum predecessor in reverse, building the target from the source."}</p>`;
        } else if (row === 0 || column === 0) {
          detail = `<h3>${row === 0 && column === 0 ? "Start with empty prefixes" : row === 0 ? "Delete the source prefix" : "Insert the target prefix"}</h3>${math(`D_{${row},${column}}=${value}`)}${prefixes}<p>${row === 0 && column === 0 ? "No operations are needed to transform an empty string into an empty string." : row === 0 ? `${column} ${column === 1 ? "deletion leaves" : "deletions leave"} an empty target. Each deletion costs 1.` : `${row} ${row === 1 ? "insertion builds" : "insertions build"} this target from an empty source. Each insertion costs 1.`}</p>`;
        } else {
          const substitution = source[column - 1] === target[row - 1] ? 0 : 1;
          const candidates = [
            { label: "Insert", r: row - 1, c: column, cost: 1 },
            { label: "Delete", r: row, c: column - 1, cost: 1 },
            { label: substitution === 0 ? "Match" : "Replace", r: row - 1, c: column - 1, cost: substitution },
          ];
          detail = `<h3>Compare three candidate costs</h3>${math(`D_{${row},${column}}=\\min\\{${candidates.map(({ r, c, cost }) => matrix[r][c].value + cost).join(",")}\\}=${value}`)}<table class="ml-candidates"><thead><tr><th>Operation</th><th>Predecessor + cost</th><th>Total</th></tr></thead><tbody>${candidates.map(({ label, r, c, cost }) => `<tr class="${matrix[r][c].value + cost === value ? "winner" : ""}"><th>${label}</th><td><button type="button" data-jump="${fillIndexes.get(`${r},${c}`)}">D(${r}, ${c})</button> = ${matrix[r][c].value} + ${cost}</td><td>${matrix[r][c].value + cost}${matrix[r][c].value + cost === value ? " ✓" : ""}</td></tr>`).join("")}</tbody></table>${prefixes}<p>${escapeHtml(`“${source[column - 1]}” ${substitution === 0 ? "matches" : "differs from"} “${target[row - 1]}”, so the diagonal adds ${substitution}.`)} ${matrix[row][column].from.length > 1 ? "Tied minima retain all their backpointers." : "The cheapest predecessor supplies this cell’s backpointer."}</p>`;
        }
        return { visual, detail };
      },
    };
  };
  return trace(0);
}

export function bindEditDistance(): void {
  bindMethodWorkspace(() => createEditDistanceTrace(document.querySelector<HTMLInputElement>("#edit-left")!.value, document.querySelector<HTMLInputElement>("#edit-right")!.value));
}
