/* weldpath failure viewer.
 *
 * Reads window.WELDPATH_TREE, written beside it by weldpath.treeview, and draws the
 * searches that failed: the cell as the planner collided against it, the two trees of each
 * pair that grew towards each other, and the gun and arm wherever a node is clicked.
 *
 * A search is a cartesian tree or a joint tree -- solves[].kind, absent in files written
 * before the joint tree was wired up -- and the kind is worth knowing while reading one,
 * because the same drawn edge means different things.  A cartesian edge is a straight tool
 * move, checked pose by pose; a joint edge is a chord in joint space, which bows through
 * tool space and bows exactly where the arm swung.  Both are drawn through the stations the
 * search validated, never as a line between two nodes.
 *
 * No dependencies on purpose.  This has to open from a file:// path on a machine with no
 * internet, so there is nothing to fetch and nothing to install -- raw WebGL 1, which every
 * browser this will meet has had for a decade.
 *
 * Conventions: everything is in the manifest's own length units (millimetres on this cell),
 * which is what the data file carries, so distances read on screen match the planner's logs.
 */
'use strict';

(function () {

const DATA = window.WELDPATH_TREE;

// ---------------------------------------------------------------- small matrix helpers
function mul(a, b) {                       // column-major 4x4, a * b
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++)
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] +
                     a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  return o;
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), d = near - far;
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0,
                           0, 0, (far + near) / d, -1, 0, 0, 2 * far * near / d, 0]);
}

function lookAt(eye, at, up) {
  const z = norm(sub(eye, at)), x = norm(cross(up, z)), y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
}

function fromQuat(q, p) {                  // [x,y,z,w] + position -> column-major 4x4
  const [x, y, z, w] = q;
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  return new Float32Array([
    1 - (yy + zz), xy + wz, xz - wy, 0,
    xy - wz, 1 - (xx + zz), yz + wx, 0,
    xz + wy, yz - wx, 1 - (xx + yy), 0,
    p[0], p[1], p[2], 1]);
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]];
function norm(a) {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}

// --------------------------------------------------------------------------- the page
const STYLE = `
:root { color-scheme: dark; }
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; overflow: hidden;
  background: #15171a; color: #e8e8ea;
  font: 13px/1.45 ui-sans-serif, system-ui, "Segoe UI", sans-serif; }
canvas { display: block; width: 100%; height: 100%; cursor: grab; }
canvas.dragging { cursor: grabbing; }
#panel { position: fixed; top: 0; left: 0; width: 320px; max-height: 100%;
  overflow-y: auto; padding: 14px 16px 18px; background: rgba(20,22,25,.92);
  border-right: 1px solid #2c3035; backdrop-filter: blur(6px); }
#panel h1 { font-size: 14px; margin: 0 0 2px; letter-spacing: .01em; }
#panel h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
  color: #8b929b; margin: 16px 0 6px; font-weight: 600; }
.fail { color: #ffb4a8; background: #2e1d1b; border: 1px solid #5d2f29;
  border-radius: 5px; padding: 7px 9px; margin: 8px 0 0; font-size: 12px;
  white-space: pre-wrap; word-break: break-word; }
label.row { display: flex; align-items: center; gap: 7px; padding: 2px 0;
  cursor: pointer; user-select: none; }
label.row input { accent-color: #6aa9ff; margin: 0; }
label.row.sub { padding-left: 17px; }
label.row .n { color: #80868f; font-variant-numeric: tabular-nums; }
.sw { width: 11px; height: 11px; border-radius: 2px; flex: 0 0 auto; }
.solve { border-top: 1px solid #23262b; padding-top: 4px; margin-top: 4px; }
.solve:first-of-type { border-top: 0; margin-top: 0; }
.lbl { color: #8b929b; font-size: 11px; padding: 1px 0 2px 17px;
  word-break: break-word; }
.bar { display: flex; gap: 6px; margin: 4px 0 2px; }
.bar button { flex: 1 1 0; background: #23262b; color: #c8ced6; cursor: pointer;
  border: 1px solid #32363d; border-radius: 4px; padding: 3px 0; font: inherit;
  font-size: 11px; }
.bar button:hover { background: #2b2f36; }
.bar button.on { background: #1d4f45; border-color: #2c7d6c; color: #d8fff6; }
#mlabel { position: fixed; transform: translate(-50%,-50%); pointer-events: none;
  padding: 2px 7px; border-radius: 4px; background: #0d2b26e8; color: #7df0d4;
  border: 1px solid #2c7d6c; font-variant-numeric: tabular-nums; font-size: 12px;
  white-space: nowrap; display: none; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
td { padding: 1px 0; vertical-align: top; }
td.k { color: #8b929b; padding-right: 8px; white-space: nowrap; }
td.v { text-align: right; word-break: break-all; }
#ramp { height: 9px; border-radius: 5px; margin: 5px 0 3px;
  background: linear-gradient(90deg,#ff4d4d,#ffd24d,#8ae234,#36c76b); }
.ends { display: flex; justify-content: space-between; color: #8b929b; font-size: 11px; }
#hint { position: fixed; bottom: 0; left: 320px; right: 0; padding: 7px 14px;
  color: #80868f; font-size: 11px; background: linear-gradient(transparent,#15171ad0); }
#q { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 11px;
  color: #b9c0c9; word-break: break-all; }
.none { color: #80868f; font-style: italic; }
`;

function el(tag, attrs, kids) {
  const n = document.createElement(tag);
  for (const k in (attrs || {})) {
    if (k === 'class') n.className = attrs[k];
    else if (k === 'text') n.textContent = attrs[k];
    else if (k === 'html') n.innerHTML = attrs[k];
    else n.setAttribute(k, attrs[k]);
  }
  for (const c of (kids || [])) n.appendChild(c);
  return n;
}

function fail(message) {
  document.head.appendChild(el('style', { text: STYLE }));
  document.body.appendChild(el('div', { id: 'panel' }, [
    el('h1', { text: 'weldpath failure viewer' }),
    el('p', { class: 'fail', text: message }),
  ]));
}

if (!DATA || DATA.format !== 'weldpath-treeview/2') {
  fail('No data loaded. Open the <segment>.treeview.html file that was written beside ' +
       'this script, not this script itself.');
  return;
}

// -------------------------------------------------------------------------- the colours
const C = {
  statik: [0.42, 0.45, 0.50],
  arm:    [0.33, 0.52, 0.80],
  gun:    [0.96, 0.72, 0.28],
  start:  [0.38, 0.72, 1.00],
  goal:   [0.78, 0.52, 1.00],
  chord:  [0.55, 0.58, 0.62],
  pick:   [1.00, 0.30, 0.30],
  // Deliberately not near any of the above.  A measurement is laid over everything else
  // in the scene, so its one job is to not be mistaken for a node, the arm or the gun.
  rule:   [0.20, 0.95, 0.78],
};

/* One hue per solve, so several shown at once stay tellable apart, and within a solve the
 * start tree is the brighter of the pair -- the same distinction the two-tree file made
 * with two fixed colours, which cannot stretch to a dozen searches.  Golden-angle steps
 * rather than an even split of the circle, because the number of solves is not known until
 * the file is read and an even split would recolour every search when one more arrives. */
function hsl(h, s, l) {
  const f = n => {
    const k = (n + h * 12) % 12;
    return l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}
function solveColour(i, which) {
  if (SOLVES.length === 1) return which === 'start' ? C.start : C.goal;
  const h = ((0.137 + i * 0.381) % 1);
  return which === 'start' ? hsl(h, 0.82, 0.68) : hsl(h, 0.55, 0.45);
}

/* Clearance to colour: red where the move is at the collision margin, green where the
 * query has run out of range and is reporting "nothing seen" rather than a distance.
 * Saturating at the probe is the honest end of the ramp -- past it every state reads the
 * same, so a greener node there would be claiming a measurement nobody made. */
function heat(mm) {
  if (mm === null || mm === undefined) return [0.45, 0.47, 0.50];
  const lo = DATA.clearance.margin, hi = Math.max(DATA.clearance.probe, lo + 1e-6);
  let t = (mm - lo) / (hi - lo);
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const stops = [[1, 0.30, 0.30], [1, 0.82, 0.30], [0.54, 0.89, 0.20], [0.21, 0.78, 0.42]];
  const x = t * (stops.length - 1), i = Math.min(Math.floor(x), stops.length - 2);
  const f = x - i, a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// ------------------------------------------------------------------------------- WebGL
const canvas = el('canvas');
document.head.appendChild(el('style', { text: STYLE }));
document.body.appendChild(canvas);

const gl = canvas.getContext('webgl', { antialias: true, depth: true });
if (!gl) { fail('This browser will not give us a WebGL context.'); return; }

function shader(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(s));
  return s;
}

function program(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, shader(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, shader(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(p));
  return p;
}

/* Two-sided lighting, deliberately.  These hulls come out of a convex decomposition and
 * their winding is not guaranteed, so a one-sided shader renders half of the cell black
 * and the viewer looks broken rather than informative. */
const meshProg = program(`
  attribute vec3 aPos; attribute vec3 aNrm;
  uniform mat4 uMVP, uModel; varying vec3 vN, vP;
  void main() {
    vN = mat3(uModel) * aNrm;
    vP = (uModel * vec4(aPos, 1.0)).xyz;
    gl_Position = uMVP * uModel * vec4(aPos, 1.0);
  }`, `
  precision mediump float;
  uniform vec3 uColor, uEye; uniform float uAlpha;
  varying vec3 vN, vP;
  void main() {
    vec3 n = normalize(vN), v = normalize(uEye - vP);
    float d = abs(dot(n, v));
    vec3 c = uColor * (0.34 + 0.66 * d) + vec3(pow(d, 18.0) * 0.16);
    gl_FragColor = vec4(c, uAlpha);
  }`);

const flatProg = program(`
  attribute vec3 aPos; attribute vec3 aCol;
  uniform mat4 uMVP; uniform float uSize; varying vec3 vC;
  void main() { vC = aCol; gl_PointSize = uSize; gl_Position = uMVP * vec4(aPos, 1.0); }`, `
  precision mediump float; varying vec3 vC; uniform float uAlpha;
  void main() { gl_FragColor = vec4(vC, uAlpha); }`);

const mA = {
  pos: gl.getAttribLocation(meshProg, 'aPos'), nrm: gl.getAttribLocation(meshProg, 'aNrm'),
  mvp: gl.getUniformLocation(meshProg, 'uMVP'), model: gl.getUniformLocation(meshProg, 'uModel'),
  color: gl.getUniformLocation(meshProg, 'uColor'), eye: gl.getUniformLocation(meshProg, 'uEye'),
  alpha: gl.getUniformLocation(meshProg, 'uAlpha'),
};
const fA = {
  pos: gl.getAttribLocation(flatProg, 'aPos'), col: gl.getAttribLocation(flatProg, 'aCol'),
  mvp: gl.getUniformLocation(flatProg, 'uMVP'), size: gl.getUniformLocation(flatProg, 'uSize'),
  alpha: gl.getUniformLocation(flatProg, 'uAlpha'),
};

function buffer(data) {
  const b = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, b);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  return b;
}

// ------------------------------------------------------- turning the pieces into meshes
/* Each convex piece becomes its own flat-shaded triangle soup: vertices are duplicated per
 * face so the normal is the face's own.  Smoothing a hull would round the very edges that
 * say where it ends, which is what is being looked at here. */
function soup(pieces) {
  let tris = 0;
  for (const p of pieces) tris += p.f.length / 3;
  const pos = new Float32Array(tris * 9), nrm = new Float32Array(tris * 9);
  let k = 0;
  for (const p of pieces) {
    for (let i = 0; i < p.f.length; i += 3) {
      const a = p.f[i] * 3, b = p.f[i + 1] * 3, c = p.f[i + 2] * 3;
      const A = [p.v[a], p.v[a + 1], p.v[a + 2]];
      const B = [p.v[b], p.v[b + 1], p.v[b + 2]];
      const D = [p.v[c], p.v[c + 1], p.v[c + 2]];
      const n = norm(cross(sub(B, A), sub(D, A)));
      for (const V of [A, B, D]) {
        pos[k] = V[0]; pos[k + 1] = V[1]; pos[k + 2] = V[2];
        nrm[k] = n[0]; nrm[k + 1] = n[1]; nrm[k + 2] = n[2];
        k += 3;
      }
    }
  }
  return { pos: buffer(pos), nrm: buffer(nrm), count: tris * 3, raw: pos };
}

const statics = DATA.geometry.static.map(o => ({ name: o.name, category: o.category,
                                                 mesh: soup(o.pieces) }));
const moving = {};
for (const m of DATA.geometry.moving) moving[m.name] = soup(m.pieces);
const gunSet = new Set(DATA.links.gun || []);

// ------------------------------------------------------------------- the sets of states
/* Trees and the chord are the same thing to the viewer: a list of states with a tool
 * position, a clearance and maybe a baked pose, plus polylines joining them.  The chord is
 * in the list because it is the one set that always exists -- a segment can fail with no
 * tree at all, and the file is written anyway. */
const SOLVES = DATA.solves || [];
/* Older files carry no kind.  They were all cartesian -- it was the only tree there was --
 * but saying so for a file that never claimed it would be the viewer inventing a reading,
 * so an unlabelled search is shown without a kind and not as one. */
const KIND = { cartesian: 'cartesian tree', joint: 'joint tree' };
const kindOf = info => (info && info.kind) ? (KIND[info.kind] || info.kind) : '';
/* The biggest search is the one shown first.  Every search is in the file and one click
 * away, but a dozen overlapping trees on opening is a picture of nothing. */
let biggest = 0;
for (let i = 1; i < SOLVES.length; i++)
  if ((SOLVES[i].nodes || 0) > (SOLVES[biggest].nodes || 0)) biggest = i;

const sets = [];
for (const t of (DATA.trees || [])) {
  const s = t.solve || 0;
  sets.push({
    name: t.name === 'start' ? 'tree from the start' : 'tree from the goal',
    key: t.name, solve: s, colour: solveColour(s, t.name),
    count: t.count, tcp: t.tcp, clearance: t.clearance, pose: t.pose,
    q: t.q, parent: t.parent, edges: t.edges,
    on: SOLVES.length <= 1 || s === biggest,
  });
}
if (DATA.chord) {
  const n = DATA.chord.tcp.length;
  const edges = [];
  for (let i = 1; i < n; i++)
    edges.push({ a: i - 1, b: i, path: [DATA.chord.tcp[i - 1], DATA.chord.tcp[i]] });
  sets.push({
    name: 'the straight chord', key: 'chord', solve: -1, colour: C.chord, count: n,
    tcp: DATA.chord.tcp, clearance: DATA.chord.clearance, pose: DATA.chord.pose,
    q: DATA.chord.q, parent: null, edges: edges, on: sets.length === 0,
  });
}

for (const s of sets) {
  const pts = new Float32Array(s.count * 3), cols = new Float32Array(s.count * 3);
  for (let i = 0; i < s.count; i++) {
    const p = s.tcp[i], c = heat(s.clearance[i]);
    pts[i * 3] = p[0]; pts[i * 3 + 1] = p[1]; pts[i * 3 + 2] = p[2];
    cols[i * 3] = c[0]; cols[i * 3 + 1] = c[1]; cols[i * 3 + 2] = c[2];
  }
  s.pointBuf = buffer(pts); s.pointCol = buffer(cols);
  s.pointColSet = buffer(nodeColours(s, true));

  let segs = 0;
  for (const e of s.edges) segs += Math.max(0, e.path.length - 1);
  const lp = new Float32Array(segs * 6), lc = new Float32Array(segs * 6);
  let k = 0;
  for (const e of s.edges) {
    for (let i = 1; i < e.path.length; i++) {
      const a = e.path[i - 1], b = e.path[i];
      lp[k] = a[0]; lp[k + 1] = a[1]; lp[k + 2] = a[2];
      lp[k + 3] = b[0]; lp[k + 4] = b[1]; lp[k + 5] = b[2];
      for (let j = 0; j < 6; j++) lc[k + j] = s.colour[j % 3];
      k += 6;
    }
  }
  s.lineBuf = buffer(lp); s.lineCol = buffer(lc); s.lineCount = segs * 2;
}

// ---------------------------------------------------------------------------- the camera
let bmin = [1e9, 1e9, 1e9], bmax = [-1e9, -1e9, -1e9];
function grow(p) {
  for (let i = 0; i < 3; i++) {
    if (p[i] < bmin[i]) bmin[i] = p[i];
    if (p[i] > bmax[i]) bmax[i] = p[i];
  }
}
for (const o of statics)
  for (let i = 0; i < o.mesh.raw.length; i += 3)
    grow([o.mesh.raw[i], o.mesh.raw[i + 1], o.mesh.raw[i + 2]]);
for (const s of sets) for (const p of s.tcp) grow(p);
if (bmin[0] > bmax[0]) { bmin = [-500, -500, -500]; bmax = [500, 500, 500]; }

const centre = [(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2];
const span = Math.max(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2], 1);

const cam = { yaw: -0.9, pitch: 0.5, dist: span * 1.5, at: centre.slice() };

function eyePos() {
  const cp = Math.cos(cam.pitch);
  return [cam.at[0] + cam.dist * cp * Math.sin(cam.yaw),
          cam.at[1] + cam.dist * cp * Math.cos(cam.yaw),
          cam.at[2] + cam.dist * Math.sin(cam.pitch)];
}

// ------------------------------------------------------------------------------ the UI
const opts = { statics: true, translucent: false, arm: true, gun: true,
               edges: true, nodes: true, chordPose: true, bySolve: false };
let picked = null;          // {set, index}
const measure = { on: false, pts: [] };     // the rule; see "measuring" below

const panel = el('div', { id: 'panel' });
document.body.appendChild(panel);
const hint = el('div', { id: 'hint' });
document.body.appendChild(hint);
const mlabel = el('div', { id: 'mlabel' });
document.body.appendChild(mlabel);

function showHint() {
  hint.textContent = measure.on
    ? 'measuring · click two points to measure between them · a third starts again · ' +
      'points snap to nodes · Esc clears · m leaves'
    : 'drag to orbit · wheel to zoom · right-drag or shift-drag to pan · ' +
      'click a node to put the robot there · m measures · R resets the view';
}
showHint();

/* Clearance is what the node colours are for, and it is the right default: it is the
 * reading that explains a failure.  But with several searches shown at once every node is
 * on the same ramp and nothing says which search it belongs to, so this repaints them in
 * their set's colour instead.  Built once per set either way -- the cost is a second
 * colour buffer, not a second draw. */
function nodeColours(s, bySolve) {
  const cols = new Float32Array(s.count * 3);
  for (let i = 0; i < s.count; i++) {
    const c = bySolve ? s.colour : heat(s.clearance[i]);
    cols[i * 3] = c[0]; cols[i * 3 + 1] = c[1]; cols[i * 3 + 2] = c[2];
  }
  return cols;
}

function checkbox(label, key, colour) {
  const input = el('input', { type: 'checkbox' });
  input.checked = opts[key];
  input.onchange = () => { opts[key] = input.checked; draw(); };
  const kids = [input];
  if (colour) kids.push(el('span', { class: 'sw' }));
  kids.push(el('span', { text: label }));
  const row = el('label', { class: 'row' }, kids);
  if (colour) row.querySelector('.sw').style.background =
    `rgb(${colour.map(v => Math.round(v * 255)).join(',')})`;
  return row;
}

function rows(pairs) {
  return el('table', {}, pairs.map(([k, v]) =>
    el('tr', {}, [el('td', { class: 'k', text: k }), el('td', { class: 'v', text: v })])));
}

const readout = el('div');
const mread = el('div');

/* Declared beside the readout rather than with the rest of the measuring code, for the
 * same reason `measure` is: `buildPanel` calls this, and it runs long before that point
 * in the file. */
function showMeasure() {
  mread.innerHTML = '';
  if (!measure.on) {
    mread.appendChild(el('p', { class: 'none', text:
      'Off. Press m, or the button above.' }));
    return;
  }
  if (!measure.pts.length) {
    mread.appendChild(el('p', { class: 'none', text:
      'Click a node, or anywhere on the geometry, for the first point.' }));
    return;
  }
  const fix = p => p.map(v => v.toFixed(1)).join(', ');
  const a = measure.pts[0], b = measure.pts[1];
  const pairs = [['from', a.what || 'a surface'], ['', fix(a.p)]];
  if (!b) {
    pairs.push(['to', 'click the second point']);
  } else {
    pairs.push(['to', b.what || 'a surface'], ['', fix(b.p)]);
    const d = sub(b.p, a.p);
    pairs.push(['distance', `${measureLength().toFixed(1)} ${DATA.units}`],
               ['Δ x, y, z', fix(d)]);
  }
  mread.appendChild(rows(pairs));
  // Said once, beside the number, rather than left to be remembered: the figure is not
  // what a CAD measurement of the same two places would give.
  if ((a && a.surface) || (b && b.surface))
    mread.appendChild(el('p', { class: 'none', text:
      'A surface point is on the convex hull the planner collides against, which stands ' +
      'a little outside the CAD it was cut from.' }));
}

function buildPanel() {
  panel.innerHTML = '';
  panel.appendChild(el('h1', { text: DATA.segment }));
  panel.appendChild(el('div', { class: 'fail', text: DATA.failure || 'no reason recorded' }));

  panel.appendChild(el('h2', { text: 'show' }));
  panel.appendChild(checkbox('cell geometry', 'statics', C.statik));
  panel.appendChild(checkbox('… see through it', 'translucent'));
  panel.appendChild(checkbox('robot arm', 'arm', C.arm));
  panel.appendChild(checkbox('gun', 'gun', C.gun));
  panel.appendChild(checkbox('edges', 'edges'));
  panel.appendChild(checkbox('nodes', 'nodes'));

  panel.appendChild(el('h2', { text: 'searches' }));
  if (SOLVES.length > 1) {
    const all = el('button', { text: 'show all' });
    const none = el('button', { text: 'hide all' });
    const set = v => () => {
      for (const s of sets) if (s.solve >= 0) s.on = v;
      buildPanel(); draw();
    };
    all.onclick = set(true);
    none.onclick = set(false);
    panel.appendChild(el('div', { class: 'bar' }, [all, none]));
  }
  /* Grouped by solve, with the pair's two trees under it.  Toggling the solve is the
   * filter asked for; the two sub-rows stay because which of the pair got stuck is the
   * question a pair of trees exists to answer. */
  for (let i = 0; i < SOLVES.length; i++) {
    const mine = sets.filter(s => s.solve === i);
    if (!mine.length) continue;
    const info = SOLVES[i];
    const group = el('div', { class: 'solve' });
    const head = el('input', { type: 'checkbox' });
    head.checked = mine.some(s => s.on);
    head.indeterminate = !mine.every(s => s.on) && head.checked;
    head.onchange = () => {
      for (const s of mine) s.on = head.checked;
      buildPanel(); draw();
    };
    group.appendChild(el('label', { class: 'row' }, [
      head,
      el('span', { text: `search ${info.id}` }),
      el('span', { class: 'n', text: `${info.nodes}` }),
    ]));
    /* The kind, then the caller's own name for the search.  Some labels already say
     * which tree ran -- "cartesian tree run 2" does -- and prefixing those with the kind
     * again reads as a stutter, so the label stands on its own where it covers it. */
    const kind = kindOf(info);
    const note = (kind && info.label && info.label.indexOf(kind) >= 0)
      ? info.label : [kind, info.label].filter(Boolean).join(' — ');
    if (note) group.appendChild(el('div', { class: 'lbl', text: note }));
    for (const s of mine) {
      const input = el('input', { type: 'checkbox' });
      input.checked = s.on;
      input.onchange = () => { s.on = input.checked; buildPanel(); draw(); };
      const sw = el('span', { class: 'sw' });
      sw.style.background = `rgb(${s.colour.map(v => Math.round(v * 255)).join(',')})`;
      group.appendChild(el('label', { class: 'row sub' },
        [input, sw, el('span', { text: s.key === 'start' ? 'from the start' : 'from the goal' }),
         el('span', { class: 'n', text: `${s.count}` })]));
    }
    panel.appendChild(group);
  }
  for (const s of sets) {
    if (s.solve >= 0) continue;
    const input = el('input', { type: 'checkbox' });
    input.checked = s.on;
    input.onchange = () => { s.on = input.checked; draw(); };
    const sw = el('span', { class: 'sw' });
    sw.style.background = `rgb(${s.colour.map(v => Math.round(v * 255)).join(',')})`;
    panel.appendChild(el('label', { class: 'row' },
      [input, sw, el('span', { text: s.name }), el('span', { class: 'n', text: `${s.count}` })]));
  }
  if (!sets.some(s => s.key !== 'chord'))
    panel.appendChild(el('p', { class: 'none', text:
      'No search tree was built for this segment — the band was off, the segment ' +
      'failed before a tree had a turn, or it failed at a locator rather than a ' +
      'transit. The straight chord is shown instead.' }));
  /* Said once here rather than per search: the two kinds draw the same way and mean
   * different things, and a reader comparing a bowed tree against a straight one needs to
   * know that is the edge and not the search doing better. */
  if (SOLVES.some(s => s.kind === 'joint') && SOLVES.some(s => s.kind === 'cartesian'))
    panel.appendChild(el('div', { class: 'lbl', text:
      'a joint tree draws its chords, which bow through space; a cartesian tree ' +
      'draws straight tool moves, which is what it was steering by' }));

  panel.appendChild(el('h2', { text: 'clearance' }));
  if (SOLVES.length > 1) panel.appendChild(checkbox('colour nodes by search', 'bySolve'));
  panel.appendChild(el('div', { id: 'ramp' }));
  panel.appendChild(el('div', { class: 'ends' }, [
    el('span', { text: `${DATA.clearance.margin} ${DATA.units} (margin)` }),
    el('span', { text: `${DATA.clearance.probe} (probe)` }),
  ]));
  // Grey is the one colour on the ramp that is not a reading, and in a capped file it is
  // most of the dots.  Worth a line: it is a node nobody measured, not a bad one.
  panel.appendChild(el('div', { class: 'lbl', text:
    'grey: no pose was baked for this node, so it is not clickable' }));

  panel.appendChild(el('h2', { text: 'node' }));
  panel.appendChild(readout);
  showPicked();

  panel.appendChild(el('h2', { text: 'measure' }));
  const mon = el('button', { text: measure.on ? 'measuring — m to stop' : 'measure (m)' });
  if (measure.on) mon.className = 'on';
  mon.onclick = () => toggleMeasure();
  const mclear = el('button', { text: 'clear' });
  mclear.onclick = () => { measure.pts = []; showMeasure(); draw(); };
  panel.appendChild(el('div', { class: 'bar' }, [mon, mclear]));
  panel.appendChild(mread);
  showMeasure();

  panel.appendChild(el('h2', { text: 'segment' }));
  const info = [['units', DATA.units]];
  if (DATA.gun_opening_mm !== null && DATA.gun_opening_mm !== undefined)
    info.push(['gun opening', `${DATA.gun_opening_mm} ${DATA.units}`]);
  if (DATA.searches)
    info.push(['tree searches', DATA.dropped
      ? `${SOLVES.length} kept of ${DATA.searches}` : `${DATA.searches}`]);
  /* Spelled out rather than left to be counted off the panel: whether both searches made
   * it into this file is the first thing a reader wants, and a segment runs the cartesian
   * tree and the joint tree on the same two endpoints. */
  const kinds = DATA.kinds || {};
  const names = Object.keys(kinds).sort();
  if (names.length)
    info.push(['of which', names.map(k => `${kinds[k]} ${KIND[k] || k}`).join(', ')]);
  if (DATA.dropped) info.push(['dropped', `${DATA.dropped} (--treeview-searches)`]);
  panel.appendChild(rows(info));
}

function showPicked() {
  readout.innerHTML = '';
  if (!picked) {
    readout.appendChild(el('p', { class: 'none', text: 'Click a node.' }));
    return;
  }
  const { set, index } = picked;
  const clr = set.clearance[index];
  const baked = set.pose[index];
  const mine = SOLVES[set.solve];
  const pairs = [
    ['from', mine ? `search ${mine.id}, ${set.name}` : set.name],
    ['index', `${index} of ${set.count - 1}`],
    ['clearance', clr === null || clr === undefined
      ? 'not measured' : `${clr} ${DATA.units}`],
  ];
  if (mine && (mine.label || mine.kind)) {
    const kind = kindOf(mine);
    pairs.push(['search', (kind && mine.label && mine.label.indexOf(kind) >= 0)
      ? mine.label : [kind, mine.label].filter(Boolean).join(', ')]);
  }
  if (set.parent) pairs.push(['parent', set.parent[index] < 0 ? 'root' : set.parent[index]]);
  const p = set.tcp[index];
  pairs.push(['tool', `${p[0]}, ${p[1]}, ${p[2]}`]);
  readout.appendChild(rows(pairs));
  if (!baked)
    readout.appendChild(el('p', { class: 'none', text:
      'No pose was baked for this node, so the robot is not drawn here. Raise ' +
      '--treeview-max-poses to bake more of them.' }));
  readout.appendChild(el('div', { id: 'q', text:
    '[' + set.q[index].map(v => v.toFixed(4)).join(', ') + ']' }));
}

buildPanel();

// ------------------------------------------------------------------------------ drawing
function drawMesh(mesh, model, colour, alpha) {
  gl.uniformMatrix4fv(mA.model, false, model);
  gl.uniform3fv(mA.color, colour);
  gl.uniform1f(mA.alpha, alpha);
  gl.bindBuffer(gl.ARRAY_BUFFER, mesh.pos);
  gl.vertexAttribPointer(mA.pos, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, mesh.nrm);
  gl.vertexAttribPointer(mA.nrm, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.TRIANGLES, 0, mesh.count);
}

function drawFlat(what, posBuf, colBuf, count, size, alpha) {
  gl.uniform1f(fA.size, size);
  gl.uniform1f(fA.alpha, alpha);
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.vertexAttribPointer(fA.pos, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
  gl.vertexAttribPointer(fA.col, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(what, 0, count);
}

let mvp = null, eye = null;

function draw() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.viewport(0, 0, w, h);
  gl.clearColor(0.082, 0.090, 0.102, 1);
  gl.enable(gl.DEPTH_TEST);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  eye = eyePos();
  const proj = perspective(Math.PI / 4, w / h, Math.max(span / 500, 0.1), span * 12);
  mvp = mul(proj, lookAt(eye, cam.at, [0, 0, 1]));
  const I = fromQuat([0, 0, 0, 1], [0, 0, 0]);

  // Opaque first, then anything see-through with the depth buffer read-only, which is the
  // cheapest order that does not let a translucent cell hide the tree behind it.
  gl.useProgram(meshProg);
  gl.enableVertexAttribArray(mA.pos);
  gl.enableVertexAttribArray(mA.nrm);
  gl.uniformMatrix4fv(mA.mvp, false, mvp);
  gl.uniform3fv(mA.eye, eye);
  gl.disable(gl.BLEND);
  gl.depthMask(true);

  if (opts.statics && !opts.translucent)
    for (const o of statics) drawMesh(o.mesh, I, C.statik, 1);

  if (picked && picked.set.pose[picked.index]) {
    const pose = picked.set.pose[picked.index];
    for (const name in pose) {
      const mesh = moving[name];
      if (!mesh) continue;
      const isGun = gunSet.has(name);
      if (isGun ? !opts.gun : !opts.arm) continue;
      drawMesh(mesh, fromQuat(pose[name].q, pose[name].p), isGun ? C.gun : C.arm, 1);
    }
  }

  gl.useProgram(flatProg);
  gl.enableVertexAttribArray(fA.pos);
  gl.enableVertexAttribArray(fA.col);
  gl.uniformMatrix4fv(fA.mvp, false, mvp);
  for (const s of sets) {
    if (!s.on) continue;
    if (opts.edges && s.lineCount) drawFlat(gl.LINES, s.lineBuf, s.lineCol, s.lineCount, 1, 0.8);
    if (opts.nodes) drawFlat(gl.POINTS, s.pointBuf,
      opts.bySolve ? s.pointColSet : s.pointCol, s.count, 5 * dpr, 1);
  }
  if (picked) {
    const p = picked.set.tcp[picked.index];
    if (!draw._pick) draw._pick = { pos: buffer(new Float32Array(3)), col: buffer(new Float32Array(C.pick)) };
    gl.bindBuffer(gl.ARRAY_BUFFER, draw._pick.pos);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(p), gl.DYNAMIC_DRAW);
    drawFlat(gl.POINTS, draw._pick.pos, draw._pick.col, 1, 13 * dpr, 1);
  }
  drawMeasure(dpr);

  if (opts.statics && opts.translucent) {
    gl.useProgram(meshProg);
    gl.enableVertexAttribArray(mA.pos);
    gl.enableVertexAttribArray(mA.nrm);
    gl.uniformMatrix4fv(mA.mvp, false, mvp);
    gl.uniform3fv(mA.eye, eye);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    for (const o of statics) drawMesh(o.mesh, I, C.statik, 0.26);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
  }

  // Last, and outside the GL work: it is a DOM element positioned off this frame's mvp.
  placeLabel();
}

// ------------------------------------------------------------------------- interaction
/* Picking in screen space rather than through a colour buffer: the sets are at most a few
 * thousand points, so projecting them all is far cheaper than a second render pass, and it
 * cannot disagree with what is on screen about which node is in front.
 *
 * A node with no baked pose is passed over rather than picked.  The whole of what picking
 * is for is putting the robot at a node, and --treeview-max-poses means most nodes in a
 * big file have no pose to put it at: selecting one used to succeed and leave the arm
 * drawn nowhere, which reads as the robot vanishing rather than as "this node was not one
 * of the ones baked".  Skipped here rather than filtered when the sets are built, because
 * every node still has to project for its own sake -- it is drawn, and it is part of the
 * shape the diagnostic exists to show. */
function posed(s, i) {
  return !s.pose || !!s.pose[i];
}

function pick(px, py, anyNode) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.width, h = canvas.height;
  let best = null, bestD = 18 * 18, bestZ = Infinity;
  for (const s of sets) {
    if (!s.on) continue;
    for (let i = 0; i < s.count; i++) {
      /* `anyNode` is the measuring tool's, and it is the one caller that wants a node
       * with no baked pose.  Putting the robot somewhere needs a pose; measuring to a
       * tool position does not -- that position is in the file for every node and is
       * drawn for every node, so refusing to measure to one would be refusing to measure
       * to a point the user is looking at. */
      if (!anyNode && !posed(s, i)) continue;
      const p = s.tcp[i];
      const cx = mvp[0] * p[0] + mvp[4] * p[1] + mvp[8] * p[2] + mvp[12];
      const cy = mvp[1] * p[0] + mvp[5] * p[1] + mvp[9] * p[2] + mvp[13];
      const cz = mvp[2] * p[0] + mvp[6] * p[1] + mvp[10] * p[2] + mvp[14];
      const cw = mvp[3] * p[0] + mvp[7] * p[1] + mvp[11] * p[2] + mvp[15];
      if (cw <= 0) continue;
      const sx = (cx / cw * 0.5 + 0.5) * w / dpr;
      const sy = (1 - (cy / cw * 0.5 + 0.5)) * h / dpr;
      const d = (sx - px) * (sx - px) + (sy - py) * (sy - py);
      const z = cz / cw;
      if (d <= bestD && (d < bestD * 0.6 || z < bestZ)) {
        best = { set: s, index: i }; bestD = Math.max(d, 16); bestZ = z;
      }
    }
  }
  return best;
}

/* ------------------------------------------------------------------------- measuring
 * Two points and the distance between them.  The reason it is worth having in a viewer
 * that already prints a clearance per node is that the clearance answers one question --
 * how close is the robot to anything at all -- and a failure asks others it cannot: how
 * far did this tree get along the retreat axis, how deep is the pocket, how much room is
 * there between the gun and the fixture *here*.  Those are distances between two places
 * the reader can see and point at.
 *
 * What a point may be: a node's tool position, snapped to when the click lands near one,
 * or the first surface the click ray meets.  Snapping first because a node is a place the
 * program reasoned about, so a measurement from one is a measurement of something the log
 * also talks about.
 *
 * **The surfaces are the convex hulls, not the source CAD**, which is what the whole file
 * draws and is deliberate -- see ``treeview``'s own note. So a distance to a face is a
 * distance to what the planner collided against, which is the useful answer here and not
 * the same as what a CAD measurement would give. The readout says so.
 *
 * `measure` itself is declared with `picked`, up where the UI is built: the panel holds
 * the toggle, and `buildPanel` runs before this point in the file.
 */

/* A ray through a pixel, built from the camera basis rather than by inverting the
 * view-projection: the basis is already how `lookAt` is defined, so there is one
 * convention in play instead of two that have to agree. */
function rayAt(px, py) {
  const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
  const z = norm(sub(eye, cam.at));           // `lookAt`'s own z: from the target to the eye
  const x = norm(cross([0, 0, 1], z));        // ...so x is screen right and y screen up
  const y = cross(z, x);
  const tan = Math.tan(Math.PI / 8);          // the perspective's fovy is PI/4
  const ndcx = (px / w) * 2 - 1, ndcy = 1 - (py / h) * 2;
  const sx = ndcx * tan * (w / h), sy = ndcy * tan;
  return { o: eye.slice(),
           d: norm([sx * x[0] + sy * y[0] - z[0],
                    sx * x[1] + sy * y[1] - z[1],
                    sx * x[2] + sy * y[2] - z[2]]) };
}

/* A rigid model matrix applied to a ray backwards, so a link's triangles can be tested
 * where they are stored.  Cheaper and steadier than transforming the triangles: the
 * rotation is orthonormal, so `t` means the same thing in both frames and the hit point
 * can be taken off the original ray. */
function intoFrame(M, o, d) {
  const w = sub(o, [M[12], M[13], M[14]]);
  const c0 = [M[0], M[1], M[2]], c1 = [M[4], M[5], M[6]], c2 = [M[8], M[9], M[10]];
  return { o: [dot(c0, w), dot(c1, w), dot(c2, w)],
           d: [dot(c0, d), dot(c1, d), dot(c2, d)] };
}

/* Moller-Trumbore over a flat-shaded soup, nearest hit under `tMax`.  Brute force and no
 * acceleration structure: this runs on a click and not per frame, and a cell of a million
 * triangles costs tens of milliseconds, where a BVH would cost a build on every load --
 * including the loads where nobody measures anything.
 *
 * Two-sided on purpose.  With "see through it" on the reader is often inside a hull, and
 * a measurement to the face they are looking at from behind is still the face they mean. */
function rayTris(raw, o, d, tMax) {
  let best = tMax;
  for (let i = 0; i < raw.length; i += 9) {
    const ax = raw[i], ay = raw[i + 1], az = raw[i + 2];
    const e1x = raw[i + 3] - ax, e1y = raw[i + 4] - ay, e1z = raw[i + 5] - az;
    const e2x = raw[i + 6] - ax, e2y = raw[i + 7] - ay, e2z = raw[i + 8] - az;
    const hx = d[1] * e2z - d[2] * e2y;
    const hy = d[2] * e2x - d[0] * e2z;
    const hz = d[0] * e2y - d[1] * e2x;
    const det = e1x * hx + e1y * hy + e1z * hz;
    if (det > -1e-12 && det < 1e-12) continue;
    const inv = 1 / det;
    const sx = o[0] - ax, sy = o[1] - ay, sz = o[2] - az;
    const u = (sx * hx + sy * hy + sz * hz) * inv;
    if (u < 0 || u > 1) continue;
    const qx = sy * e1z - sz * e1y;
    const qy = sz * e1x - sx * e1z;
    const qz = sx * e1y - sy * e1x;
    const v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv;
    if (v < 0 || u + v > 1) continue;
    const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
    if (t > 1e-6 && t < best) best = t;
  }
  return best;
}

/* The nearest surface under a pixel, over exactly what is drawn: a hidden static is not
 * measured to, and the arm and gun are only there at all when a node is picked, that
 * being the only pose they are drawn at. */
function surfaceHit(px, py) {
  const ray = rayAt(px, py);
  let best = Infinity, what = null;
  if (opts.statics)
    for (const o of statics) {
      const t = rayTris(o.mesh.raw, ray.o, ray.d, best);
      if (t < best) { best = t; what = o.name; }
    }
  const pose = picked && picked.set.pose[picked.index];
  if (pose)
    for (const name in pose) {
      const mesh = moving[name];
      if (!mesh) continue;
      const isGun = gunSet.has(name);
      if (isGun ? !opts.gun : !opts.arm) continue;
      const L = intoFrame(fromQuat(pose[name].q, pose[name].p), ray.o, ray.d);
      const t = rayTris(mesh.raw, L.o, L.d, best);
      if (t < best) { best = t; what = name; }
    }
  if (!isFinite(best)) return null;
  return { p: [ray.o[0] + ray.d[0] * best, ray.o[1] + ray.d[1] * best,
               ray.o[2] + ray.d[2] * best],
           what: what, surface: true };
}

function measurePoint(px, py) {
  const hit = pick(px, py, true);
  if (hit) {
    const mine = SOLVES[hit.set.solve];
    return { p: hit.set.tcp[hit.index].slice(),
             what: (mine ? `search ${mine.id}, ` : '') +
                   `${hit.set.name} node ${hit.index}`,
             clearance: hit.set.clearance[hit.index] };
  }
  return surfaceHit(px, py);
}

function measureClick(px, py) {
  const got = measurePoint(px, py);
  // A click that hits nothing leaves the reading alone.  Clicking past the geometry into
  // the background is easy to do while orbiting round a measurement to read it from
  // another angle, and throwing the measurement away for it would be the one way to lose
  // a reading by accident.  `clear` and Esc are how to lose one on purpose.
  if (!got) return;
  // A third click starts again from it rather than being ignored or replacing the second.
  // Measuring is usually a run of readings from one place, and starting again from the
  // place just clicked is the one behaviour that needs no extra control to drive.
  measure.pts = measure.pts.length === 1 ? [measure.pts[0], got] : [got];
  showMeasure();
}

function measureLength() {
  if (measure.pts.length < 2) return null;
  return Math.hypot(...sub(measure.pts[1].p, measure.pts[0].p));
}

function toggleMeasure(on) {
  measure.on = on === undefined ? !measure.on : !!on;
  // The points are kept across a toggle.  Leaving the mode to orbit round and read a
  // measurement from another angle is the common thing to want, and losing it on the way
  // out would make the mode something to stay in rather than something to use.
  buildPanel();
  showHint();
  draw();
}

/* The label is HTML over the canvas rather than glyphs in the scene.  The panel is
 * already a DOM, so this needs no font atlas and no third shader, and it stays legible
 * against whatever is behind it. */
function placeLabel() {
  const d = measure.on ? measureLength() : null;
  if (d === null || !mvp) { mlabel.style.display = 'none'; return; }
  const a = measure.pts[0].p, b = measure.pts[1].p;
  const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
  const cx = mvp[0] * m[0] + mvp[4] * m[1] + mvp[8] * m[2] + mvp[12];
  const cy = mvp[1] * m[0] + mvp[5] * m[1] + mvp[9] * m[2] + mvp[13];
  const cw = mvp[3] * m[0] + mvp[7] * m[1] + mvp[11] * m[2] + mvp[15];
  if (cw <= 0) { mlabel.style.display = 'none'; return; }   // behind the eye
  const r = canvas.getBoundingClientRect();
  mlabel.style.left = `${r.left + (cx / cw * 0.5 + 0.5) * r.width}px`;
  mlabel.style.top = `${r.top + (1 - (cy / cw * 0.5 + 0.5)) * r.height}px`;
  mlabel.textContent = `${d.toFixed(1)} ${DATA.units}`;
  mlabel.style.display = 'block';
}

/* Drawn with the flat program, which the caller already has bound.  The line goes down
 * twice: once against the depth buffer, and once through it at low alpha, so a rule that
 * passes inside a panel can still be followed to its far end -- which is most of them,
 * the interesting distances here running through the fixture rather than beside it. */
function drawMeasure(dpr) {
  const n = measure.pts.length;
  if (!measure.on || !n) return;
  if (!drawMeasure._b)
    drawMeasure._b = { pos: buffer(new Float32Array(6)),
                       col: buffer(new Float32Array([C.rule, C.rule].flat())) };
  const pos = new Float32Array(6);
  for (let i = 0; i < n; i++)
    for (let k = 0; k < 3; k++) pos[i * 3 + k] = measure.pts[i].p[k];
  gl.bindBuffer(gl.ARRAY_BUFFER, drawMeasure._b.pos);
  gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
  if (n === 2) {
    drawFlat(gl.LINES, drawMeasure._b.pos, drawMeasure._b.col, 2, 1, 0.95);
    gl.disable(gl.DEPTH_TEST);
    drawFlat(gl.LINES, drawMeasure._b.pos, drawMeasure._b.col, 2, 1, 0.3);
    gl.enable(gl.DEPTH_TEST);
  }
  // The ends over the top of the line, and through the geometry as well: a point taken on
  // a face the reader has since orbited behind is still where the measurement is from.
  gl.disable(gl.DEPTH_TEST);
  drawFlat(gl.POINTS, drawMeasure._b.pos, drawMeasure._b.col, n, 11 * dpr, 1);
  gl.enable(gl.DEPTH_TEST);
}

let drag = null;
canvas.addEventListener('pointerdown', e => {
  canvas.setPointerCapture(e.pointerId);
  drag = { x: e.clientX, y: e.clientY, moved: 0,
           pan: e.button === 2 || e.shiftKey };
  canvas.classList.add('dragging');
});
canvas.addEventListener('pointermove', e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  drag.moved += Math.abs(dx) + Math.abs(dy);
  if (drag.pan) {
    // Pan along the camera's own right and up, so dragging moves what is under the cursor
    // rather than something that depends on where the model happens to be.
    const f = norm(sub(cam.at, eye)), r = norm(cross([0, 0, 1], f)), u = cross(f, r);
    const k = cam.dist * 0.0018;
    for (let i = 0; i < 3; i++) cam.at[i] += (-r[i] * dx + u[i] * dy) * k;
  } else {
    cam.yaw += dx * 0.008;
    cam.pitch = Math.max(-1.52, Math.min(1.52, cam.pitch + dy * 0.008));
  }
  draw();
});
canvas.addEventListener('pointerup', e => {
  canvas.classList.remove('dragging');
  const was = drag;
  drag = null;
  if (!was || was.moved > 6) return;
  const r = canvas.getBoundingClientRect();
  if (measure.on) {
    // A measuring click does not move the robot.  The arm and gun are drawn at the picked
    // node, so re-picking here would move the very surfaces the second point is about to
    // be taken against, between the two clicks of one measurement.
    measureClick(e.clientX - r.left, e.clientY - r.top);
    draw();
    return;
  }
  const hit = pick(e.clientX - r.left, e.clientY - r.top);
  if (hit) { picked = hit; showPicked(); draw(); }
});
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  cam.dist *= Math.exp((e.deltaY > 0 ? 1 : -1) * 0.12);
  cam.dist = Math.max(span * 0.03, Math.min(span * 10, cam.dist));
  draw();
}, { passive: false });
window.addEventListener('keydown', e => {
  if (e.key === 'r' || e.key === 'R') {
    cam.yaw = -0.9; cam.pitch = 0.5; cam.dist = span * 1.5; cam.at = centre.slice();
    draw();
  } else if (e.key === 'm' || e.key === 'M') {
    toggleMeasure();
  } else if (e.key === 'Escape') {
    measure.pts = [];
    showMeasure();
    draw();
  }
});
window.addEventListener('resize', draw);

draw();

})();
