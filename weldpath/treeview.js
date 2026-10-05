/* weldpath failure viewer.
 *
 * Reads window.WELDPATH_TREE, written beside it by weldpath.treeview, and draws the
 * Cartesian search that failed: the cell as the planner collided against it, the two trees
 * that grew towards each other, and the gun and arm wherever a node is clicked.
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
.sw { width: 11px; height: 11px; border-radius: 2px; flex: 0 0 auto; }
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

if (!DATA || DATA.format !== 'weldpath-treeview/1') {
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
};

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
const sets = [];
for (const t of (DATA.trees || [])) {
  sets.push({
    name: t.name === 'start' ? 'tree from the start' : 'tree from the goal',
    key: t.name, colour: t.name === 'start' ? C.start : C.goal,
    count: t.count, tcp: t.tcp, clearance: t.clearance, pose: t.pose,
    q: t.q, parent: t.parent, edges: t.edges, on: true,
  });
}
if (DATA.chord) {
  const n = DATA.chord.tcp.length;
  const edges = [];
  for (let i = 1; i < n; i++)
    edges.push({ a: i - 1, b: i, path: [DATA.chord.tcp[i - 1], DATA.chord.tcp[i]] });
  sets.push({
    name: 'the straight chord', key: 'chord', colour: C.chord, count: n,
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
               edges: true, nodes: true, chordPose: true };
let picked = null;          // {set, index}

const panel = el('div', { id: 'panel' });
document.body.appendChild(panel);
document.body.appendChild(el('div', { id: 'hint', text:
  'drag to orbit · wheel to zoom · right-drag or shift-drag to pan · ' +
  'click a node to put the robot there · R resets the view' }));

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
  for (const s of sets) {
    const input = el('input', { type: 'checkbox' });
    input.checked = s.on;
    input.onchange = () => { s.on = input.checked; draw(); };
    const sw = el('span', { class: 'sw' });
    sw.style.background = `rgb(${s.colour.map(v => Math.round(v * 255)).join(',')})`;
    panel.appendChild(el('label', { class: 'row' },
      [input, sw, el('span', { text: `${s.name} (${s.count})` })]));
  }
  if (!sets.some(s => s.key !== 'chord'))
    panel.appendChild(el('p', { class: 'none', text:
      'No Cartesian tree was built for this segment — the band was off, the segment ' +
      'failed before the tree had a turn, or it failed at a locator rather than a ' +
      'transit. The straight chord is shown instead.' }));

  panel.appendChild(el('h2', { text: 'clearance' }));
  panel.appendChild(el('div', { id: 'ramp' }));
  panel.appendChild(el('div', { class: 'ends' }, [
    el('span', { text: `${DATA.clearance.margin} ${DATA.units} (margin)` }),
    el('span', { text: `${DATA.clearance.probe} (probe)` }),
  ]));

  panel.appendChild(el('h2', { text: 'node' }));
  panel.appendChild(readout);
  showPicked();

  panel.appendChild(el('h2', { text: 'segment' }));
  const info = [['units', DATA.units]];
  if (DATA.gun_opening_mm !== null && DATA.gun_opening_mm !== undefined)
    info.push(['gun opening', `${DATA.gun_opening_mm} ${DATA.units}`]);
  if (DATA.searches) info.push(['tree searches', DATA.searches]);
  if (DATA.tree_label) info.push(['biggest', DATA.tree_label]);
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
  const pairs = [
    ['from', set.name],
    ['index', `${index} of ${set.count - 1}`],
    ['clearance', clr === null || clr === undefined
      ? 'not measured' : `${clr} ${DATA.units}`],
  ];
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
    if (opts.nodes) drawFlat(gl.POINTS, s.pointBuf, s.pointCol, s.count, 5 * dpr, 1);
  }
  if (picked) {
    const p = picked.set.tcp[picked.index];
    if (!draw._pick) draw._pick = { pos: buffer(new Float32Array(3)), col: buffer(new Float32Array(C.pick)) };
    gl.bindBuffer(gl.ARRAY_BUFFER, draw._pick.pos);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(p), gl.DYNAMIC_DRAW);
    drawFlat(gl.POINTS, draw._pick.pos, draw._pick.col, 1, 13 * dpr, 1);
  }

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
}

// ------------------------------------------------------------------------- interaction
/* Picking in screen space rather than through a colour buffer: the sets are at most a few
 * thousand points, so projecting them all is far cheaper than a second render pass, and it
 * cannot disagree with what is on screen about which node is in front. */
function pick(px, py) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.width, h = canvas.height;
  let best = null, bestD = 18 * 18, bestZ = Infinity;
  for (const s of sets) {
    if (!s.on) continue;
    for (let i = 0; i < s.count; i++) {
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
  }
});
window.addEventListener('resize', draw);

draw();

})();
