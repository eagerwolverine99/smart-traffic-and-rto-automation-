// ════════════════════════════════════════════════════
//  DATA STORE
// ════════════════════════════════════════════════════
const TYPES  = ['bike','car','bus','truck'];
const ZONES  = ['MG Road','NH-48','Ring Road','City Center','Highway-24','Airport Rd'];
const COLORS = { bike:'#a371f7', car:'#3fb950', bus:'#d29922', truck:'#f0883e' };

let counts   = { bike:0, car:0, bus:0, truck:0 };
let detections = [];
let framesProcessed = 0;
let currentFilter = 'all';
let flowHistory = { bike:[], car:[], bus:[], truck:[] };
let timeLabels  = [];
let detIdSeq    = 1;

// ════════════════════════════════════════════════════
//  CLOCK
// ════════════════════════════════════════════════════
function updateClock() {
  document.getElementById('clock').textContent = new Date().toLocaleString('en-IN',{
    hour:'2-digit', minute:'2-digit', second:'2-digit',
    day:'2-digit', month:'short', year:'numeric'
  });
}
setInterval(updateClock, 1000);
updateClock();

// ════════════════════════════════════════════════════
//  FLOW CHART (line, rolling 60 pts)
// ════════════════════════════════════════════════════
const flowCtx = document.getElementById('flowChart').getContext('2d');
const flowChart = new Chart(flowCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label:'Bikes',  data:[], borderColor:'#a371f7', backgroundColor:'rgba(163,113,247,.08)', tension:.4, pointRadius:0, fill:true, borderWidth:1.5 },
      { label:'Cars',   data:[], borderColor:'#3fb950', backgroundColor:'rgba(63,185,80,.08)',   tension:.4, pointRadius:0, fill:true, borderWidth:1.5 },
      { label:'Buses',  data:[], borderColor:'#d29922', backgroundColor:'rgba(210,153,34,.08)',  tension:.4, pointRadius:0, fill:true, borderWidth:1.5 },
      { label:'Trucks', data:[], borderColor:'#f0883e', backgroundColor:'rgba(240,136,62,.08)',  tension:.4, pointRadius:0, fill:true, borderWidth:1.5 },
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false, animation:{ duration:0 },
    plugins: { legend:{ display:false } },
    scales: {
      x: { grid:{ color:'rgba(48,54,61,.5)', drawBorder:false }, ticks:{ color:'#8b949e', font:{ size:10 }, maxTicksLimit:10 } },
      y: { grid:{ color:'rgba(48,54,61,.5)', drawBorder:false }, ticks:{ color:'#8b949e', font:{ size:10 } }, beginAtZero:true }
    }
  }
});

// ════════════════════════════════════════════════════
//  DONUT CHART
// ════════════════════════════════════════════════════
const donutCtx = document.getElementById('donutChart').getContext('2d');
const donutChart = new Chart(donutCtx, {
  type: 'doughnut',
  data: {
    labels: ['Bikes','Cars','Buses','Trucks'],
    datasets:[{
      data:[0,0,0,0],
      backgroundColor:['rgba(163,113,247,.8)','rgba(63,185,80,.8)','rgba(210,153,34,.8)','rgba(240,136,62,.8)'],
      borderColor:['#a371f7','#3fb950','#d29922','#f0883e'],
      borderWidth:1.5, hoverOffset:8
    }]
  },
  options: {
    responsive:true, maintainAspectRatio:false, cutout:'68%',
    plugins: {
      legend:{ position:'bottom', labels:{ color:'#8b949e', font:{size:11}, padding:12, boxWidth:12 } }
    }
  }
});

// ════════════════════════════════════════════════════
//  HOURLY BAR CHART
// ════════════════════════════════════════════════════
const hourlyCtx = document.getElementById('hourlyChart').getContext('2d');
const hourlyData = Array.from({length:24}, (_,i) => {
  const peak = (i>=8&&i<=10)||(i>=17&&i<=19) ? 1.6 : (i>=6&&i<=21 ? 1 : .2);
  return Math.floor((Math.random()*80+20)*peak);
});
const hourlyChart = new Chart(hourlyCtx, {
  type:'bar',
  data:{
    labels: Array.from({length:24},(_,i)=>`${i}:00`),
    datasets:[{
      label:'Vehicles',
      data: hourlyData,
      backgroundColor: hourlyData.map(v =>
        v>140 ? 'rgba(248,81,73,.7)' : v>100 ? 'rgba(210,153,34,.7)' : 'rgba(0,170,255,.6)'
      ),
      borderRadius:4, borderSkipped:false
    }]
  },
  options:{
    responsive:true, maintainAspectRatio:false,
    plugins:{ legend:{ display:false } },
    scales:{
      x:{ grid:{ display:false }, ticks:{ color:'#8b949e', font:{size:9}, maxTicksLimit:12 } },
      y:{ grid:{ color:'rgba(48,54,61,.5)' }, ticks:{ color:'#8b949e', font:{size:10} }, beginAtZero:true }
    }
  }
});

// ════════════════════════════════════════════════════
//  WEEKLY HEATMAP
// ════════════════════════════════════════════════════
function buildHeatmap() {
  const days    = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  const hours   = ['6AM','8AM','10AM','12PM','2PM','4PM','6PM','8PM'];
  const hourIdx = [6,8,10,12,14,16,18,20];

  const container = document.getElementById('heatmapContainer');
  container.innerHTML = '';

  const dayRow = document.createElement('div');
  dayRow.className = 'heatmap-days';
  days.forEach(d => { const s=document.createElement('span'); s.textContent=d; dayRow.appendChild(s); });
  container.appendChild(dayRow);

  const hm = document.createElement('div');
  hm.className = 'heatmap';

  hours.forEach((h,hi) => {
    const row = document.createElement('div');
    row.className = 'heatmap-row';
    const lbl = document.createElement('div');
    lbl.className = 'heatmap-label';
    lbl.textContent = h;
    row.appendChild(lbl);

    const cells = document.createElement('div');
    cells.className = 'heatmap-cells';

    days.forEach((_,di) => {
      const isPeak = (hi>=1&&hi<=2)||(hi>=5&&hi<=6);
      const isWknd = di>=5;
      let v = Math.random()*(isPeak?(isWknd?.5:.9):(isWknd?.3:.6))+.05;
      v = Math.min(1,v);
      const cell = document.createElement('div');
      cell.className = 'heatmap-cell';
      cell.style.background = `rgba(0,170,255,${v.toFixed(2)})`;
      cell.title = `${days[di]} ${h}: ${Math.round(v*200)} vehicles`;
      cells.appendChild(cell);
    });

    row.appendChild(cells);
    hm.appendChild(row);
  });

  container.appendChild(hm);
}
buildHeatmap();

// ════════════════════════════════════════════════════
//  CONGESTION ZONES
// ════════════════════════════════════════════════════
let zonePcts = ZONES.map(() => Math.floor(Math.random()*80+10));

function renderZones() {
  const list = document.getElementById('zoneList');
  list.innerHTML = '';
  zonePcts.forEach((pct, i) => {
    const status = pct>=70 ? 'high' : pct>=40 ? 'medium' : 'low';
    const color  = pct>=70 ? '#f85149' : pct>=40 ? '#d29922' : '#3fb950';
    const label  = pct>=70 ? 'HIGH' : pct>=40 ? 'MED' : 'LOW';
    list.innerHTML += `
      <div class="zone-item">
        <div class="zone-name">${ZONES[i]}</div>
        <div class="zone-bar-bg">
          <div class="zone-bar-fill" style="width:${pct}%;background:${color}"></div>
        </div>
        <div class="zone-pct">${pct}%</div>
        <div class="zone-status status-${status}">${label}</div>
      </div>`;
  });
}
renderZones();

// ════════════════════════════════════════════════════
//  CAMERA FEED SIMULATION (canvas)
// ════════════════════════════════════════════════════
const camVehicles = [[], [], [], []];
const camColors   = ['#a371f7','#3fb950','#d29922','#f0883e'];

function initCam(idx) {
  const n = Math.floor(Math.random()*4)+1;
  for (let i=0;i<n;i++) {
    camVehicles[idx].push({
      x: Math.random()*140+10, y: Math.random()*70+15,
      vx:(Math.random()-.5)*1.2, vy:(Math.random()-.5)*.5,
      type: TYPES[Math.floor(Math.random()*4)],
      w: Math.random()*16+12, h: Math.random()*10+8
    });
  }
}
[0,1,2,3].forEach(initCam);

function drawCam(idx) {
  const canvas = document.getElementById(`cam${idx+1}`);
  const ctx    = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  // BG
  ctx.fillStyle = '#0a0e14';
  ctx.fillRect(0,0,W,H);

  // Road lines
  ctx.strokeStyle = 'rgba(255,255,255,.08)';
  ctx.setLineDash([12,8]);
  ctx.beginPath();
  ctx.moveTo(0,H/3); ctx.lineTo(W,H/3);
  ctx.moveTo(0,H*2/3); ctx.lineTo(W,H*2/3);
  ctx.stroke();
  ctx.setLineDash([]);

  // scan line
  const scanY = (Date.now()/20)%(H+10)-5;
  const grad = ctx.createLinearGradient(0,scanY-10,0,scanY+2);
  grad.addColorStop(0,'rgba(0,170,255,0)');
  grad.addColorStop(1,'rgba(0,170,255,.18)');
  ctx.fillStyle = grad;
  ctx.fillRect(0,scanY-10,W,12);

  // Vehicles
  camVehicles[idx].forEach(v => {
    v.x += v.vx; v.y += v.vy;
    if (v.x<0||v.x>W-v.w) v.vx*=-1;
    if (v.y<0||v.y>H-v.h) v.vy*=-1;

    const c = COLORS[v.type];
    ctx.fillStyle = c+'33';
    ctx.strokeStyle = c;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(v.x, v.y, v.w, v.h, 2);
    ctx.fill(); ctx.stroke();

    // bbox corners
    ctx.strokeStyle = c;
    ctx.lineWidth = 1;
    const cs = 3;
    [[v.x,v.y],[v.x+v.w,v.y],[v.x,v.y+v.h],[v.x+v.w,v.y+v.h]].forEach(([px,py]) => {
      const dx = px===v.x?1:-1, dy = py===v.y?1:-1;
      ctx.beginPath();
      ctx.moveTo(px,py+dy*cs); ctx.lineTo(px,py); ctx.lineTo(px+dx*cs,py);
      ctx.stroke();
    });
  });

  // Frame counter overlay
  ctx.fillStyle = 'rgba(0,170,255,.7)';
  ctx.font = '8px monospace';
  ctx.fillText(`${camVehicles[idx].length} detected`, 6, H-6);

  document.getElementById(`cam${idx+1}-count`).textContent = `${camVehicles[idx].length} vehicles`;
}

function animCams() {
  [0,1,2,3].forEach(drawCam);
  requestAnimationFrame(animCams);
}
animCams();

// ════════════════════════════════════════════════════
//  DETECTION TABLE
// ════════════════════════════════════════════════════
function renderTable() {
  const filter = currentFilter;
  const search = document.getElementById('searchBox').value.toLowerCase();
  const filtered = detections.filter(d =>
    (filter==='all' || d.type===filter) &&
    (d.zone.toLowerCase().includes(search) || d.type.includes(search))
  ).slice(-60).reverse();

  document.getElementById('det-count-label').textContent = `${filtered.length} records`;
  const tbody = document.getElementById('detTableBody');
  tbody.innerHTML = filtered.map(d => `
    <tr>
      <td style="color:var(--muted);font-family:monospace">#${String(d.id).padStart(4,'0')}</td>
      <td><span class="type-badge type-${d.type}">${d.type.toUpperCase()}</span></td>
      <td style="color:var(--muted);font-size:11px">${d.zone}</td>
      <td>
        <div class="conf-bar">
          <div class="conf-bg"><div class="conf-fill" style="width:${d.conf}%;background:${COLORS[d.type]}"></div></div>
          <span style="font-size:10px;color:var(--muted);width:30px">${d.conf}%</span>
        </div>
      </td>
      <td style="color:var(--muted);font-size:10px;white-space:nowrap">${d.time}</td>
    </tr>`).join('');
}

function setFilter(f, btn) {
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderTable();
}

// ════════════════════════════════════════════════════
//  MAIN SIMULATION TICK
// ════════════════════════════════════════════════════
let tick = 0;
function simTick() {
  tick++;
  framesProcessed += 24;

  // Generate detections this tick
  const n = Math.floor(Math.random()*5)+1;
  for (let i=0;i<n;i++) {
    const type = TYPES[Math.floor(Math.random()*4)];
    const zone = ZONES[Math.floor(Math.random()*ZONES.length)];
    const conf = Math.floor(Math.random()*15+82);
    const now  = new Date().toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
    counts[type]++;
    detections.push({ id:detIdSeq++, type, zone, conf, time:now });
    if (detections.length>500) detections.shift();
  }

  // Update stat cards
  const total = counts.bike+counts.car+counts.bus+counts.truck;
  document.getElementById('s-total').textContent    = total.toLocaleString();
  document.getElementById('s-total-sub').textContent = `vehicles today`;
  TYPES.forEach(t => {
    document.getElementById(`s-${t}`).textContent = counts[t].toLocaleString();
    const pct = total ? ((counts[t]/total)*100).toFixed(1) : 0;
    document.getElementById(`s-${t}-pct`).textContent = `${pct}% of total`;
  });

  // Update donut
  donutChart.data.datasets[0].data = [counts.bike, counts.car, counts.bus, counts.truck];
  donutChart.update('none');

  // Update flow chart (per-second snapshot of new detections)
  const nowLabel = new Date().toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  timeLabels.push(nowLabel);
  const snap = { bike:0, car:0, bus:0, truck:0 };
  detections.slice(-n).forEach(d => snap[d.type]++);
  TYPES.forEach((t,i) => {
    flowChart.data.datasets[i].data.push(snap[t]);
    if (flowChart.data.datasets[i].data.length>60) flowChart.data.datasets[i].data.shift();
  });
  if (timeLabels.length>60) timeLabels.shift();
  flowChart.data.labels = [...timeLabels];
  flowChart.update('none');

  // Frames
  document.getElementById('frames-val').textContent = framesProcessed.toLocaleString();
  document.getElementById('fps-val').textContent    = (22+Math.floor(Math.random()*5))+' fps';

  // Drift zone congestion
  zonePcts = zonePcts.map(p => Math.min(99,Math.max(5, p+(Math.random()*6-3)|0)));
  renderZones();

  renderTable();
}

// Warm-up: run 20 ticks instantly
for (let i=0;i<20;i++) simTick();
setInterval(simTick, 1200);
