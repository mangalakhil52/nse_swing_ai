const state={agents:{},selectedAgent:'TECHNICAL',scan:{universe:2557,filtered:347,candidates:83,intel:21,final:2,processed:0,status:'IDLE'},connected:false};
const $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>[...r.querySelectorAll(s)];
const API_BASE=window.NSE_AI_API_BASE||'/api';

function setText(selector,value){const el=$(selector);if(el)el.textContent=value}
function setConnection(connected){state.connected=connected;document.body.classList.toggle('live-connected',connected);const dot=$('.market-status .dot');if(dot)dot.className=`dot ${connected?'green':'amber'}`;const status=$('.market-status strong');if(status)status.textContent=connected?'LIVE':'OFFLINE'}
function renderScan(){
  const vals=[state.scan.universe,state.scan.filtered,state.scan.candidates,state.scan.intel,state.scan.final];
  $$('.pipeline b').forEach((n,i)=>n.textContent=Number(vals[i]||0).toLocaleString());
  const core=$('.core strong'); if(core)core.textContent=Number(state.scan.processed||state.scan.universe).toLocaleString();
  const small=$('.core small'); if(small)small.textContent=state.scan.status;
}
function agentPanel(agent){
  const panel=$('.agent-inspector'); if(!panel)return;
  const data=state.agents[agent]||{status:'WAITING',progress:0,processed:0,decision:'No live event received'};
  panel.innerHTML=`<div class="inspector-head"><div><span class="eyebrow">AGENT TELEMETRY</span><h2>${agent}</h2></div><span class="agent-status">${data.status}</span></div><div class="inspector-grid"><div><small>PROGRESS</small><b>${data.progress||0}%</b></div><div><small>PROCESSED</small><b>${Number(data.processed||0).toLocaleString()}</b></div><div><small>DECISION</small><b>${data.decision||'—'}</b></div></div><div class="agent-log">${(data.log||['Awaiting live telemetry…']).map(x=>`<div><span>›</span>${x}</div>`).join('')}</div>`;
}
function selectAgent(name){state.selectedAgent=name;$$('.agent').forEach(a=>a.classList.toggle('active',a.dataset.agent===name));agentPanel(name)}
$$('.agent').forEach(a=>a.addEventListener('click',()=>selectAgent(a.dataset.agent||a.textContent.trim().split(/\s+/)[0])));

function applyEvent(e){
 if(!e||typeof e!=='object')return;
 if(e.type==='scan_progress')Object.assign(state.scan,e);
 if(e.type==='agent')state.agents[e.agent]=e;
 if(e.type==='connection')setConnection(Boolean(e.connected));
 if(e.type==='alert'&&e.message){const stream=$('.alerts');if(stream){const row=document.createElement('div');row.className=`alert ${e.severity||'amber'}`;row.innerHTML=`<b>${(e.severity||'info').toUpperCase()}</b><span>${e.message}</span><small>NOW</small>`;stream.prepend(row);while(stream.children.length>7)stream.lastElementChild.remove()}}
 renderScan();agentPanel(state.selectedAgent);
}
async function hydrate(){
 try{const r=await fetch(`${API_BASE}/dashboard`,{cache:'no-store'});if(!r.ok)throw new Error(r.status);const data=await r.json();Object.entries(data.agents||{}).forEach(([k,v])=>state.agents[k]=v);if(data.scan)Object.assign(state.scan,data.scan);if(data.connected!==undefined)setConnection(data.connected);renderScan();agentPanel(state.selectedAgent)}catch{setConnection(false)}
}
function connectStream(){
 try{const es=new EventSource(`${API_BASE}/events`);es.onopen=()=>setConnection(true);es.onerror=()=>setConnection(false);es.onmessage=e=>{try{applyEvent(JSON.parse(e.data))}catch{}};return es}catch{return null}
}

renderScan();selectAgent('TECHNICAL');hydrate();connectStream();
setInterval(hydrate,15000);
