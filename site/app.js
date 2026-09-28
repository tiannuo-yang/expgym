const models = [
  {name:'Kimi K3',search:[63.4,49.1,15.7]},
  {name:'GLM-5.3',search:[64.9,53.1,19.0]},
  {name:'Qwen3.8',search:[57.7,50.2,18.2]},
  {name:'DeepSeek V4 Flash',search:[43.4,41.6,16.9]},
  {name:'GPT-5.6-sol',search:[64.1,52.6,15.4]},
  {name:'Gemini 3.8 Flash',search:[71.9,54.4,17.6]}
];
function drawHero(){
  const selected=Number(document.querySelector('#model-select').value),model=models[selected];
  const w=Math.max(280,document.querySelector('#hero-chart').clientWidth),x=[43,w/2,w-35],y=v=>240-v*2.45;
  let s=`<svg viewBox="0 0 ${w} 286" role="img" aria-labelledby="hero-svg-title hero-svg-desc"><title id="hero-svg-title">Search performance under shrinking feedback budgets</title><desc id="hero-svg-desc">${model.name}: ${model.search[0]} percent with free feedback, ${model.search[1]} with a moderate budget, and ${model.search[2]} with a tight budget. Gray lines show the other five models.</desc>`;
  for(const v of [0,20,40,60,80])s+=`<line x1="33" y1="${y(v)}" x2="${w-20}" y2="${y(v)}" stroke="#dde1e5" stroke-dasharray="2 4"/><text x="22" y="${y(v)+4}" text-anchor="end" fill="#626972" font-size="12" font-family="monospace">${v}</text>`;
  const line=(m,color,width)=>`<polyline points="${m.search.map((v,i)=>`${x[i]},${y(v)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round"/>`;
  models.forEach((m,i)=>{if(i!==selected)s+=line(m,'#c8cdd3',1.5)});
  s+=line(model,'#d94420',3);
  model.search.forEach((v,i)=>{s+=`<circle cx="${x[i]}" cy="${y(v)}" r="5" fill="#d94420" stroke="#f9fafb" stroke-width="2"/><text x="${x[i]}" y="${y(v)-16}" text-anchor="middle" fill="#b53113" font-size="20" font-family="Arial" font-weight="500" paint-order="stroke" stroke="#f9fafb" stroke-width="7" stroke-linejoin="round">${v.toFixed(1)}</text>`});
  ['Free','Moderate','Tight'].forEach((label,i)=>{s+=`<text x="${x[i]}" y="267" text-anchor="middle" fill="#626972" font-size="12" font-family="Arial">${label}</text>`});
  s+='</svg>';
  document.querySelector('#hero-chart').innerHTML=s;
  document.querySelector('#hero-drop').innerHTML=`−${(model.search[0]-model.search[2]).toFixed(1)}<span>percentage points</span>`;
}
document.querySelector('#model-select').addEventListener('change',drawHero);
function drawEvidence(){
  const w=Math.max(280,document.querySelector('#evidence-chart').clientWidth),x=[42,w/2,w-30],y=v=>196-v*1.48;
  let s=`<svg viewBox="0 0 ${w} 236" role="img" aria-labelledby="evidence-title evidence-desc"><title id="evidence-title">Correct labels hide incomplete evidence</title><desc id="evidence-desc">For hypotheses requiring evidence, label accuracy is 85.9, 86.0 and 83.4 percent across Free, Moderate and Tight regimes. Exact evidence accuracy falls from 81.6 to 64.0 to 45.1 percent.</desc>`;
  s+=`<text x="0" y="13" fill="#626972" font-size="12" font-family="monospace">Evidence audit · accuracy (%)</text>`;
  for(const v of [0,50,100])s+=`<line x1="32" y1="${y(v)}" x2="${w-20}" y2="${y(v)}" stroke="#dde1e5" stroke-dasharray="2 4"/><text x="23" y="${y(v)+4}" text-anchor="end" fill="#626972" font-size="12" font-family="monospace">${v}</text>`;
  [[85.9,86,83.4],[81.6,64,45.1]].forEach((values,index)=>{
    const color=index?'#d94420':'#626972';
    s+=`<polyline points="${values.map((v,i)=>`${x[i]},${y(v)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2.5"/>`;
    values.forEach((v,i)=>{s+=`<circle cx="${x[i]}" cy="${y(v)}" r="4" fill="${color}" stroke="#f9fafb" stroke-width="1.5"/>`;if(i!==1)s+=`<text x="${x[i]}" y="${y(v)+(index?23:-12)}" text-anchor="middle" fill="${color}" font-size="14" font-family="Arial" paint-order="stroke" stroke="#f9fafb" stroke-width="5">${v.toFixed(1)}</text>`});
    s+=`<text x="${w/2}" y="${index?140:49}" text-anchor="middle" fill="${color}" font-size="12" font-family="Arial" paint-order="stroke" stroke="#f9fafb" stroke-width="5">${index?'Complete evidence':'Correct label'}</text>`;
  });
  ['Free','Moderate','Tight'].forEach((t,i)=>s+=`<text x="${x[i]}" y="224" text-anchor="middle" fill="#626972" font-size="12" font-family="Arial">${t}</text>`);
  document.querySelector('#evidence-chart').innerHTML=s+'</svg>';
}
let activeRegime='tight';
function drawScaling(){
  const w=Math.max(280,document.querySelector('#scaling-chart').clientWidth),end=w-55,x=n=>40+(n-2)/6*(end-40),y=v=>261-v*2.2;
  const data=scalingData[activeRegime];
  let s=`<svg viewBox="0 0 ${w} 308" role="img" aria-labelledby="scaling-title scaling-desc"><title id="scaling-title">PoolAct scaling under ${activeRegime} budget</title><desc id="scaling-desc">GLM-5.3 exact evidence-set accuracy at 2, 4, 6 and 8 agents. ${Object.entries(data).map(([k,v])=>k+': '+v.map(d=>d[1].toFixed(1)).join(', ')).join('. ')}. Fixed per-agent budgets, with increasing total budget.</desc>`;
  for(const v of [0,25,50,75,100])s+=`<line x1="34" y1="${y(v)}" x2="${end+9}" y2="${y(v)}" stroke="#3d444d" stroke-dasharray="2 5"/><text x="25" y="${y(v)+4}" text-anchor="end" fill="#a8b0ba" font-size="12" font-family="monospace">${v}</text>`;
  const specs=[['naive','#85909e','5 5'],['cached','#d3d9e1',''],['poolact','#ff7652','']];
  specs.forEach(([key,color,dash])=>{
    const values=data[key];
    s+=`<polyline points="${values.map(([n,v])=>`${x(n)},${y(v)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="${key==='poolact'?3:2}" stroke-dasharray="${dash}" stroke-linejoin="round"/>`;
    values.forEach(([n,v])=>s+=`<circle cx="${x(n)}" cy="${y(v)}" r="${key==='poolact'?4.5:3.5}" fill="${color}" stroke="#161a1f" stroke-width="2"/>`);
    const v=values[values.length-1][1];
    const adjust=activeRegime==='tight'?(key==='cached'?-6:key==='naive'?11:4):4;
    s+=`<text x="${end+10}" y="${y(v)+adjust}" fill="${color}" font-size="13" font-family="Arial" font-weight="500">${v.toFixed(1)}</text>`;
  });
  [2,4,6,8].forEach(n=>s+=`<text x="${x(n)}" y="281" text-anchor="middle" fill="#c4cbd4" font-size="12" font-family="monospace">${n}</text>`);
  s+=`<text x="${w/2}" y="304" text-anchor="middle" fill="#a8b0ba" font-size="12" font-family="Arial">Number of agents</text></svg>`;
  document.querySelector('#scaling-chart').innerHTML=s;
  const pool=data.poolact.at(-1)[1],naive=data.naive.at(-1)[1];
  document.querySelector('#scaling-conclusion').innerHTML=`<strong>${pool===100?'100':pool.toFixed(1)}%</strong><span>with PoolAct at 8 agents<br>vs. ${naive.toFixed(1)}% independent</span>`;
}
document.querySelectorAll('[data-regime]').forEach(button=>button.addEventListener('click',()=>{
  activeRegime=button.dataset.regime;
  document.querySelectorAll('[data-regime]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
  drawScaling();
}));
const quickTabs=Array.from(document.querySelectorAll('.quick-tabs [role="tab"]'));
function selectQuickTab(tab,moveFocus=false){
  quickTabs.forEach(item=>{
    const selected=item===tab;
    item.setAttribute('aria-selected',String(selected));
    item.tabIndex=selected?0:-1;
    document.getElementById(item.getAttribute('aria-controls')).hidden=!selected;
  });
  if(moveFocus)tab.focus();
}
quickTabs.forEach((tab,index)=>{
  tab.addEventListener('click',()=>selectQuickTab(tab));
  tab.addEventListener('keydown',event=>{
    let next;
    if(event.key==='ArrowRight')next=(index+1)%quickTabs.length;
    if(event.key==='ArrowLeft')next=(index-1+quickTabs.length)%quickTabs.length;
    if(event.key==='Home')next=0;
    if(event.key==='End')next=quickTabs.length-1;
    if(next!==undefined){event.preventDefault();selectQuickTab(quickTabs[next],true)}
  });
});
document.querySelectorAll('[data-next]').forEach(button=>button.addEventListener('click',()=>selectQuickTab(document.getElementById('tab-'+button.dataset.next),true)));
document.querySelectorAll('.quick-panel pre').forEach(pre=>{pre.tabIndex=0;pre.setAttribute('aria-label','Scrollable command example')});
document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{
  const code=document.getElementById(button.dataset.copy);
  try{
    await navigator.clipboard.writeText(code.textContent);
    button.textContent='Copied';
    document.querySelector('#copy-status').textContent='Commands copied to clipboard.';
    setTimeout(()=>button.textContent='Copy',2000);
  }catch{
    const selection=window.getSelection(),range=document.createRange();range.selectNodeContents(code);selection.removeAllRanges();selection.addRange(range);
    button.textContent='Selected';
    document.querySelector('#copy-status').textContent='Commands selected. Use your keyboard’s copy shortcut.';
  }
}));
const resizeObserver=new ResizeObserver(()=>{drawHero();drawEvidence();drawScaling()});
resizeObserver.observe(document.querySelector('#hero-chart'));
resizeObserver.observe(document.querySelector('#evidence-chart'));
resizeObserver.observe(document.querySelector('#scaling-chart'));
drawHero();drawEvidence();drawScaling();
