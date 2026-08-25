const agents=document.querySelectorAll('.agent');agents.forEach(a=>a.addEventListener('click',()=>{agents.forEach(x=>x.classList.remove('active'));a.classList.add('active')}));
let values=[2557,347,83,21,2];let nodes=document.querySelectorAll('.pipeline b');setInterval(()=>{const i=Math.floor(Math.random()*values.length);if(nodes[i])nodes[i].textContent=values[i].toLocaleString();},1400);
