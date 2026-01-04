// Mobile nav toggle
(function(){
  const btn = document.getElementById('navbtn');
  const nav = document.getElementById('nav');
  if(btn && nav){
    btn.addEventListener('click', () => {
      const isOpen = nav.classList.toggle('open');
      btn.setAttribute('aria-expanded', String(isOpen));
    });

    // close on click
    nav.querySelectorAll('a').forEach(a => {
      a.addEventListener('click', () => {
        nav.classList.remove('open');
        btn.setAttribute('aria-expanded', 'false');
      });
    });
  }
})();

// Scrollspy for nav active link
(function(){
  const links = Array.from(document.querySelectorAll('.nav a'));
  const map = new Map();
  links.forEach(a => {
    const id = a.getAttribute('href');
    if(id && id.startsWith('#')) map.set(id, a);
  });

  const sections = Array.from(document.querySelectorAll('main section[id]'));
  if(!sections.length) return;

  const obs = new IntersectionObserver((entries) => {
    // pick the entry that is most visible
    const visible = entries.filter(e => e.isIntersecting).sort((a,b)=>b.intersectionRatio - a.intersectionRatio)[0];
    if(!visible) return;

    const id = '#' + visible.target.id;
    links.forEach(a => a.classList.remove('active'));
    const active = map.get(id);
    if(active) active.classList.add('active');
  }, { root: null, threshold: [0.22, 0.35, 0.5, 0.65] });

  sections.forEach(s => obs.observe(s));
})();

// Simple carousel (vanilla)
(function(){
  const carousels = Array.from(document.querySelectorAll('[data-carousel]'));
  carousels.forEach((root) => {
    const track = root.querySelector('.carousel__track');
    const items = Array.from(root.querySelectorAll('.carousel__item'));
    const btnPrev = root.querySelector('[data-carousel-prev]');
    const btnNext = root.querySelector('[data-carousel-next]');
    if(!track || items.length <= 1) return;

    let idx = 0;
    const clamp = (n) => (n + items.length) % items.length;

    const render = () => {
      track.style.transform = `translateX(-${idx * 100}%)`;
      root.setAttribute('data-index', String(idx));
    };

    const go = (delta) => {
      idx = clamp(idx + delta);
      render();
    };

    btnPrev && btnPrev.addEventListener('click', () => go(-1));
    btnNext && btnNext.addEventListener('click', () => go(1));

    // swipe for mobile
    let startX = null;
    root.addEventListener('touchstart', (e) => { startX = e.touches[0].clientX; }, {passive:true});
    root.addEventListener('touchend', (e) => {
      if(startX === null) return;
      const endX = e.changedTouches[0].clientX;
      const dx = endX - startX;
      startX = null;
      if(Math.abs(dx) < 40) return;
      go(dx > 0 ? -1 : 1);
    }, {passive:true});

    render();
  });
})();


// Lightbox for model images
(function(){
  const lb = document.getElementById('lightbox');
  if(!lb) return;

  const img = lb.querySelector('.lightbox__img');
  const title = lb.querySelector('[data-lightbox-title]');
  const btnClose = lb.querySelector('[data-lightbox-close]');

  const open = (src, text) => {
    if(!img) return;
    img.src = src;
    img.alt = text || 'Ảnh mô hình';
    if(title) title.textContent = text || '';
    lb.classList.add('open');
    lb.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
  };

  const close = () => {
    lb.classList.remove('open');
    lb.setAttribute('aria-hidden', 'true');
    if(img) img.src = '';
    document.body.style.overflow = '';
  };

  btnClose && btnClose.addEventListener('click', close);
  lb.addEventListener('click', (e) => { if(e.target === lb) close(); });
  document.addEventListener('keydown', (e) => {
    if(e.key === 'Escape' && lb.classList.contains('open')) close();
  });

  document.querySelectorAll('[data-lightbox]').forEach((el) => {
    el.addEventListener('click', () => {
      const src = el.getAttribute('src');
      const text = el.getAttribute('data-lightbox') || el.getAttribute('alt') || '';
      if(src) open(src, text);
    });
  });
})();
