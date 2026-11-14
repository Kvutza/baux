open Js_of_ocaml

(* Shared documentation controls. All content is inserted as text or DOM nodes. *)
let init () =
  let boot = Js.Unsafe.js_expr {|
(function () {
  const input = document.getElementById('docs-search');
  if (!input) return;
  const panel = document.getElementById('docs-results');
  const matches = document.getElementById('docs-search-matches');
  const status = document.getElementById('docs-search-status');
  const nav = document.getElementById('docs-nav');
  const menu = document.getElementById('docs-menu');
  const close = document.getElementById('docs-close');
  const shade = document.getElementById('docs-shade');
  const theme = document.getElementById('docs-theme');
  const safeRoute = value => /^[a-z0-9-]+\.html(?:#[A-Za-z0-9_.-]+)?$/.test(value);
  const index = Array.from(document.querySelectorAll('#docs-index a')).map(link => ({
    title: link.textContent.trim(),
    route: link.getAttribute('href'),
    summary: link.dataset.summary || '',
    text: (link.textContent + ' ' + (link.dataset.search || '')).toLowerCase()
  })).filter(item => safeRoute(item.route));

  function closeSearch() {
    panel.hidden = true;
    input.setAttribute('aria-expanded', 'false');
  }
  function search() {
    const query = input.value.trim().toLowerCase();
    matches.replaceChildren();
    if (!query) { closeSearch(); return; }
    const words = query.split(/\s+/);
    const found = index.filter(item => words.every(word => item.text.includes(word)));
    found.sort((a, b) => Number(b.title.toLowerCase().includes(query)) - Number(a.title.toLowerCase().includes(query)));
    status.textContent = found.length ? found.length + ' results' + (found.length > 30 ? ' · showing the first 30' : '') : 'No matching documentation. Try a function name or topic.';
    for (const item of found.slice(0, 30)) {
      const link = document.createElement('a');
      link.href = item.route;
      const title = document.createElement('strong');
      title.textContent = item.title;
      const summary = document.createElement('span');
      summary.textContent = item.summary;
      link.append(title, summary);
      matches.append(link);
    }
    panel.hidden = false;
    input.setAttribute('aria-expanded', 'true');
  }
  input.addEventListener('input', search);
  input.addEventListener('focus', search);
  input.addEventListener('keydown', event => {
    if (event.key === 'ArrowDown' && matches.firstElementChild) {
      event.preventDefault(); matches.firstElementChild.focus();
    } else if (event.key === 'Enter' && !panel.hidden && matches.firstElementChild) {
      event.preventDefault(); matches.firstElementChild.click();
    }
  });
  matches.addEventListener('keydown', event => {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      const next = event.key === 'ArrowDown' ? document.activeElement.nextElementSibling : document.activeElement.previousElementSibling;
      (next || input).focus();
    }
  });
  matches.addEventListener('click', closeSearch);
  document.addEventListener('click', event => { if (!event.target.closest('.docs-search')) closeSearch(); });

  function setNav(open) {
    document.body.dataset.navOpen = String(open);
    menu.setAttribute('aria-expanded', String(open));
    menu.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    shade.hidden = !open;
    if (open) close.focus();
  }
  menu.addEventListener('click', () => setNav(document.body.dataset.navOpen !== 'true'));
  close.addEventListener('click', () => { setNav(false); menu.focus(); });
  shade.addEventListener('click', () => { setNav(false); menu.focus(); });
  nav.addEventListener('click', event => { if (event.target.closest('a')) setNav(false); });
  document.addEventListener('keydown', event => {
    const editable = event.target.closest('input, textarea, [contenteditable="true"]');
    if ((event.key === '/' && !editable) || ((event.ctrlKey || event.metaKey) && event.key === 'k')) {
      event.preventDefault(); input.focus(); input.select();
    }
    if (event.key === 'Escape') {
      if (document.body.dataset.navOpen === 'true') { setNav(false); menu.focus(); }
      else if (!panel.hidden) { input.focus(); closeSearch(); }
    }
    if (event.key === 'Tab' && document.body.dataset.navOpen === 'true') {
      const links = Array.from(nav.querySelectorAll('a, button'));
      if (event.shiftKey && document.activeElement === links[0]) { event.preventDefault(); links.at(-1).focus(); }
      else if (!event.shiftKey && document.activeElement === links.at(-1)) { event.preventDefault(); links[0].focus(); }
    }
  });
  function setTheme(dark) {
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    theme.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
  }
  let preference;
  try { preference = localStorage.getItem('baux-docs-theme'); } catch (_) {}
  setTheme(preference ? preference === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches);
  theme.addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme !== 'dark';
    setTheme(dark);
    try { localStorage.setItem('baux-docs-theme', dark ? 'dark' : 'light'); } catch (_) {}
  });

  function redirect() {
    const id = location.hash.slice(1);
    const target = document.getElementById(id);
    if (target && target.classList.contains('reference-alias')) {
      const route = target.querySelector('a')?.getAttribute('href');
      if (route && safeRoute(route)) location.replace(route);
    }
  }
  redirect();
  window.addEventListener('hashchange', redirect);

  const keywords = new Set(['import', 'from', 'as', 'for', 'in', 'if', 'else', 'elif', 'return', 'def', 'class', 'with', 'assert', 'True', 'False', 'None', 'lambda', 'and', 'or', 'not']);
  for (const pre of document.querySelectorAll('article pre')) {
    const code = pre.textContent;
    const tokens = /#[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b\d+(?:\.\d+)?\b|\b[A-Za-z_]\w*\b/g;
    const fragment = document.createDocumentFragment();
    let end = 0;
    for (const match of code.matchAll(tokens)) {
      fragment.append(document.createTextNode(code.slice(end, match.index)));
      const word = match[0];
      const kind = word[0] === '#' ? 'comment' : /["']/.test(word[0]) ? 'string' : /^\d/.test(word) ? 'number' : keywords.has(word) ? 'keyword' : '';
      if (kind) {
        const span = document.createElement('span');
        span.className = 'code-' + kind;
        span.textContent = word;
        fragment.append(span);
      } else fragment.append(document.createTextNode(word));
      end = match.index + word.length;
    }
    fragment.append(document.createTextNode(code.slice(end)));
    pre.replaceChildren(fragment);
    const wrapper = document.createElement('div');
    wrapper.className = 'code-wrap';
    pre.before(wrapper); wrapper.append(pre);
    const copy = document.createElement('button');
    copy.type = 'button'; copy.className = 'code-copy'; copy.textContent = 'Copy';
    copy.setAttribute('aria-label', 'Copy code');
    copy.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(code); copy.textContent = 'Copied'; }
      catch (_) { copy.textContent = 'Select code'; const range = document.createRange(); range.selectNodeContents(pre); const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range); }
      setTimeout(() => { copy.textContent = 'Copy'; }, 2000);
    });
    wrapper.append(copy);
  }
})
|} in
  ignore (Js.Unsafe.fun_call boot [||])
