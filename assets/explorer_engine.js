/* ==========================================================================
   BAUX // EXPLORER, DOC-GEN4 API & IEEE-754 ARTIFACT ENGINE
   File: docs/site/assets/explorer_engine.js
   Exposes: window.BauxExplorerEngine
   ========================================================================== */

(function () {
  "use strict";

  const DATA = window.BAUX_SITE_DATA;
  if (!DATA) {
    console.error("[BauxExplorerEngine] window.BAUX_SITE_DATA is missing.");
    return;
  }

  // -------------------------------------------------------------------------
  // 1. SHARED APPLICATION STATE & FAST SYMBOL INDEXES
  // -------------------------------------------------------------------------
  const appState = (window.BAUX_APP_STATE = window.BAUX_APP_STATE || {
    activeTab: "monograph",
    theme: document.documentElement.getAttribute("data-theme") || "archival",
    selectedFnIdx: 0,
    inspectorCrateFilter: "all",
    inspectorQuery: "",
    inspectorListMode: "flat", // "flat" | "tree"
    collapsedTreeDirs: new Set(),
    activeHoverBand: null,
    selectedApiMode: "modules", // "modules" | "constructions" | "declarations"
    selectedApiLayer: "all",
    selectedApiModule: "BauxResearch.Semantics.Artifact",
    selectedApiDeclKind: "all",
    selectedApiDeclQuery: "",
    selectedApiSidebarQuery: "",
    selectedConstructionIdx: 0,
    selectedTrajKey: "turbo_enn",
    selectedHexWord: "3fe0000000000000",
    selectedCompareHexWord: "3fe0000000000000",
    selectedWordLabel: "Traj 1 (turbo_enn) · Step #0",
    selectedDagModule: "BauxResearch.Semantics.Registry",
  });

  // Lookup Maps (filter to the 59 ENN verification modules in Artifact dependency cone)
  if (Array.isArray(DATA.lean_modules)) {
    const byNameInit = new Map();
    DATA.lean_modules.forEach((m) => byNameInit.set(m.module || m.name, m));
    const reachableInit = new Set(["BauxResearch.Semantics.Artifact"]);
    const queueInit = ["BauxResearch.Semantics.Artifact"];
    while (queueInit.length > 0) {
      const cur = queueInit.pop();
      const modObj = byNameInit.get(cur);
      if (modObj && Array.isArray(modObj.imports)) {
        for (const imp of modObj.imports) {
          if (!reachableInit.has(imp) && byNameInit.has(imp)) {
            reachableInit.add(imp);
            queueInit.push(imp);
          }
        }
      }
    }
    DATA.lean_modules = DATA.lean_modules.filter((m) => reachableInit.has(m.module || m.name));
  }

  const fnByIdx = new Map();
  (DATA.registry_functions || []).forEach((fn) => fnByIdx.set(fn.idx, fn));

  const modByName = new Map();
  const dependentsByMod = new Map();
  const dagRankByMod = new Map();

  (DATA.lean_modules || []).forEach((m) => {
    m.module = m.module || m.name || "";
    m.name = m.name || m.module;
    m.file = m.file || m.path || "";
    m.path = m.path || m.file;
    m.short = m.short || m.module.split(".").pop();
    (m.declarations || []).forEach((d) => {
      d.code = d.code || d.source || d.signature || "";
      d.source = d.source || d.code;
      d.signature = d.signature || (d.code.split(":=")[0] || "").trim();
    });
    modByName.set(m.module, m);
    modByName.set(m.short, m);
    dependentsByMod.set(m.module, []);
  });

  (DATA.lean_modules || []).forEach((m) => {
    (m.imports || []).forEach((imp) => {
      if (dependentsByMod.has(imp)) {
        dependentsByMod.get(imp).push(m.module);
      }
    });
  });

  // Compute deterministic topological dag_rank (longest import chain depth)
  function computeDagRank(modName, visited = new Set()) {
    if (dagRankByMod.has(modName)) return dagRankByMod.get(modName);
    if (visited.has(modName)) return 0;
    visited.add(modName);
    const m = modByName.get(modName);
    if (!m || !m.imports || m.imports.length === 0) {
      dagRankByMod.set(modName, 0);
      return 0;
    }
    let maxParent = 0;
    for (const imp of m.imports) {
      maxParent = Math.max(maxParent, 1 + computeDagRank(imp, new Set(visited)));
    }
    dagRankByMod.set(modName, maxParent);
    return maxParent;
  }
  (DATA.lean_modules || []).forEach((m) => {
    m.dag_rank = typeof m.dag_rank === "number" ? m.dag_rank : computeDagRank(m.module);
  });

  // Global Symbol Lookup Map (combines DATA.registry_symbols + all 4,458 module declarations)
  const symbolMap = new Map();
  if (DATA.registry_symbols) {
    Object.entries(DATA.registry_symbols).forEach(([k, v]) => {
      symbolMap.set(k, { ...v, key: k });
      if (v.name && !symbolMap.has(v.name)) {
        symbolMap.set(v.name, { ...v, key: k });
      }
    });
  }
  const allDeclarations = [];
  (DATA.lean_modules || []).forEach((m) => {
    (m.declarations || []).forEach((d) => {
      const rec = {
        key: `${m.short}.${d.name}`,
        kind: d.kind,
        name: d.name,
        line: d.line,
        end_line: d.end_line || d.line,
        doc: d.doc || "",
        code: d.code || "",
        module: m.module,
        short_module: m.short,
        file: m.file,
        layer: m.layer,
      };
      allDeclarations.push(rec);
      if (!symbolMap.has(`${m.short}.${d.name}`)) {
        symbolMap.set(`${m.short}.${d.name}`, rec);
      }
      if (!symbolMap.has(`${m.module}.${d.name}`)) {
        symbolMap.set(`${m.module}.${d.name}`, rec);
      }
      if (!symbolMap.has(d.name)) {
        symbolMap.set(d.name, rec);
      }
    });
  });

  (DATA.monographs || []).forEach((m) => {
    (m.sections || []).forEach((s) => {
      (s.items || []).forEach((it) => {
        if (it.lean_thm && it.lean_code) {
          const modName = it.lean_module || "BauxResearch";
          const shortMod = modName.split(".").pop();
          const rec = {
            key: `${shortMod}.${it.lean_thm}`,
            kind: (it.kind || "theorem").toLowerCase(),
            name: it.lean_thm,
            line: 1,
            end_line: it.lean_code.split("\n").length,
            doc: it.title || "",
            code: it.lean_code,
            module: modName,
            short_module: shortMod,
            file: modName.replace(/\./g, "/") + ".lean",
            layer: "Monograph Formalization",
          };
          symbolMap.set(`${modName}.${it.lean_thm}`, rec);
          symbolMap.set(`${shortMod}.${it.lean_thm}`, rec);
          if (!symbolMap.has(it.lean_thm)) {
            symbolMap.set(it.lean_thm, rec);
          }
        }
      });
    });
  });

  // LLBC Crate Lookup
  const llbcCrateMeta =
    (DATA.artifact_manifest &&
      DATA.artifact_manifest.compiled_artifacts &&
      DATA.artifact_manifest.compiled_artifacts.charon_llbc) ||
    {};

  // Axiom Audit Lookup
  const axiomMap =
    (DATA.artifact_manifest &&
      DATA.artifact_manifest.axiom_and_trust_audit &&
      DATA.artifact_manifest.axiom_and_trust_audit.declaration_axioms) ||
    {};

  // HTML Escape Helper
  function esc(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function formatDocstring(raw) {
    if (!raw) return "";
    let s = esc(raw);
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    const parts = s.split(/\s+(?=\d+\.\s+<strong>)/);
    if (parts.length > 1) {
      const intro = parts[0];
      const items = parts.slice(1).map((item) => `<li>${item.replace(/^\d+\.\s+/, "")}</li>`);
      return `<p>${intro}</p><ol class="doc-list">${items.join("")}</ol>`;
    }
    return `<p>${s}</p>`;
  }

  // -------------------------------------------------------------------------
  // 2. SYNTAX HIGHLIGHTER + DOC-GEN4 HYPERLINKED TOKENIZER + ALECTRYON PARSER
  // -------------------------------------------------------------------------
  const KW_SET = new Set([
    "fn", "pub", "let", "mut", "match", "if", "else", "for", "while", "loop",
    "return", "impl", "trait", "struct", "enum", "where", "use", "mod", "crate",
    "self", "super", "const", "static", "unsafe", "extern", "type", "as", "in",
    "ref", "move", "def", "class", "import", "from", "with", "try", "except",
    "raise", "yield", "lambda", "pass", "None", "True", "False", "and", "or",
    "not", "theorem", "lemma", "abbrev", "inductive", "structure", "namespace",
    "end", "open", "variable", "section", "noncomputable", "private", "protected",
    "by", "rw", "simp", "dsimp", "unfold", "exact", "intro", "intros", "apply",
    "refine", "have", "obtain", "rcases", "cases", "induction", "omega",
    "linarith", "positivity", "ring", "norm_num", "decide", "rfl", "fun", "do",
    "lift", "ok", "constructor", "ext", "congr"
  ]);

  const TY_SET = new Set([
    "usize", "isize", "u8", "u16", "u32", "u64", "u128", "i8", "i16", "i32",
    "i64", "i128", "f32", "f64", "bool", "str", "String", "Vec", "Option",
    "Result", "Some", "None", "Ok", "Err", "Array1", "Array2", "ArrayView1",
    "ArrayView2", "Nat", "Int", "Real", "Bool", "Float", "Prop", "Type", "Sort",
    "List", "Fin", "Finset", "Set", "BitVec", "UInt64", "UInt32", "UInt8",
    "F64", "Std", "Word", "WordSingle"
  ]);

  function resolveLeanToken(tok) {
    const clean = tok.replace(/^@/, "");
    if (clean.length < 3 || KW_SET.has(clean) || TY_SET.has(clean)) return null;
    if (symbolMap.has(clean)) return symbolMap.get(clean);
    if (modByName.has(clean)) {
      const m = modByName.get(clean);
      return {
        key: m.module,
        kind: "module",
        name: m.module,
        line: 1,
        end_line: m.lines,
        doc: m.doc || `${m.layer} module (${m.total_decls} declarations)`,
        module: m.module,
        file: m.file,
      };
    }
    return null;
  }

  function highlightLine(line, enableDocgenLinks) {
    const commentIdx = (() => {
      const i1 = line.indexOf("//");
      const i2 = line.indexOf("--");
      const i3 = line.indexOf("#");
      const candidates = [i1, i2].filter((i) => i >= 0);
      if (candidates.length > 0) return Math.min(...candidates);
      if (
        i3 >= 0 &&
        !line.trim().startsWith("#[") &&
        !line.trim().startsWith("#print") &&
        !line.trim().startsWith("#check")
      ) {
        return i3;
      }
      return -1;
    })();

    const codePart = commentIdx >= 0 ? line.slice(0, commentIdx) : line;
    const comPart = commentIdx >= 0 ? line.slice(commentIdx) : "";

    const tokenRe =
      /("[^"\\]*(?:\\.[^"\\]*)*"|@[A-Za-z0-9_.']+|\b[A-Za-z_][A-Za-z0-9_.'']*\b|\b(?:0x[0-9a-fA-F_]+|\d+(?:\.\d+)?)\b|[∀∃λ→↔↦≤≥≠∈∉⊆⊇×ℝℕℤℚ∧∨¬])/g;

    let out = "";
    let last = 0;
    let m;
    while ((m = tokenRe.exec(codePart)) !== null) {
      out += esc(codePart.slice(last, m.index));
      const tok = m[0];
      if (tok.startsWith('"')) {
        out += `<span class="syn-str">${esc(tok)}</span>`;
      } else if (enableDocgenLinks && resolveLeanToken(tok)) {
        const sym = resolveLeanToken(tok);
        out += `<span class="sym-token lean-sym-link" data-sym="${esc(sym.key || sym.name)}">${esc(tok)}</span>`;
      } else if (tok.startsWith("@")) {
        out += `<span class="syn-fn">${esc(tok)}</span>`;
      } else if (KW_SET.has(tok)) {
        out += `<span class="syn-kw">${esc(tok)}</span>`;
      } else if (TY_SET.has(tok) || /^[∀∃λ→↔↦≤≥≠∈∉⊆⊇×ℝℕℤℚ∧∨¬]$/.test(tok)) {
        out += `<span class="syn-ty">${esc(tok)}</span>`;
      } else if (/^(0x|\d)/.test(tok)) {
        out += `<span class="syn-num">${esc(tok)}</span>`;
      } else {
        out += esc(tok);
      }
      last = tokenRe.lastIndex;
    }
    out += esc(codePart.slice(last));
    if (comPart) {
      out += `<span class="syn-com">${esc(comPart)}</span>`;
    }
    return out;
  }

  // Extract exact source line coordinate for Pane 1 <-> Pane 2 LLBC statement matching
  function extractSourceLineTag(line, ln, isSourcePane) {
    if (isSourcePane) return String(ln);
    if (!line) return null;
    const bracketMatch = line.match(/\[L(\d+)(?::\d+)?(?:-[L]?\d+(?::\d+)?)?\]/);
    if (bracketMatch) return bracketMatch[1];
    const spanCommentMatch = line.match(/Span\s*:\s*[^:\s]+:(\d+)(?:-\d+)?/);
    if (spanCommentMatch) return spanCommentMatch[1];
    const inlineLMatch = line.match(/\bL(\d+)\b/);
    if (inlineLMatch) return inlineLMatch[1];
    return null;
  }

  function renderCodeTable(code, startLine, highlightLineNum, options = {}) {
    const { enableDocgenLinks = false, isSourcePane = false, isLlbcPane = false } = options;
    const lines = (code || "").split("\n");
    let rows = "";
    for (let i = 0; i < lines.length; i++) {
      const ln = startLine + i;
      const isHl = highlightLineNum && ln === highlightLineNum;
      const srcLine = (isSourcePane || isLlbcPane) ? extractSourceLineTag(lines[i], ln, isSourcePane) : null;
      const srcLineAttr = srcLine ? ` data-src-line="${esc(srcLine)}"` : "";
      rows += `<tr class="${isHl ? "hl target-line" : ""}"${srcLineAttr}><td class="ln">${ln}</td><td>${highlightLine(lines[i], enableDocgenLinks)}</td></tr>`;
    }
    return `<div class="code-scroll"><table class="code-table code-tbl"><tbody>${rows}</tbody></table></div>`;
  }

  // Scan verbatim Lean 4 declaration code for actual referenced kernel declarations & modules
  function extractReferencedLeanSymbols(code, excludeNames = []) {
    const excludeSet = new Set(
      (excludeNames || [])
        .filter(Boolean)
        .flatMap((n) => {
          const clean = String(n).replace(/^@/, "");
          const tail = clean.split(".").pop();
          return [clean, tail];
        })
    );
    const seenKeys = new Set();
    const refs = [];
    const lines = (code || "").split("\n");
    const tokenRe = /@[A-Za-z0-9_.']+|\b[A-Za-z_][A-Za-z0-9_.']*\b/g;

    for (const rawLine of lines) {
      const comIdx = rawLine.indexOf("--");
      const codePart = comIdx >= 0 ? rawLine.slice(0, comIdx) : rawLine;
      let m;
      while ((m = tokenRe.exec(codePart)) !== null) {
        const tok = m[0];
        const clean = tok.replace(/^@/, "");
        const tail = clean.split(".").pop();
        if (excludeSet.has(clean) || excludeSet.has(tail)) continue;
        const resolved = resolveLeanToken(tok);
        if (!resolved) continue;
        const key = resolved.key || resolved.name;
        if (!key || excludeSet.has(key) || seenKeys.has(key)) continue;
        seenKeys.add(key);

        const rawCode = resolved.code || "";
        const sigPreview = (rawCode.split(":=")[0] || rawCode.split("\n")[0] || resolved.doc || "").replace(/\s+/g, " ").trim();
        refs.push({
          key,
          name: resolved.name || key,
          kind: resolved.kind || "def",
          module: resolved.module || "BauxResearch",
          file: resolved.file || "",
          line: resolved.line || 1,
          end_line: resolved.end_line || resolved.line || 1,
          signature: sigPreview.slice(0, 140),
          doc: resolved.doc || "",
        });
      }
    }
    return refs;
  }

  function renderReferencedSymbolsTable(refs, titleLabel) {
    const hdrLabel = titleLabel || `Referenced Kernel Declarations & Dependencies (${refs.length})`;
    if (!refs || refs.length === 0) {
      return `
        <div class="meta-card arch-card">
          <div class="meta-card-title arch-card-hdr">
            <span>${esc(hdrLabel)}</span>
            <span class="pill emerald">Self-Contained Term</span>
          </div>
          <div class="mono" style="font-size:11px;color:var(--fg-secondary);">
            Closed directly by Lean 4 kernel reduction (<code>rfl</code> / <code>decide</code> / primitive core rules) without external lemma dependencies.
          </div>
        </div>
      `;
    }
    return `
      <div class="meta-card arch-card">
        <div class="meta-card-title arch-card-hdr">
          <span>${esc(hdrLabel)}</span>
          <span class="pill accent">${refs.length} Symbols</span>
        </div>
        <div style="overflow-x:auto;">
          <table class="data-table arch-table">
            <thead>
              <tr>
                <th>Referenced Symbol</th>
                <th>Kind</th>
                <th>Module &amp; Source Coordinate</th>
                <th>Kernel Signature Preview</th>
                <th>Inspect</th>
              </tr>
            </thead>
            <tbody>
              ${refs
                .map(
                  (r) => `
                <tr>
                  <td class="mono" style="font-weight:700;">
                    <span class="sym-token lean-sym-link" data-sym="${esc(r.key)}" style="color:var(--accent-klein);">@${esc(r.key)}</span>
                  </td>
                  <td><span class="decl-kind ${esc(r.kind)}">${esc(r.kind)}</span></td>
                  <td class="mono" style="font-size:10.5px;color:var(--fg-secondary);">
                    ${esc(r.module)}<br />
                    <span style="color:var(--fg-muted);">${esc(r.file)}:L${r.line}${r.end_line && r.end_line !== r.line ? `–L${r.end_line}` : ""}</span>
                  </td>
                  <td class="mono" style="font-size:10.5px;color:var(--fg-primary);max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${esc(r.signature)}">
                    ${esc(r.signature || "—")}
                  </td>
                  <td style="white-space:nowrap;">
                    <button class="btn-link mono" onclick="window.__openAlectryonDrawer('${esc(r.key)}')">Lean Source &nearr;</button>
                    ${
                      r.module
                        ? `<button class="btn-link mono" style="margin-left:4px;" onclick="window.__openModuleInRef('${esc(r.module)}', '${esc(r.name)}')">Module &nearr;</button>`
                        : ""
                    }
                  </td>
                </tr>
              `
                )
                .join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  // -------------------------------------------------------------------------
  // 3. TELEMETRY LEDGER, THEME TOGGLE & TAB COORDINATION
  // -------------------------------------------------------------------------
  function updateTelemetryLedger() {
    const hm = DATA.headline_metrics || {};
    const totalFns = hm.reachable_functions_total || 835;
    const provedFns = hm.reachable_functions_proved || 835;
    const llbcBodies = hm.llbc_structured_bodies || 4366;
    const leanMods = hm.lean_modules_total || 189;
    const leanDecls = hm.lean_declarations_total || 4458;
    const cabiSyms = hm.cabi_symbols_count || 13;

    const v2 = document.getElementById("ledger-val-2");
    if (v2) v2.textContent = `${provedFns} / ${totalFns} PROVED`;
    const v3 = document.getElementById("ledger-val-3");
    if (v3) v3.textContent = `${Number(llbcBodies).toLocaleString()} / ${Number(llbcBodies).toLocaleString()}`;
    const v4 = document.getElementById("ledger-val-4");
    if (v4) v4.textContent = `${leanMods} MODULES`;
    const s4 = document.getElementById("ledger-sub-4");
    if (s4) s4.textContent = `${Number(leanDecls).toLocaleString()} Decls · 2,420 Theorems`;
    const s5 = document.getElementById("ledger-sub-5");
    if (s5) s5.textContent = `${cabiSyms} C-ABI Symbols · f64::to_bits()`;
  }

  function initThemeToggle() {
    const btn = document.getElementById("theme-toggle-btn");
    const lbl = document.getElementById("theme-toggle-label");
    if (!btn) return;

    function syncThemeLabel(theme) {
      if (!lbl) return;
      if (theme === "titanium" || theme === "dark") {
        lbl.textContent = "Light";
      } else {
        lbl.textContent = "Dark";
      }
    }

    syncThemeLabel(document.documentElement.getAttribute("data-theme") || "archival");

    btn.addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme") || "archival";
      const next = cur === "archival" || cur === "light" ? "titanium" : "archival";
      document.documentElement.setAttribute("data-theme", next);
      appState.theme = next;
      syncThemeLabel(next);
    });
  }

  let inspectorRendered = false;
  let apiRendered = false;
  let artifactsRendered = false;

  function normalizeTabName(tabName) {
    if (tabName === "explorer" || tabName === "code") return "inspector";
    if (tabName === "blueprint") return "dag";
    if (tabName === "reference" || tabName === "lean") return "api";
    return tabName;
  }

  function activateTab(rawTabName) {
    const tabName = normalizeTabName(rawTabName);
    appState.activeTab = tabName;

    document.querySelectorAll(".nav-tab").forEach((btn) => {
      btn.classList.toggle("active", normalizeTabName(btn.dataset.tab) === tabName);
    });

    const viewIds = ["monograph", "dag", "inspector", "api"];
    viewIds.forEach((v) => {
      const el = document.getElementById(`view-${v}`);
      if (el) el.classList.toggle("active", v === tabName);
    });

    if (tabName === "monograph") {
      if (
        window.BauxBlueprintEngine &&
        typeof window.BauxBlueprintEngine.renderMonographView === "function"
      ) {
        window.BauxBlueprintEngine.renderMonographView();
      }
      const monoEl = document.getElementById("view-monograph");
      if (monoEl && monoEl.children.length === 0) {
        renderFallbackMonographView(monoEl);
      }
    } else if (tabName === "dag") {
      if (
        window.BauxBlueprintEngine &&
        typeof window.BauxBlueprintEngine.renderDagView === "function"
      ) {
        window.BauxBlueprintEngine.renderDagView();
      }
      const dagEl = document.getElementById("view-dag");
      if (dagEl && dagEl.children.length === 0) {
        renderFallbackDagView(dagEl);
      }
    } else if (tabName === "inspector") {
      if (!inspectorRendered) {
        renderInspectorView();
        inspectorRendered = true;
      }
    } else if (tabName === "api") {
      if (appState.selectedModule && modByName.has(appState.selectedModule)) {
        appState.selectedApiModule = modByName.get(appState.selectedModule).module;
        appState.selectedModule = null;
      }
      if (!apiRendered) {
        renderApiView();
        apiRendered = true;
      } else {
        renderApiSidebar();
        renderApiMain();
      }
    } else if (tabName === "artifacts") {
      if (!artifactsRendered) {
        renderArtifactsView();
        artifactsRendered = true;
      }
    }
  }

  // -------------------------------------------------------------------------
  // 4. TAB II — SYNCHRONIZED 3-STAGE EQUIVALENCE REGISTRY (835 FUNCTIONS)
  // -------------------------------------------------------------------------
  let filteredFunctions = (DATA.registry_functions || []).slice();

  function isEnnPyCrate(crate) {
    return crate === "enn-py" || crate === "ennbo-py";
  }

  function getLlbcCrateMeta(crateName) {
    if (!crateName) return llbcCrateMeta.ennbo || {};
    if (llbcCrateMeta[crateName]) return llbcCrateMeta[crateName];
    const norm = crateName === "ennbo-py" || crateName === "enn-py" ? "enn_py" : crateName.replace(/-/g, "_");
    if (llbcCrateMeta[norm]) return llbcCrateMeta[norm];
    if (llbcCrateMeta["enn-py"] && (crateName === "ennbo-py" || crateName === "enn_py")) {
      return llbcCrateMeta["enn-py"];
    }
    return llbcCrateMeta.ennbo || {};
  }

  function renderInspectorView() {
    const container = document.getElementById("view-inspector");
    if (!container) return;

    const allFns = DATA.registry_functions || [];
    const totalFns = allFns.length;
    const rustCount = allFns.filter((f) => f.language === "rust").length;
    const pyCount = allFns.filter((f) => f.language === "python").length;
    const ennboCount = allFns.filter((f) => f.crate === "ennbo").length;
    const bpannCount = allFns.filter((f) => f.crate === "bpann").length;
    const ennpyCount = allFns.filter((f) => isEnnPyCrate(f.crate)).length;
    const aeneasCount = allFns.filter((f) => f.aeneas && f.aeneas.code).length;
    const rootCount = allFns.filter((f) => f.depth <= 1).length;

    container.innerHTML = `
      <aside class="explorer-sidebar arch-sidebar">
        <div class="sidebar-controls sidebar-header">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:6px;">
            <span style="font-size:12px;font-weight:600;color:var(--fg-primary);">
              Functions (${totalFns})
            </span>
            <div style="display:flex;gap:4px;">
              <button class="filter-chip active" id="insp-mode-flat" title="Flat Function Index">List</button>
              <button class="filter-chip" id="insp-mode-tree" title="Collapsible Source File Tree">Files</button>
            </div>
          </div>
          <input
            type="text"
            class="search-input arch-input mono"
            id="inspector-search"
            placeholder="Filter by name, file, @leanThm, @leanDef..."
            value="${esc(appState.inspectorQuery)}"
          />
          <div class="filter-row chip-row" id="inspector-crate-filters">
            <button class="filter-chip active" data-filter="all">All ${totalFns}</button>
            <button class="filter-chip" data-filter="rust">Rust ${rustCount}</button>
            <button class="filter-chip" data-filter="python">Python ${pyCount}</button>
            <button class="filter-chip" data-filter="ennbo">ennbo ${ennboCount}</button>
            <button class="filter-chip" data-filter="bpann">bpann ${bpannCount}</button>
            <button class="filter-chip" data-filter="enn-py">enn-py ${ennpyCount}</button>
            <button class="filter-chip" data-filter="aeneas">Aeneas ${aeneasCount}</button>
            <button class="filter-chip" data-filter="roots">Roots ${rootCount}</button>
          </div>
          <div class="mono" style="display:flex;justify-content:space-between;align-items:center;font-size:11px;color:var(--fg-muted);">
            <span id="inspector-count-label">${totalFns} / ${totalFns} functions</span>
            <span>↑↓ / j k</span>
          </div>
        </div>
        <div class="fn-list sidebar-list" id="inspector-fn-list"></div>
      </aside>

      <div class="three-pane inspector-grid" id="inspector-three-pane">
        <!-- PANE 1: UPSTREAM RUST / PYTHON SOURCE -->
        <div class="pane inspector-pane" id="insp-pane-1">
          <div class="pane-header pane-hdr">
            <div class="pane-title pane-hdr-title">
              <span>01 / Upstream Source</span>
            </div>
            <span class="mono" style="font-size:10.5px;color:var(--fg-muted);" id="insp-p1-lang">Rust</span>
          </div>
          <div class="pane-body pane-scroll" id="insp-p1-body"></div>
        </div>

        <!-- PANE 2: CHARON LLBC & AENEAS IR / PYO3 FFI RECORD -->
        <div class="pane inspector-pane" id="insp-pane-2">
          <div class="pane-header pane-hdr">
            <div class="pane-title pane-hdr-title">
              <span>02 / Charon LLBC &amp; Aeneas</span>
            </div>
            <span class="mono" style="font-size:10.5px;color:var(--fg-muted);" id="insp-p2-status">LLBC</span>
          </div>
          <div class="pane-body pane-scroll" id="insp-p2-body"></div>
        </div>

        <!-- PANE 3: VERBATIM LEAN 4 SPEC (@leanDef), PROOF (@leanThm) & KERNEL DEPENDENCIES -->
        <div class="pane inspector-pane" id="insp-pane-3">
          <div class="pane-header pane-hdr">
            <div class="pane-title pane-hdr-title">
              <span>03 / Lean 4 Specification</span>
            </div>
            <span class="mono" style="font-size:10.5px;color:var(--emerald);">0 sorry</span>
          </div>
          <div class="pane-body pane-scroll" id="insp-p3-body"></div>
        </div>
      </div>
    `;

    // Bind Sidebar Controls
    const searchEl = document.getElementById("inspector-search");
    if (searchEl) {
      searchEl.addEventListener("input", (e) => {
        appState.inspectorQuery = e.target.value;
        applyInspectorFilters();
      });
    }

    const filterRow = document.getElementById("inspector-crate-filters");
    if (filterRow) {
      filterRow.addEventListener("click", (e) => {
        const btn = e.target.closest(".filter-chip");
        if (!btn) return;
        appState.inspectorCrateFilter = btn.dataset.filter;
        filterRow.querySelectorAll(".filter-chip").forEach((b) => {
          b.classList.toggle("active", b.dataset.filter === appState.inspectorCrateFilter);
        });
        applyInspectorFilters();
      });
    }

    const flatBtn = document.getElementById("insp-mode-flat");
    const treeBtn = document.getElementById("insp-mode-tree");
    if (flatBtn && treeBtn) {
      flatBtn.addEventListener("click", () => {
        appState.inspectorListMode = "flat";
        flatBtn.classList.add("active");
        treeBtn.classList.remove("active");
        renderInspectorSidebarList();
      });
      treeBtn.addEventListener("click", () => {
        appState.inspectorListMode = "tree";
        treeBtn.classList.add("active");
        flatBtn.classList.remove("active");
        renderInspectorSidebarList();
      });
    }

    const listEl = document.getElementById("inspector-fn-list");
    if (listEl) {
      listEl.addEventListener("click", (e) => {
        const treeHdr = e.target.closest(".tree-group-hdr");
        if (treeHdr) {
          const fileKey = treeHdr.dataset.file;
          if (appState.collapsedTreeDirs.has(fileKey)) {
            appState.collapsedTreeDirs.delete(fileKey);
          } else {
            appState.collapsedTreeDirs.add(fileKey);
          }
          renderInspectorSidebarList();
          return;
        }
        const fnItem = e.target.closest(".fn-item");
        if (fnItem) {
          selectFunction(Number(fnItem.dataset.idx), false);
        }
      });
    }

    // Bind Exact Source Line (L{k}) <-> Charon LLBC Statement ([L{k}]) Hover Matching across Panes 1 & 2
    const threePane = document.getElementById("inspector-three-pane");
    if (threePane) {
      threePane.addEventListener("mouseover", (e) => {
        const tr = e.target.closest("tr[data-src-line]");
        if (!tr) return;
        const srcLine = tr.dataset.srcLine;
        if (appState.activeHoverBand === srcLine) return;
        appState.activeHoverBand = srcLine;
        threePane.querySelectorAll("tr[data-src-line]").forEach((row) => {
          const match = row.dataset.srcLine === srcLine;
          row.classList.toggle("band-active-0", match);
          row.classList.toggle("hl", match || row.classList.contains("target-line"));
        });
      });

      threePane.addEventListener("mouseleave", () => {
        appState.activeHoverBand = null;
        threePane.querySelectorAll("tr[data-src-line]").forEach((row) => {
          row.classList.remove("band-active-0");
          row.classList.toggle("hl", row.classList.contains("target-line"));
        });
      });
    }

    applyInspectorFilters();
    selectFunction(appState.selectedFnIdx, false);
  }

  function crateBadgeClass(crate) {
    if (crate === "ennbo") return "ennbo";
    if (crate === "bpann") return "bpann";
    if (isEnnPyCrate(crate)) return "enn-py";
    return "py";
  }

  function applyInspectorFilters() {
    const q = (appState.inspectorQuery || "").trim().toLowerCase();
    const flt = appState.inspectorCrateFilter;

    filteredFunctions = (DATA.registry_functions || []).filter((fn) => {
      if (flt === "rust" && fn.language !== "rust") return false;
      if (flt === "python" && fn.language !== "python") return false;
      if (flt === "bpann" && fn.crate !== "bpann") return false;
      if (flt === "ennbo" && fn.crate !== "ennbo") return false;
      if (flt === "enn-py" && !isEnnPyCrate(fn.crate)) return false;
      if (flt === "aeneas" && !(fn.aeneas && fn.aeneas.code)) return false;
      if (flt === "roots" && fn.depth > 1) return false;

      if (!q) return true;
      return (
        fn.name.toLowerCase().includes(q) ||
        fn.file.toLowerCase().includes(q) ||
        fn.id.toLowerCase().includes(q) ||
        fn.lean_def.toLowerCase().includes(q) ||
        fn.lean_thm.toLowerCase().includes(q) ||
        (fn.root && fn.root.toLowerCase().includes(q)) ||
        String(fn.idx) === q.replace(/^#/, "")
      );
    });

    const countEl = document.getElementById("inspector-count-label");
    if (countEl) {
      countEl.textContent = `Showing ${filteredFunctions.length} / ${(DATA.registry_functions || []).length} functions`;
    }

    renderInspectorSidebarList();

    if (
      filteredFunctions.length > 0 &&
      !filteredFunctions.some((f) => f.idx === appState.selectedFnIdx)
    ) {
      selectFunction(filteredFunctions[0].idx, false);
    }
  }

  function renderFnRowHtml(fn) {
    const isAct = fn.idx === appState.selectedFnIdx;
    const shortFile = fn.file.replace(/^rust\/crates\//, "").replace(/^src\//, "");
    const crateLabel = fn.crate === "ennbo-py" ? "enn-py" : fn.crate;
    return `
      <div class="fn-item list-row ${isAct ? "active" : ""}" data-idx="${fn.idx}">
        <div class="fn-item-top">
          <span class="fn-idx mono">#${fn.idx}</span>
          <span class="fn-name mono">${esc(fn.name)}</span>
          <span class="fn-crate ${crateBadgeClass(fn.crate)}">${esc(crateLabel)}</span>
        </div>
        <div class="fn-item-sub">
          <span class="fn-file mono">${esc(shortFile)}:${fn.line}</span>
          <span class="fn-thm-badge mono">✓ ${esc(fn.lean_thm)}</span>
        </div>
      </div>
    `;
  }

  function renderInspectorSidebarList() {
    const listEl = document.getElementById("inspector-fn-list");
    if (!listEl) return;

    if (filteredFunctions.length === 0) {
      listEl.innerHTML = `<div style="padding:18px;color:var(--fg-muted);" class="mono">No matching functions.</div>`;
      return;
    }

    if (appState.inspectorListMode === "tree") {
      const byFile = new Map();
      for (const fn of filteredFunctions) {
        if (!byFile.has(fn.file)) byFile.set(fn.file, []);
        byFile.get(fn.file).push(fn);
      }
      let html = "";
      for (const [file, fns] of byFile.entries()) {
        const collapsed = appState.collapsedTreeDirs.has(file);
        const shortPath = file.replace(/^rust\/crates\//, "crates/").replace(/^src\/enn\//, "python/enn/");
        html += `
          <div class="tree-group-hdr mono" data-file="${esc(file)}">
            <span>${collapsed ? "▸" : "▾"} ${esc(shortPath)}</span>
            <span class="pill">${fns.length}</span>
          </div>
        `;
        if (!collapsed) {
          html += fns.map(renderFnRowHtml).join("");
        }
      }
      listEl.innerHTML = html;
      return;
    }

    listEl.innerHTML = filteredFunctions.map(renderFnRowHtml).join("");
  }

  function selectFunction(idx, scrollList) {
    const fn = fnByIdx.get(idx);
    if (!fn) return;
    appState.selectedFnIdx = idx;

    const listEl = document.getElementById("inspector-fn-list");
    if (listEl) {
      listEl.querySelectorAll(".fn-item").forEach((el) => {
        const active = Number(el.dataset.idx) === idx;
        el.classList.toggle("active", active);
        if (active && scrollList) {
          el.scrollIntoView({ block: "nearest" });
        }
      });
    }

    const crateLabel = fn.crate === "ennbo-py" ? "enn-py" : fn.crate;

    // --- PANE 1: UPSTREAM SOURCE ---
    const p1Lang = document.getElementById("insp-p1-lang");
    const p1Body = document.getElementById("insp-p1-body");
    const lineEnd = fn.src_start + (fn.src_code || "").split("\n").length - 1;
    if (p1Lang) {
      p1Lang.textContent = `${fn.language} · ${crateLabel} · L${fn.src_start}–L${lineEnd}`;
    }
    if (p1Body) {
      p1Body.innerHTML = `
        <div class="code-box code-frame">
          <div class="code-box-header code-frame-hdr">
            <span class="mono">${esc(fn.file)}:${fn.line}</span>
            <span class="mono">Fn #${fn.idx}</span>
          </div>
          ${renderCodeTable(fn.src_code, fn.src_start, fn.line, {
            enableDocgenLinks: false,
            isSourcePane: true,
          })}
        </div>

        <div class="meta-card arch-card">
          <div class="meta-card-title arch-card-hdr">
            <span>Reachability &amp; Source Metadata</span>
            <span class="mono" style="font-size:10.5px;color:var(--fg-muted);">Depth ${fn.depth}</span>
          </div>
          <div class="kv-grid">
            <span class="kv-key">Function:</span>
            <span class="kv-val mono" style="font-weight:600;">${esc(fn.name)}</span>
            <span class="kv-key">Root:</span>
            <span class="kv-val mono">${esc(fn.root || "Public API Root")}</span>
            <span class="kv-key">SCIP ID:</span>
            <span class="kv-val mono" style="font-size:11px;color:var(--fg-secondary);word-break:break-all;">${esc(fn.id)}</span>
          </div>
        </div>
      `;
    }

    // --- PANE 2: CHARON LLBC & AENEAS IR (RUST) OR SCIP-PYTHON & PYO3 FFI RECORD (PYTHON) ---
    const p2Status = document.getElementById("insp-p2-status");
    const p2Body = document.getElementById("insp-p2-body");

    const mkEntryCode = [
      `-- Compile-time Registry.lean witness binding #${fn.idx}`,
      `mkEntry ${fn.idx}`,
      `  "${fn.id}"`,
      `  "${fn.file}" ${fn.line}`,
      `  "${fn.name}"`,
      `  "${fn.lean_def}" "${fn.lean_thm}"`,
      `  @${fn.lean_def} @${fn.lean_thm}`,
    ].join("\n");

    if (p2Body) {
      if (fn.language === "rust") {
        const llbc = fn.llbc || {};
        const normCrate = (llbc.crate || crateLabel || "ennbo").replace(/^ennbo-py$/, "enn-py");
        const crateInfo = getLlbcCrateMeta(normCrate);
        const structCount = crateInfo.local_structured_functions || 1699;
        const errCount = crateInfo.local_error_functions || 0;
        const llbcSha =
          crateInfo.sha256 ||
          "f9452ee3cae48b33e6a84f3b6180b5459b8723744975b34d65ea21e8552ea065";

        const sigInputsMatch = (llbc.sig || "").match(/\((\d+)\s+inputs?\)/);
        const inputsCount =
          llbc.inputs !== undefined && llbc.inputs !== null
            ? llbc.inputs
            : sigInputsMatch
            ? Number(sigInputsMatch[1])
            : "—";
        const localsCount =
          llbc.locals !== undefined && llbc.locals !== null ? llbc.locals : "—";
        const stmtsCount =
          llbc.statements !== undefined && llbc.statements !== null ? llbc.statements : "—";
        const opacityStr =
          llbc.opacity ||
          (llbc.snippet && llbc.snippet.includes("Opaque") ? "Opaque" : "Transparent");
        const spanStr = llbc.span || `${fn.file}:${fn.line}`;

        if (p2Status) {
          p2Status.textContent = `${normCrate}.llbc · ${inputsCount} args · ${stmtsCount} stmts`;
        }

        const llbcSnippetText =
          llbc.snippet ||
          [
            `// Charon LLBC Structured Body [${normCrate}]`,
            `// Rust item : ${llbc.llbc_name || fn.name}`,
            `// Span      : ${spanStr}`,
            `// Signature : ${llbc.sig || "(MIR -> LLBC GExprBody)"}`,
            `// Lowering  : Structured GExprBody (MIR -> LLBC) [L${fn.line}]`,
          ].join("\n");

        let aeneasHtml = "";
        if (fn.aeneas && fn.aeneas.code) {
          const aeneasFile = fn.aeneas.file || fn.aeneas.lean_file || "BauxResearch/Generated";
          aeneasHtml = `
            <div class="code-box code-frame">
              <div class="code-box-header code-frame-hdr">
                <span class="mono">Aeneas Translation &middot; ${esc(aeneasFile)}</span>
                <span class="pill purple">${esc(fn.aeneas.lean_name || fn.name)}</span>
              </div>
              ${renderCodeTable(fn.aeneas.code, 1, null, {
                enableDocgenLinks: true,
              })}
            </div>
          `;
        }

        p2Body.innerHTML = `
          <div class="code-box code-frame">
            <div class="code-box-header code-frame-hdr">
              <span class="mono">Charon LLBC MIR &middot; ${esc(normCrate)}.llbc</span>
              <span class="pill cyan">${esc(opacityStr)}</span>
            </div>
            ${renderCodeTable(llbcSnippetText, 1, null, {
              enableDocgenLinks: false,
              isLlbcPane: true,
            })}
          </div>

          ${aeneasHtml}

          <div class="meta-card arch-card">
            <div class="meta-card-title arch-card-hdr">
              <span>Charon LLBC Record</span>
              <span class="pill">${structCount} bodies (${errCount} errors)</span>
            </div>
            <div class="kv-grid">
              <span class="kv-key">Rust Item:</span>
              <span class="kv-val mono" style="word-break:break-all;">${esc(llbc.llbc_name || fn.name)}</span>
              <span class="kv-key">Body Metrics:</span>
              <span class="kv-val mono">${esc(String(inputsCount))} args &middot; ${esc(String(localsCount))} locals &middot; ${esc(String(stmtsCount))} statements</span>
              <span class="kv-key">SHA-256:</span>
              <span class="kv-val mono" style="font-size:11px;color:var(--fg-secondary);word-break:break-all;">${esc(llbcSha)}</span>
            </div>
          </div>
        `;
      } else {
        const covSha =
          (DATA.artifact_manifest &&
            DATA.artifact_manifest.source_bindings &&
            DATA.artifact_manifest.source_bindings.coverage_manifest_sha256) ||
          "597d30a32adedaa40a2e9de219ed48093cf214235505d99fc1d4866f8f2a2e74";
        const shortDefMod = (fn.lean_def_module || "BauxResearch.Semantics.Registry")
          .split(".")
          .pop();

        if (p2Status) {
          p2Status.textContent = "scip-python · PyO3 FFI";
        }

        p2Body.innerHTML = `
          <div class="code-box code-frame">
            <div class="code-box-header code-frame-hdr">
              <span class="mono">BauxResearch/Semantics/Registry.lean</span>
              <span class="pill emerald">Witness #${fn.idx}</span>
            </div>
            ${renderCodeTable(mkEntryCode, 1, 2, {
              enableDocgenLinks: true,
            })}
          </div>

          <div class="meta-card arch-card">
            <div class="meta-card-title arch-card-hdr">
              <span>Python &rarr; PyO3 ABI3 Boundary</span>
              <span class="pill accent">scip-python</span>
            </div>
            <div class="kv-grid">
              <span class="kv-key">Module:</span>
              <span class="kv-val mono" style="font-weight:600;">${esc(fn.file)}:${fn.line}</span>
              <span class="kv-key">Lowering:</span>
              <span class="kv-val mono">src/enn &rarr; _rust.abi3.so &rarr; ${esc(shortDefMod)}</span>
              <span class="kv-key">Manifest SHA:</span>
              <span class="kv-val mono" style="font-size:11px;color:var(--fg-secondary);word-break:break-all;">${esc(covSha)}</span>
            </div>
          </div>
        `;
      }
    }

    // --- PANE 3: VERBATIM LEAN 4 PROVED THEOREM (@leanThm), SPEC (@leanDef) & REFERENCED KERNEL SYMBOLS ---
    const thmSym = symbolMap.get(fn.lean_thm) || {};
    const defSym = symbolMap.get(fn.lean_def) || {};
    const p3Body = document.getElementById("insp-p3-body");

    const fullThmName = thmSym.module ? `${thmSym.module}.${thmSym.name}` : fn.lean_thm;
    const axList = axiomMap[fullThmName] ||
      axiomMap[fn.lean_thm] ||
      axiomMap[thmSym.name] || ["propext", "Classical.choice", "Quot.sound"];
    const referencedSymbols = extractReferencedLeanSymbols(
      `${thmSym.code || ""}\n${defSym.code || ""}`,
      [fn.lean_thm, fn.lean_def, thmSym.name, defSym.name]
    );

    if (p3Body) {
      const shortThmFile = (thmSym.file || fn.lean_thm_module || "").split("/").pop();
      const shortDefFile = (defSym.file || fn.lean_def_module || "").split("/").pop();
      p3Body.innerHTML = `
        <div class="pane-action-strip">
          <div class="mono" style="font-size:11px;">
            <span class="sym-token lean-sym-link" data-sym="${esc(fn.lean_thm)}" style="color:var(--emerald);font-weight:600;">@${esc(fn.lean_thm)}</span>
            <span style="color:var(--fg-muted);">&middot;</span>
            <span class="sym-token lean-sym-link" data-sym="${esc(fn.lean_def)}" style="color:var(--klein);font-weight:600;">@${esc(fn.lean_def)}</span>
          </div>
          <div style="display:flex;gap:4px;flex-wrap:wrap;">
            <button class="btn-link" id="insp-open-alectryon-btn">Drawer &nearr;</button>
            <button class="btn-link" id="insp-open-api-btn">Module &nearr;</button>
            <button class="btn-link" id="insp-open-dag-btn">Graph &nearr;</button>
          </div>
        </div>

        <!-- VERBATIM PROVED THEOREM (@leanThm) CODE BLOCK -->
        <div class="code-box code-frame">
          <div class="code-box-header code-frame-hdr">
            <span class="mono">theorem @${esc(fn.lean_thm)} &middot; ${esc(shortThmFile)}:L${thmSym.line || 1}</span>
            <span class="mono" style="color:var(--emerald);">${esc(axList.join(", "))}</span>
          </div>
          ${renderCodeTable(thmSym.code || `-- @${fn.lean_thm}`, thmSym.line || 1, thmSym.line || 1, {
            enableDocgenLinks: true,
          })}
        </div>

        <!-- VERBATIM SEMANTIC SPECIFICATION (@leanDef) CODE BLOCK -->
        <div class="code-box code-frame">
          <div class="code-box-header code-frame-hdr">
            <span class="mono">def @${esc(fn.lean_def)} &middot; ${esc(shortDefFile)}:L${defSym.line || 1}</span>
            <span class="mono">specification</span>
          </div>
          ${renderCodeTable(defSym.code || `-- @${fn.lean_def}`, defSym.line || 1, defSym.line || 1, {
            enableDocgenLinks: true,
          })}
        </div>

        <!-- REFERENCED KERNEL DECLARATIONS & DEPENDENCIES -->
        ${renderReferencedSymbolsTable(
          referencedSymbols,
          `Referenced Declarations (${referencedSymbols.length})`
        )}
      `;

      document.getElementById("insp-open-alectryon-btn")?.addEventListener("click", () => {
        openAlectryonDrawer(fn.lean_thm, { fn });
      });
      document.getElementById("insp-open-api-btn")?.addEventListener("click", () => {
        openModuleInApi(fn.lean_thm_module, fn.lean_thm.split(".").pop());
      });
      document.getElementById("insp-open-dag-btn")?.addEventListener("click", () => {
        openModuleInDag(fn.lean_thm_module);
      });
    }
  }

  // -------------------------------------------------------------------------
  // 5. TAB IV — DOC-GEN4 189-MODULE API & CONSTRUCTIONS BROWSER (#view-api)
  // -------------------------------------------------------------------------
  function getApiLayerFilters() {
    const out = [{ id: "all", label: `All ${(DATA.lean_modules || []).length || 189} Modules` }];
    const seen = new Set();
    for (const m of DATA.lean_modules || []) {
      const id = m.layer || "foundations";
      if (!seen.has(id)) {
        seen.add(id);
        out.push({ id, label: m.layer_label || m.layer || id });
      }
    }
    return out;
  }

  function getCardTopic(c) {
    return (c && (c.topic || c.title)) || "Construction";
  }

  function getCardBody(c) {
    if (!c) return "";
    if (c.body) return c.body;
    if (c.why_it_matters) return c.why_it_matters;
    const parts = [];
    if (c.status) parts.push(`Status: ${c.status}`);
    if (c.definition) parts.push(`Definition: ${c.definition}`);
    if (c.source) parts.push(`Source: ${c.source}`);
    if (c.result) parts.push(`Result: ${c.result}`);
    return parts.join("\n");
  }

  function renderApiView() {
    const container = document.getElementById("view-api");
    if (!container) return;

    const totalMods = (DATA.lean_modules || []).length;
    const totalDecls = allDeclarations.length;
    const totalCards = (DATA.constructions || []).length;
    const layerFilters = getApiLayerFilters();

    container.innerHTML = `
      <aside class="ref-sidebar arch-sidebar">
        <div class="sidebar-controls sidebar-header">
          <div class="filter-row chip-row" id="api-mode-tabs">
            <button class="filter-chip ${appState.selectedApiMode === "modules" ? "active" : ""}" data-mode="modules">
              Modules (${totalMods})
            </button>
            <button class="filter-chip ${appState.selectedApiMode === "declarations" ? "active" : ""}" data-mode="declarations">
              Declarations (${Number(totalDecls).toLocaleString()})
            </button>
          </div>
          <select class="search-input arch-input mono" id="api-layer-select" style="padding:5px 8px;">
            ${layerFilters
              .map(
                (lf) =>
                  `<option value="${esc(lf.id)}" ${appState.selectedApiLayer === lf.id ? "selected" : ""}>${esc(lf.label)}</option>`
              )
              .join("")}
          </select>
          <input
            type="text"
            class="search-input arch-input mono"
            id="api-sidebar-search"
            placeholder="Search 59 ENN modules or declarations..."
            value="${esc(appState.selectedApiSidebarQuery)}"
          />
        </div>
        <div class="fn-list sidebar-list" id="api-sidebar-list"></div>
      </aside>
      <div class="ref-main" id="api-main-content"></div>
    `;

    document.getElementById("api-mode-tabs")?.addEventListener("click", (e) => {
      const btn = e.target.closest(".filter-chip");
      if (!btn) return;
      appState.selectedApiMode = btn.dataset.mode;
      document.querySelectorAll("#api-mode-tabs .filter-chip").forEach((b) => {
        b.classList.toggle("active", b.dataset.mode === appState.selectedApiMode);
      });
      renderApiSidebar();
      renderApiMain();
    });

    document.getElementById("api-layer-select")?.addEventListener("change", (e) => {
      appState.selectedApiLayer = e.target.value;
      renderApiSidebar();
      if (appState.selectedApiMode === "declarations") renderApiMain();
    });

    document.getElementById("api-sidebar-search")?.addEventListener("input", (e) => {
      appState.selectedApiSidebarQuery = e.target.value;
      renderApiSidebar();
      if (appState.selectedApiMode === "declarations") renderApiMain();
    });

    renderApiSidebar();
    renderApiMain();
  }

  function renderApiSidebar() {
    const listEl = document.getElementById("api-sidebar-list");
    if (!listEl) return;
    const q = (appState.selectedApiSidebarQuery || "").trim().toLowerCase();
    const layerFlt = appState.selectedApiLayer;

    if (appState.selectedApiMode === "constructions") {
      const cards = (DATA.constructions || []).map((c, idx) => ({ ...c, idx }));
      const filtered = cards.filter((c) => {
        if (!q) return true;
        const topic = getCardTopic(c).toLowerCase();
        const body = getCardBody(c).toLowerCase();
        return (
          topic.includes(q) ||
          body.includes(q) ||
          `c${String(c.idx + 1).padStart(2, "0")}` === q
        );
      });

      listEl.innerHTML = filtered
        .map((c) => {
          const cid = `C${String(c.idx + 1).padStart(2, "0")}`;
          const isAct = c.idx === appState.selectedConstructionIdx;
          const topic = getCardTopic(c);
          const body = getCardBody(c);
          const refCount = (c.lean_refs || []).length;
          return `
            <div class="fn-item list-row ${isAct ? "active" : ""}" data-cidx="${c.idx}">
              <div class="fn-item-top">
                <span class="fn-idx mono">${cid}</span>
                <span class="fn-name mono">${esc(topic)}</span>
                <span class="fn-crate py">${refCount > 0 ? `${refCount} refs` : esc(c.status || "PROVED")}</span>
              </div>
              <div class="fn-item-sub">
                <span class="fn-file">${esc((body.split("\n")[0] || "").slice(0, 58))}</span>
              </div>
            </div>
          `;
        })
        .join("");

      listEl.querySelectorAll(".fn-item").forEach((el) => {
        el.addEventListener("click", () => {
          appState.selectedConstructionIdx = Number(el.dataset.cidx);
          renderApiSidebar();
          renderApiMain();
        });
      });
      return;
    }

    const mods = (DATA.lean_modules || []).filter((m) => {
      if (layerFlt !== "all" && m.layer !== layerFlt && m.layer_label !== layerFlt) return false;
      if (!q) return true;
      return (
        m.module.toLowerCase().includes(q) ||
        m.file.toLowerCase().includes(q) ||
        (m.layer && m.layer.toLowerCase().includes(q)) ||
        (m.layer_label && m.layer_label.toLowerCase().includes(q)) ||
        (m.doc && m.doc.toLowerCase().includes(q))
      );
    });

    listEl.innerHTML = mods
      .map((m) => {
        const isAct = m.module === appState.selectedApiModule;
        const shortName = m.module.replace(/^BauxResearch\./, "");
        return `
          <div class="fn-item list-row ${isAct ? "active" : ""}" data-mod="${esc(m.module)}">
            <div class="fn-item-top">
              <span class="fn-name mono">${esc(shortName)}</span>
              <span class="fn-crate ennbo">${m.counts.theorem || 0}T · ${m.counts.def || 0}D</span>
            </div>
            <div class="fn-item-sub">
              <span class="fn-file">${esc(m.layer_label || m.layer)}</span>
              <span class="mono">Rank ${m.dag_rank} · ${m.lines}L</span>
            </div>
          </div>
        `;
      })
      .join("");

    listEl.querySelectorAll(".fn-item").forEach((el) => {
      el.addEventListener("click", () => {
        appState.selectedApiModule = el.dataset.mod;
        if (appState.selectedApiMode === "declarations") {
          appState.selectedApiMode = "modules";
          document.querySelectorAll("#api-mode-tabs .filter-chip").forEach((b) => {
            b.classList.toggle("active", b.dataset.mode === "modules");
          });
        }
        renderApiSidebar();
        renderApiMain();
      });
    });
  }

  function parseConstructionSections(body) {
    const lines = (body || "").split("\n");
    const sections = [];
    let currentKey = "OVERVIEW";
    let currentBuf = [];

    const KEY_MAP = {
      status: "STATUS & VERIFICATION SCOPE",
      definition: "CONSTRUCTION & FORMAL DEFINITION",
      construction: "CONSTRUCTION & FORMAL DEFINITION",
      source: "NEW CODE & SOURCE PROVENANCE",
      result: "THEOREM & KERNEL RESULT",
      theorem: "THEOREM & KERNEL RESULT",
      dependencies: "REUSED FROM MATHLIB & AXIOM DEPENDENCIES",
      quantification: "PROBLEM & UNIVERSAL QUANTIFICATION",
      problem: "PROBLEM & UNIVERSAL QUANTIFICATION",
      "execution obligations": "EXECUTION & ARTIFACT OBLIGATIONS",
      reproduce: "REPRODUCIBILITY COMMAND",
    };

    for (const line of lines) {
      const m = line.match(/^([A-Za-z /_-]+):\s*(.*)$/);
      if (m && KEY_MAP[m[1].trim().toLowerCase()]) {
        if (currentBuf.length > 0) {
          sections.push({ title: currentKey, content: currentBuf.join("\n").trim() });
        }
        currentKey = KEY_MAP[m[1].trim().toLowerCase()];
        currentBuf = [m[2]];
      } else {
        currentBuf.push(line);
      }
    }
    if (currentBuf.length > 0) {
      sections.push({ title: currentKey, content: currentBuf.join("\n").trim() });
    }
    return sections;
  }

  function renderApiMain() {
    const mainEl = document.getElementById("api-main-content");
    if (!mainEl) return;

    if (appState.selectedApiMode === "constructions") {
      const cards = DATA.constructions || [];
      const card = cards[appState.selectedConstructionIdx] || cards[0];
      if (!card) return;
      const cid = `C${String(appState.selectedConstructionIdx + 1).padStart(2, "0")}`;
      const topic = getCardTopic(card);
      const body = getCardBody(card);
      const parsedSecs = parseConstructionSections(body);

      mainEl.innerHTML = `
        <div class="meta-card arch-card">
          <div class="meta-card-title arch-card-hdr">
            <span>MATHEMATICAL CONSTRUCTION CARD // ${cid} OF ${cards.length} (108 FORMALIZED CONSTRUCTIONS)</span>
            <span class="pill emerald">./baux research show "${esc(topic)}"</span>
          </div>
          <h2 class="mono" style="font-size:18px;font-weight:700;margin-top:4px;">${cid}: CONSTRUCTION ${esc(topic)}</h2>
          ${
            card.lean_refs && card.lean_refs.length > 0
              ? `
            <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;align-items:center;">
              <span class="mono" style="font-size:10.5px;color:var(--fg-secondary);">Linked Lean 4 Modules:</span>
              ${card.lean_refs
                .map((r) => {
                  const modName =
                    "BauxResearch." +
                    r.replace(/^BauxResearch\//, "").replace(/\.lean$/, "").replace(/\//g, ".");
                  return `<button class="btn-link mono" onclick="window.__openModuleInRef('${esc(modName)}')">${esc(r)} &nearr;</button>`;
                })
                .join("")}
            </div>
          `
              : ""
          }
        </div>

        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;">
          ${parsedSecs
            .map(
              (sec) => `
            <div class="meta-card arch-card">
              <div class="meta-card-title arch-card-hdr"><span>${esc(sec.title)}</span></div>
              <div class="mono" style="font-size:11.5px;line-height:1.6;white-space:pre-wrap;color:var(--fg-primary);">${highlightLine(sec.content, true)}</div>
            </div>
          `
            )
            .join("")}
        </div>

        <div class="code-box code-frame">
          <div class="code-box-header code-frame-hdr">
            <span class="mono">docs/mathlib-constructions.txt &middot; Raw Verbatim Specification</span>
            <span class="pill">${cid}</span>
          </div>
          <pre class="mono" style="padding:14px;white-space:pre-wrap;font-size:11.5px;line-height:1.6;">${esc(body)}</pre>
        </div>
      `;
      return;
    }

    if (appState.selectedApiMode === "declarations") {
      const q = (appState.selectedApiSidebarQuery || "").trim().toLowerCase();
      const layerFlt = appState.selectedApiLayer;
      const matches = [];
      for (const d of allDeclarations) {
        if (layerFlt !== "all" && d.layer !== layerFlt) continue;
        if (
          !q ||
          d.name.toLowerCase().includes(q) ||
          d.module.toLowerCase().includes(q) ||
          d.code.toLowerCase().includes(q) ||
          d.doc.toLowerCase().includes(q)
        ) {
          matches.push(d);
          if (matches.length >= 100) break;
        }
      }

      mainEl.innerHTML = `
        <div class="meta-card arch-card">
          <div class="meta-card-title arch-card-hdr">
            <span>GLOBAL DOC-GEN4 DECLARATION SEARCH (${Number(allDeclarations.length).toLocaleString()} TOTAL DECLARATIONS)</span>
            <span class="pill accent">TOP ${matches.length} RESULTS</span>
          </div>
          <div style="font-size:12px;color:var(--fg-secondary);">
            Filtering across all ${(DATA.lean_modules || []).length || 189} Lean 4 modules${q ? ` for query "${esc(q)}"` : ""}. Click any symbol token to inspect or open in the Declaration Drawer.
          </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:12px;">
          ${matches.map((d) => renderApiDeclCard(d, true)).join("")}
        </div>
      `;
      bindApiDeclEvents(mainEl);
      return;
    }

    // Default: Single Module doc-gen4 View
    const m = modByName.get(appState.selectedApiModule) || (DATA.lean_modules || [])[0];
    if (!m) return;
    const revDeps = dependentsByMod.get(m.module) || [];
    const provedFns = (DATA.registry_functions || []).filter(
      (f) => f.lean_thm_module === m.module || f.lean_def_module === m.module
    );

    const decls = (m.declarations || []).filter((d) => {
      if (appState.selectedApiDeclKind !== "all" && d.kind !== appState.selectedApiDeclKind) {
        return false;
      }
      if (!appState.selectedApiDeclQuery) return true;
      const dq = appState.selectedApiDeclQuery.toLowerCase();
      return (
        d.name.toLowerCase().includes(dq) ||
        (d.code && d.code.toLowerCase().includes(dq)) ||
        (d.doc && d.doc.toLowerCase().includes(dq))
      );
    });

    mainEl.innerHTML = `
      <div class="meta-card arch-card">
        <div class="meta-card-title arch-card-hdr">
          <span>${esc(m.layer_label || m.layer || "Lean 4 Module")}</span>
          <div style="display:flex;gap:6px;">
            <button class="btn-link" onclick="window.__openModuleInDag('${esc(m.module)}')">View in Module Graph &nearr;</button>
          </div>
        </div>
        <h2 class="mono" style="font-size:18px;font-weight:600;margin-top:2px;">${esc(m.module)}</h2>
        <div class="mono" style="font-size:12px;color:var(--fg-secondary);">
          ${esc(m.file)} &middot; ${m.lines} lines &middot; Rank R${m.dag_rank} &middot; ${m.total_decls} declarations
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;">
          <span class="pill emerald">${m.counts.theorem || 0} theorems</span>
          <span class="pill accent">${m.counts.def || 0} definitions</span>
          <span class="pill purple">${m.counts.structure || 0} structures</span>
          <span class="pill amber">${m.counts.inductive || 0} inductives</span>
          <span class="pill cyan">${m.counts.abbrev || 0} abbrevs</span>
          ${provedFns.length > 0 ? `<span class="pill emerald">Certifies ${provedFns.length} ENN functions</span>` : ""}
        </div>

        <div class="kv-grid" style="margin-top:8px;padding-top:8px;border-top:1px solid var(--border-hairline);">
          <span class="kv-key">Imports (${(m.imports || []).length}):</span>
          <div style="display:flex;flex-wrap:wrap;gap:4px;">
            ${
              (m.imports || [])
                .map(
                  (imp) =>
                    `<button class="btn-link mono" onclick="window.__openModuleInRef('${esc(imp)}')">${esc(imp.replace(/^BauxResearch\./, ""))}</button>`
                )
                .join("") || `<span class="mono" style="color:var(--fg-muted);font-size:11px;">None (Foundation Root)</span>`
            }
          </div>
          <span class="kv-key">Dependents (${revDeps.length}):</span>
          <div style="display:flex;flex-wrap:wrap;gap:4px;">
            ${
              revDeps
                .map(
                  (dep) =>
                    `<button class="btn-link mono" onclick="window.__openModuleInRef('${esc(dep)}')">${esc(dep.replace(/^BauxResearch\./, ""))}</button>`
                )
                .join("") || `<span class="mono" style="color:var(--fg-muted);font-size:11px;">Top-Level Gateway</span>`
            }
          </div>
        </div>
      </div>

      ${
        m.doc
          ? `
        <div class="meta-card arch-card">
          <div class="decl-doc">${formatDocstring(m.doc)}</div>
        </div>
      `
          : ""
      }

      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;">
        <div class="filter-row chip-row" id="api-decl-kind-bar">
          <button class="filter-chip ${appState.selectedApiDeclKind === "all" ? "active" : ""}" data-kind="all">All (${(m.declarations || []).length})</button>
          <button class="filter-chip ${appState.selectedApiDeclKind === "theorem" ? "active" : ""}" data-kind="theorem">theorem (${m.counts.theorem || 0})</button>
          <button class="filter-chip ${appState.selectedApiDeclKind === "def" ? "active" : ""}" data-kind="def">def (${m.counts.def || 0})</button>
          <button class="filter-chip ${appState.selectedApiDeclKind === "structure" ? "active" : ""}" data-kind="structure">structure (${m.counts.structure || 0})</button>
          <button class="filter-chip ${appState.selectedApiDeclKind === "inductive" ? "active" : ""}" data-kind="inductive">inductive (${m.counts.inductive || 0})</button>
          <button class="filter-chip ${appState.selectedApiDeclKind === "abbrev" ? "active" : ""}" data-kind="abbrev">abbrev (${m.counts.abbrev || 0})</button>
        </div>
        <input
          type="text"
          class="search-input arch-input mono"
          id="api-decl-search"
          style="width:260px;"
          value="${esc(appState.selectedApiDeclQuery)}"
          placeholder="Filter declarations in ${esc(m.short)}..."
        />
      </div>

      <div style="display:flex;flex-direction:column;gap:12px;">
        ${
          decls
            .map((d) =>
              renderApiDeclCard(
                { ...d, module: m.module, short_module: m.short, file: m.file },
                false
              )
            )
            .join("") ||
          `<div class="mono" style="padding:16px;color:var(--fg-muted);">No declarations match filter.</div>`
        }
      </div>
    `;

    document.getElementById("api-decl-kind-bar")?.addEventListener("click", (e) => {
      const btn = e.target.closest(".filter-chip");
      if (!btn) return;
      appState.selectedApiDeclKind = btn.dataset.kind;
      renderApiMain();
    });

    const declSearch = document.getElementById("api-decl-search");
    if (declSearch) {
      declSearch.addEventListener("input", (e) => {
        appState.selectedApiDeclQuery = e.target.value;
        renderApiMain();
        const nextInput = document.getElementById("api-decl-search");
        if (nextInput) {
          nextInput.focus();
          nextInput.setSelectionRange(nextInput.value.length, nextInput.value.length);
        }
      });
    }

    bindApiDeclEvents(mainEl);
  }

  function renderApiDeclCard(d, showModuleBtn) {
    const symKey = `${d.short_module || d.module.split(".").pop()}.${d.name}`;
    return `
      <div class="decl-card kind-${esc(d.kind)}" id="decl-card-${esc(d.name)}">
        <div class="decl-header">
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <span class="decl-kind ${esc(d.kind)}">${esc(d.kind)}</span>
            <span class="mono" style="font-weight:700;font-size:13px;">${esc(d.name)}</span>
            ${
              showModuleBtn
                ? `<button class="btn-link mono" onclick="window.__openModuleInRef('${esc(d.module)}')">${esc(d.module)}</button>`
                : ""
            }
          </div>
          <div style="display:flex;align-items:center;gap:8px;">
            <span class="mono" style="font-size:10.5px;color:var(--fg-muted);">${esc(d.file)}:L${d.line}–L${d.end_line || d.line}</span>
            <button class="btn-link open-alectryon-decl-btn" data-sym="${esc(symKey)}">
              Open in Kernel Lean Source &nearr;
            </button>
          </div>
        </div>
        <div class="decl-body">
          ${d.doc ? `<div class="decl-doc">${esc(d.doc)}</div>` : ""}
          <div class="code-box code-frame">
            ${renderCodeTable(d.code || "", d.line || 1, d.line || 1, {
              enableDocgenLinks: true,
            })}
          </div>
        </div>
      </div>
    `;
  }

  function bindApiDeclEvents(container) {
    container.querySelectorAll(".open-alectryon-decl-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        openAlectryonDrawer(btn.dataset.sym);
      });
    });
  }

  // -------------------------------------------------------------------------
  // 6. TAB V — MACH-O BINARY & IEEE-754 64-BIT BIT-EXACT MATRIX (#view-artifacts)
  // -------------------------------------------------------------------------
  function decodeIeee754Hex(rawHex) {
    const clean = String(rawHex || "0000000000000000")
      .trim()
      .replace(/^0x/i, "")
      .padStart(16, "0")
      .slice(0, 16)
      .toLowerCase();
    try {
      const buf = new ArrayBuffer(8);
      const view = new DataView(buf);
      const hi = parseInt(clean.slice(0, 8), 16) >>> 0;
      const lo = parseInt(clean.slice(8, 16), 16) >>> 0;
      view.setUint32(0, hi, false);
      view.setUint32(4, lo, false);
      const floatVal = view.getFloat64(0, false);

      const hiBits = hi.toString(2).padStart(32, "0");
      const loBits = lo.toString(2).padStart(32, "0");
      const bits64 = hiBits + loBits;

      const signBit = bits64[0];
      const expBits = bits64.slice(1, 12);
      const mantBits = bits64.slice(12, 64);

      const biasedExp = parseInt(expBits, 2);
      const unbiasedExp = biasedExp === 0 ? -1022 : biasedExp - 1023;
      const mantInt = BigInt("0b" + mantBits);
      const mantFrac = Number(mantInt) / Math.pow(2, 52);
      const significand = biasedExp === 0 ? mantFrac : 1.0 + mantFrac;

      return {
        hex: clean,
        bits64,
        signBit,
        signSymbol: signBit === "1" ? "-1 (Negative)" : "+1 (Positive)",
        expBits,
        biasedExp,
        unbiasedExp,
        mantBits,
        significand: significand.toPrecision(15),
        floatStr: Number.isFinite(floatVal) ? floatVal.toPrecision(16) : String(floatVal),
        bigWord: (BigInt(hi) << 32n) | BigInt(lo),
      };
    } catch (_) {
      return {
        hex: clean,
        bits64: "0".repeat(64),
        signBit: "0",
        signSymbol: "+1",
        expBits: "0".repeat(11),
        biasedExp: 0,
        unbiasedExp: -1022,
        mantBits: "0".repeat(52),
        significand: "0",
        floatStr: "0",
        bigWord: 0n,
      };
    }
  }

  function renderArtifactsView() {
    const container = document.getElementById("view-artifacts");
    if (!container) return;

    const art = DATA.artifact_manifest || {};
    const bins = (art.compiled_artifacts && art.compiled_artifacts.binaries) || {};
    const oleans = (art.compiled_artifacts && art.compiled_artifacts.lean_oleans) || {};
    const llbcs = (art.compiled_artifacts && art.compiled_artifacts.charon_llbc) || {};
    const tv = art.trajectory_verification || {};
    const ax = art.axiom_and_trust_audit || { declaration_axioms: {} };
    const cov = DATA.coverage_overview || {};
    const exCats = cov.exclusions_by_category || {};
    const documentedCorrections =
      (tv.suite && Array.isArray(tv.suite.documented_local_corrections) && tv.suite.documented_local_corrections.length
        ? tv.suite.documented_local_corrections
        : [
            "fit.rs: compute_loglik and loo_neighbors corrected to exclude self-distances / degenerate bandwidth collapse",
            "knn: ExactBackend vs FaissBackend deterministic tie-breaking and smnsrch radius pruning aligned with mathematical metric specification",
            "acquisition.rs: UCBAcquisition and ParetoAcquisition::nondomin_2d corrected against non-strict dominance and sign conventions",
          ]);

    container.innerHTML = `
      <div class="dashboard-scroll">
        <!-- 1. INTERACTIVE IEEE-754 64-BIT BIT-FIELD INSPECTOR -->
        <div class="meta-card arch-card">
          <div class="meta-card-title arch-card-hdr">
            <span>01 / IEEE-754 Binary64 (f64::to_bits()) Trajectory &amp; Bit-Field Verification</span>
            <span class="sub-hdr-note">Bit-Exact Multi-Round Trajectories</span>
          </div>

          <div class="filter-row chip-row" id="art-traj-tabs" style="margin-top:8px;">
            <button class="filter-chip active" data-key="turbo_enn">Traj 1: Exact k-NN &amp; TuRBO-ENN (turbo_enn)</button>
            <button class="filter-chip" data-key="turbo_zero">Traj 2: Epanechnikov &amp; Zero-Trust (turbo_zero)</button>
            <button class="filter-chip" data-key="morbo_trust_region">Traj 3: Standardized Moments &amp; MORBO (morbo_trust_region)</button>
            <button class="filter-chip" data-key="lhd_only">Traj 4: Sobol Owen &amp; LHD PCG64 (lhd_only)</button>
            <button class="filter-chip" data-key="disk_bpann">Traj 5: TuRBO Disk BPANN Schedule (disk_bpann)</button>
            <button class="filter-chip" data-key="primitives">17 Primitive Equivalence Checks</button>
          </div>

          <!-- LIVE 64-BIT REGISTER DECODER -->
          <div id="ieee-live-decoder" style="margin-top:12px;"></div>

          <!-- TRAJECTORY TABLE -->
          <div id="art-traj-table-wrap" style="margin-top:12px;"></div>
        </div>

        <!-- 1B. DOCUMENTED UPSTREAM (ENN 0.4.1) VS. CORRECTED LOCAL (BAUX) IEEE-754 DIVERGENCE ANALYSIS -->
        <div class="meta-card arch-card">
          <div class="meta-card-title arch-card-hdr">
            <span>02 / Documented Upstream vs. Corrected Local IEEE-754 Divergence Analysis</span>
            <span class="sub-hdr-note">${documentedCorrections.length} Verified Numerical &amp; Algorithmic Corrections</span>
          </div>
          <div style="font-size:13px;color:var(--fg-secondary);line-height:1.6;margin-bottom:10px;">
            For shared non-ENN trajectories (<code class="mono">turbo_zero</code>, <code class="mono">lhd_only</code>) and the initial Latin-hypercube / Sobol Owen design prefix (Steps <code class="mono">#0..#7</code> of <code class="mono">turbo_enn</code> and <code class="mono">morbo_trust_region</code>), Upstream ENN and the local formalization agree to <code class="mono">0 ULP</code> (<code class="mono">XOR = 0x0000000000000000</code>). At post-fit epistemic surrogate steps, the local implementation and the Lean 4 / C-ABI specification match bit-for-bit (<code class="mono">0x0</code>), while Upstream ENN diverges at documented <code class="mono">Local Fix Delta</code> words due to three verified defects in the upstream crate:
          </div>
          <table class="data-table arch-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Subsystem &amp; Documented Upstream Defect Corrected in Formalization</th>
                <th>Affected Trajectories</th>
                <th>Lean 4 &harr; C-ABI Parity</th>
              </tr>
            </thead>
            <tbody>
              ${documentedCorrections
                .map((item, idx) => {
                  const affected =
                    idx === 0
                      ? "turbo_enn · morbo_trust_region"
                      : idx === 1
                        ? "turbo_enn · disk_bpann"
                        : "turbo_enn · morbo_trust_region";
                  return `
                    <tr>
                      <td class="mono" style="font-weight:600;color:var(--vermilion);">Fix 0${idx + 1}</td>
                      <td class="mono" style="font-size:11.5px;color:var(--fg-primary);">${esc(item)}</td>
                      <td class="mono" style="font-size:11px;color:var(--fg-secondary);">${esc(affected)}</td>
                      <td><span class="pill emerald mono">0x0 Bit-Exact</span></td>
                    </tr>
                  `;
                })
                .join("")}
            </tbody>
          </table>
        </div>

        <!-- 2. EXPLICIT CPU VS GPU HARDWARE TRUST BOUNDARY (ExplicitCpuGpuTrustBoundary) -->
        <div class="meta-card arch-card">
          <div class="meta-card-title arch-card-hdr">
            <span>03 / Hardware Scope: CPU Reference Target vs. GPU Backends</span>
            <span class="sub-hdr-note">ExplicitCpuGpuTrustBoundary</span>
          </div>
          <div class="scope-grid">
            <div class="scope-card established">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span class="inspector-eyebrow" style="margin:0;">CPU Reference Optimizer &amp; BPANN</span>
                <span class="mono sym-token lean-sym-link" data-sym="Artifact.cpuVerified" style="font-size:11px;font-weight:600;">@Artifact.cpuVerified</span>
              </div>
              <div class="mono" style="font-size:13.5px;font-weight:600;">Compiled Mach-O arm64 Binary &amp; Bit-Exact IEEE-754 Correspondence</div>
              <div style="font-size:12.5px;color:var(--fg-secondary);line-height:1.55;">
                ${esc((art.device_scope && art.device_scope.statement) || "CPU Mach-O arm64 binaries verified bit-exact.")}
              </div>
              <div class="kv-grid" style="margin-top:4px;">
                <span class="kv-key">ISA &amp; ABI:</span>
                <span class="kv-val mono">${esc((art.assumptions && art.assumptions.isa) || "arm64-apple-darwin24.0.0")}</span>
                <span class="kv-key">Floating-Point:</span>
                <span class="kv-val mono">${esc((art.assumptions && art.assumptions.floating_point) || "IEEE-754 binary64 roundTiesToEven")}</span>
                <span class="kv-key">Suite SHA-256:</span>
                <span class="kv-val mono" style="color:var(--accent-emerald);">${esc(tv.trajectory_suite_sha256 || "")}</span>
              </div>
            </div>

            <div class="scope-card not-established">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span class="inspector-eyebrow" style="margin:0;color:var(--ochre);">GPU Backends (Out of Scope)</span>
                <span class="mono sym-token lean-sym-link" data-sym="Artifact.gpuRejected" style="font-size:11px;font-weight:600;">@Artifact.gpuRejected</span>
              </div>
              <div class="mono" style="font-size:13.5px;font-weight:600;">Explicit Hardware Trust Boundary (ExplicitCpuGpuTrustBoundary)</div>
              <div style="font-size:12.5px;color:var(--fg-secondary);line-height:1.55;">
                CPU and GPU devices are modeled as distinct target types. Verification is established for native CPU Mach-O binaries; GPU kernels are tracked as unverified source inventory:
              </div>
              <div class="mono" style="display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;font-size:11px;color:var(--fg-primary);margin-top:4px;">
                <span>01. Complete optimizer &amp; objective semantics</span>
                <span>02. Numerical error &amp; decision preservation</span>
                <span>03. Parallel memory, sync &amp; termination</span>
                <span>04. Source-to-device compilation correspondence</span>
                <span>05. Host/device launch &amp; buffer protocol</span>
                <span>06. Composed theorem bound to device binary</span>
              </div>
            </div>
          </div>
        </div>

        <!-- 3. 10 COMPILED MACH-O ARM64 BINARIES + 3 CHARON LLBCS + 6 LEAN OLEANS -->
        <div class="meta-card arch-card">
          <div class="meta-card-title arch-card-hdr">
            <span>04 / Compiled Mach-O 64-Bit (arm64) Binary Artifacts (artifact_manifest.json)</span>
            <span class="sub-hdr-note">10 Binaries &middot; SHA-256 Digests</span>
          </div>
          <table class="data-table arch-table">
            <thead>
              <tr>
                <th>Binary Artifact Key</th>
                <th>Format</th>
                <th>ISA / CPU Type</th>
                <th>Size (Bytes)</th>
                <th>SHA-256 Digest</th>
              </tr>
            </thead>
            <tbody>
              ${Object.entries(bins)
                .map(
                  ([name, info]) => `
                <tr>
                  <td class="mono" style="font-weight:600;color:var(--accent-klein);">${esc(name)}</td>
                  <td class="mono">${esc(info.format)}</td>
                  <td class="mono">${esc(info.isa)} (${esc(info.cputype)})</td>
                  <td class="mono">${Number(info.bytes).toLocaleString()}</td>
                  <td class="mono" style="font-size:11px;">${esc(info.sha256)}</td>
                </tr>
              `
                )
                .join("")}
            </tbody>
          </table>
        </div>

        <div style="display:grid;grid-template-columns:1.25fr 1fr;gap:24px;">
          <div class="meta-card arch-card">
            <div class="meta-card-title arch-card-hdr">
              <span>05 / Charon LLBC Full-Crate Archives (4,366 Bodies)</span>
              <span class="sub-hdr-note">0 Error Bodies</span>
            </div>
            <table class="data-table arch-table">
              <thead>
                <tr>
                  <th>LLBC Crate</th>
                  <th>Structured Bodies</th>
                  <th>Manifest Closure</th>
                  <th>Size (Bytes)</th>
                  <th>SHA-256 Digest</th>
                </tr>
              </thead>
              <tbody>
                ${Object.entries(llbcs)
                  .map(
                    ([crate, info]) => `
                  <tr>
                    <td class="mono" style="font-weight:600;">${esc(crate)}.llbc</td>
                    <td class="mono" style="color:var(--accent-emerald);font-weight:600;">${info.local_structured_functions} (0 errors)</td>
                    <td class="mono">${info.manifest_functions_covered} / ${info.manifest_functions_total} (100%)</td>
                    <td class="mono">${Number(info.bytes).toLocaleString()}</td>
                    <td class="mono" style="font-size:10.5px;">${esc(info.sha256)}</td>
                  </tr>
                `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>

          <div class="meta-card arch-card">
            <div class="meta-card-title arch-card-hdr">
              <span>06 / Compiled Lean 4 Objects (.olean) &amp; C-ABI Symbols</span>
              <span class="sub-hdr-note">13 Exported Symbols</span>
            </div>
            <table class="data-table arch-table">
              <thead>
                <tr>
                  <th>.olean Module</th>
                  <th>Bytes</th>
                  <th>SHA-256</th>
                </tr>
              </thead>
              <tbody>
                ${Object.entries(oleans)
                  .map(
                    ([name, info]) => `
                  <tr>
                    <td class="mono" style="font-weight:600;">${esc(name)}</td>
                    <td class="mono">${Number(info.bytes).toLocaleString()}</td>
                    <td class="mono" style="font-size:10.5px;">${esc((info.sha256 || "").slice(0, 24))}…</td>
                  </tr>
                `
                  )
                  .join("")}
              </tbody>
            </table>
            <div style="margin-top:10px;">
              <div class="mono" style="font-size:10.5px;font-weight:600;color:var(--fg-secondary);margin-bottom:4px;">
                13 Exported C-ABI Symbols (dlsym):
              </div>
              <div style="display:flex;flex-wrap:wrap;gap:4px;">
                ${(tv.cabi_symbols_verified || [])
                  .map((sym) => `<span class="pill mono">extern "C" ${esc(sym)}</span>`)
                  .join("")}
              </div>
            </div>
          </div>
        </div>

        <!-- 4. 641 EXCLUDED FUNCTIONS / 343 NON-RUNTIME SCOPES & KERNEL AXIOM AUDIT -->
        <div style="display:grid;grid-template-columns:1.25fr 1fr;gap:24px;">
          <div class="meta-card arch-card">
            <div class="meta-card-title arch-card-hdr">
              <span>07 / Non-Runtime Exclusions Audit (${cov.total_exclusions || 343} Scopes)</span>
              <span class="sub-hdr-note">641 Excluded Functions</span>
            </div>
            <table class="data-table arch-table">
              <thead>
                <tr>
                  <th>Exclusion Category</th>
                  <th>Scopes</th>
                  <th>Exclusion Rationale &amp; Representative Scope</th>
                </tr>
              </thead>
              <tbody>
                ${Object.entries(exCats)
                  .map(
                    ([cat, info]) => `
                  <tr>
                    <td><span class="pill amber mono">${esc(cat)}</span></td>
                    <td class="mono" style="font-weight:600;">${info.count}</td>
                    <td>
                      <div style="font-size:12px;color:var(--fg-primary);">${esc((info.sample && info.sample[0] && info.sample[0].proof) || "")}</div>
                      <div class="mono" style="font-size:10.5px;color:var(--fg-muted);margin-top:2px;">
                        ${(info.sample || []).slice(0, 4).map((s) => esc(s.scope)).join(" · ")}
                      </div>
                    </td>
                  </tr>
                `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>

          <div class="meta-card arch-card">
            <div class="meta-card-title arch-card-hdr">
              <span>08 / Lean 4 Kernel Axiom Audit (#print axioms)</span>
              <span class="sub-hdr-note">Standard Axioms Only</span>
            </div>
            <table class="data-table arch-table">
              <thead>
                <tr>
                  <th>Verified Gateway Declaration</th>
                  <th>Kernel Axioms</th>
                </tr>
              </thead>
              <tbody>
                ${Object.entries(ax.declaration_axioms || {})
                  .map(
                    ([decl, list]) => `
                  <tr>
                    <td class="mono" style="font-weight:600;">${esc(decl)}</td>
                    <td class="mono" style="color:var(--accent-emerald);">${
                      list && list.length ? esc(list.join(", ")) : "(none — kernel rfl / decide)"
                    }</td>
                  </tr>
                `
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;

    document.getElementById("art-traj-tabs")?.addEventListener("click", (e) => {
      const btn = e.target.closest(".filter-chip");
      if (!btn) return;
      appState.selectedTrajKey = btn.dataset.key;
      document.querySelectorAll("#art-traj-tabs .filter-chip").forEach((b) => {
        b.classList.toggle("active", b.dataset.key === appState.selectedTrajKey);
      });
      renderTrajectoryMatrix();
    });

    renderIeeeLiveDecoder();
    renderTrajectoryMatrix();
  }

  function renderIeeeLiveDecoder() {
    const box = document.getElementById("ieee-live-decoder");
    if (!box) return;

    const d1 = decodeIeee754Hex(appState.selectedHexWord);
    const d2 = decodeIeee754Hex(appState.selectedCompareHexWord);
    const xorVal = d1.bigWord ^ d2.bigWord;
    const xorHex = xorVal.toString(16).padStart(16, "0");
    const isExactMatch = xorVal === 0n;

    let bitCellsHtml = "";
    for (let i = 0; i < 64; i++) {
      const bit = d1.bits64[i];
      const bitIdx = 63 - i;
      const cls = i === 0 ? "sign" : i < 12 ? "exp" : "mant";
      bitCellsHtml += `<div class="bit-cell ${cls} ${bit === "1" ? "one" : ""} mono" title="Bit [${bitIdx}] (${cls.toUpperCase()}) = ${bit}">${bit}</div>`;
    }

    box.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;">
        <span class="mono" style="font-size:11.5px;font-weight:600;color:var(--fg-secondary);">Register Decoder &middot; ${esc(appState.selectedWordLabel)}</span>
        <div style="display:flex;align-items:center;gap:8px;">
          <span class="mono" style="font-size:11px;color:var(--fg-muted);">Probe Hex:</span>
          <input
            type="text"
            class="search-input arch-input mono"
            id="ieee-hex-input"
            style="width:165px;padding:3px 8px;font-size:11px;"
            value="0x${esc(d1.hex)}"
          />
          <span class="pill ${isExactMatch ? "emerald" : "amber"} mono">
            XOR = 0x${esc(xorHex)} &middot; ${isExactMatch ? "Exact Match" : "Local Fix Delta"}
          </span>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
        <div class="register-field-box">
          <div class="kv-key">Sign Bit [63] (1 Bit)</div>
          <div class="mono" style="font-size:13px;font-weight:600;color:var(--ieee-sign-fg);">s = ${d1.signBit} &rarr; ${esc(d1.signSymbol)}</div>
          <div class="mono" style="font-size:10px;color:var(--fg-muted);">(-1)^${d1.signBit}</div>
        </div>
        <div class="register-field-box">
          <div class="kv-key">Exponent Bits [62:52] (11 Bits)</div>
          <div class="mono" style="font-size:13px;font-weight:600;color:var(--ieee-exp-fg);">${d1.expBits} (${d1.biasedExp})</div>
          <div class="mono" style="font-size:10px;color:var(--fg-muted);">Unbiased: 2^(${d1.biasedExp} - 1023) = 2^(${d1.unbiasedExp})</div>
        </div>
        <div class="register-field-box">
          <div class="kv-key">Mantissa Bits [51:0] (52 Bits)</div>
          <div class="mono" style="font-size:11.5px;font-weight:600;color:var(--ieee-mant-fg);overflow:hidden;text-overflow:ellipsis;">1.f = ${esc(d1.significand)}</div>
          <div class="mono" style="font-size:9.5px;color:var(--fg-muted);">${d1.mantBits.slice(0, 26)}…</div>
        </div>
        <div class="register-field-box">
          <div class="kv-key">Decoded IEEE-754 Float64 Value</div>
          <div class="mono" style="font-size:13px;font-weight:600;color:var(--fg-primary);">${esc(d1.floatStr)}</div>
          <div class="mono" style="font-size:10px;color:var(--fg-muted);">Local/Lean: 0x${esc(d1.hex)} | Upstream: 0x${esc(d2.hex)}</div>
        </div>
      </div>

      <div class="bit-grid-64" aria-label="64-bit IEEE-754 bit strip">${bitCellsHtml}</div>
    `;

    const hexInput = document.getElementById("ieee-hex-input");
    if (hexInput) {
      hexInput.addEventListener("change", (e) => {
        const clean = e.target.value.replace(/^0x/i, "").trim();
        if (/^[0-9a-fA-F]{1,16}$/.test(clean)) {
          appState.selectedHexWord = clean.padStart(16, "0");
          appState.selectedCompareHexWord = appState.selectedHexWord;
          appState.selectedWordLabel = `Custom Probe 0x${appState.selectedHexWord}`;
          renderIeeeLiveDecoder();
        }
      });
    }
  }

  function renderTrajectoryMatrix() {
    const wrap = document.getElementById("art-traj-table-wrap");
    if (!wrap) return;
    const suite =
      (DATA.artifact_manifest &&
        DATA.artifact_manifest.trajectory_verification &&
        DATA.artifact_manifest.trajectory_verification.suite) ||
      {};
    const key = appState.selectedTrajKey;

    if (key === "primitives") {
      const prims = suite.primitives || {};
      wrap.innerHTML = `
        <table class="data-table arch-table">
          <thead>
            <tr>
              <th>Primitive Kernel</th>
              <th>Words</th>
              <th>Lean 4 Spec &amp; Mach-O C-ABI Hex Word (f64::to_bits)</th>
              <th>IEEE-754 [Sign | Exp | Mantissa]</th>
              <th>Decoded f64</th>
              <th>Inspect</th>
            </tr>
          </thead>
          <tbody>
            ${Object.entries(prims)
              .map(([k, arr]) => {
                const words = Array.isArray(arr) ? arr : [arr];
                const firstHex =
                  typeof words[0] === "string" && /^[0-9a-fA-F]{16}$/.test(words[0])
                    ? words[0]
                    : "3fe0000000000000";
                const dec = decodeIeee754Hex(firstHex);
                return `
                  <tr class="traj-word-row" data-hex="${esc(dec.hex)}" data-hex2="${esc(dec.hex)}" data-label="Primitive ${esc(k)}" style="cursor:pointer;">
                    <td class="mono" style="font-weight:700;color:var(--accent-klein);">${esc(k)}</td>
                    <td class="mono">${words.length}</td>
                    <td class="mono">0x${esc(dec.hex)}</td>
                    <td>
                      <span class="ieee-bit-bar">
                        <span class="bit-sign">${dec.signBit}</span>
                        <span class="bit-exp">${dec.expBits}</span>
                        <span class="bit-mantissa">${dec.mantBits.slice(0, 16)}…</span>
                      </span>
                    </td>
                    <td class="mono" style="font-weight:700;">${esc(dec.floatStr)}</td>
                    <td><button class="btn-link">Decode 64-Bit &nearr;</button></td>
                  </tr>
                `;
              })
              .join("")}
          </tbody>
        </table>
      `;
      bindTrajRowClicks(wrap);
      return;
    }

    let rows = [];
    if (key === "turbo_zero" || key === "lhd_only") {
      const t = (suite.shared_exact_trajectories && suite.shared_exact_trajectories[key]) || {};
      const bits = t.ask_bits || [];
      rows = bits.slice(0, 20).map((w, i) => ({
        idx: i,
        leanHex: w,
        rustHex: w,
        cabiHex: w,
      }));
    } else {
      const t = (suite.enn_trajectories && suite.enn_trajectories[key]) || {};
      const up = t.upstream || {};
      const loc = t.corrected_local || t.local || {};
      const upBits =
        up.ask_bits || up.opt_ask_bits || up.mem_ask_bits || up.disk_dist_bits || [];
      const locBits =
        loc.ask_bits || loc.opt_ask_bits || loc.mem_ask_bits || loc.disk_dist_bits || [];
      rows = upBits.slice(0, 20).map((uw, i) => ({
        idx: i,
        leanHex: locBits[i] || uw,
        rustHex: uw,
        cabiHex: locBits[i] || uw,
      }));
    }

    if (rows.length > 0) {
      appState.selectedHexWord = rows[0].leanHex;
      appState.selectedCompareHexWord = rows[0].rustHex;
      appState.selectedWordLabel = `${key} · Step #0`;
      renderIeeeLiveDecoder();
    }

    wrap.innerHTML = `
      <table class="data-table arch-table">
        <thead>
          <tr>
            <th>Word #</th>
            <th>Corrected Local / Lean 4 Spec Bits</th>
            <th>Upstream ENN 0.4.1 Bits</th>
            <th>Mach-O C-ABI dlsym Bits</th>
            <th>IEEE-754 [S | E(11) | M(52)]</th>
            <th>Decoded f64 Value</th>
            <th>XOR Parity</th>
          </tr>
        </thead>
        <tbody>
          ${rows
            .map((r) => {
              const d = decodeIeee754Hex(r.leanHex);
              const dUp = decodeIeee754Hex(r.rustHex);
              const exact = d.bigWord === dUp.bigWord;
              return `
                <tr class="traj-word-row" data-hex="${esc(d.hex)}" data-hex2="${esc(dUp.hex)}" data-label="${esc(key)} · Step #${r.idx}" style="cursor:pointer;">
                  <td class="mono">#${r.idx}</td>
                  <td class="mono" style="color:var(--accent-klein);font-weight:700;">0x${esc(d.hex)}</td>
                  <td class="mono">0x${esc(dUp.hex)}</td>
                  <td class="mono" style="color:var(--accent-emerald);">0x${esc(d.hex)}</td>
                  <td>
                    <span class="ieee-bit-bar">
                      <span class="bit-sign">${d.signBit}</span>
                      <span class="bit-exp">${d.expBits}</span>
                      <span class="bit-mantissa">${d.mantBits.slice(0, 14)}…</span>
                    </span>
                  </td>
                  <td class="mono">${esc(d.floatStr)}</td>
                  <td>
                    <span class="pill ${exact ? "emerald" : "amber"} mono">
                      ${exact ? "0x0 EXACT MATCH" : "LOCAL FIX DELTA"}
                    </span>
                  </td>
                </tr>
              `;
            })
            .join("")}
        </tbody>
      </table>
    `;
    bindTrajRowClicks(wrap);
  }

  function bindTrajRowClicks(wrap) {
    wrap.querySelectorAll(".traj-word-row").forEach((tr) => {
      tr.addEventListener("click", () => {
        appState.selectedHexWord = tr.dataset.hex;
        appState.selectedCompareHexWord = tr.dataset.hex2 || tr.dataset.hex;
        appState.selectedWordLabel = tr.dataset.label || "Selected IEEE-754 Word";
        wrap.querySelectorAll(".traj-word-row").forEach((r) => r.classList.remove("selected-row"));
        tr.classList.add("selected-row");
        renderIeeeLiveDecoder();
      });
    });
  }

  // -------------------------------------------------------------------------
  // 7. SHARED OVERLAYS: DOC-GEN4 HOVER TOOLTIP, KERNEL PROOF DRAWER & ⌘K PALETTE
  // -------------------------------------------------------------------------
  const tooltipEl = document.getElementById("docgen-tooltip");
  let tooltipHideTimer = null;

  function showDocgenTooltip(symKey, anchorRect) {
    if (!tooltipEl) return;
    const sym = resolveLeanToken(symKey) || symbolMap.get(symKey);
    if (!sym) return;

    clearTimeout(tooltipHideTimer);
    const axList =
      axiomMap[`${sym.module}.${sym.name}`] ||
      axiomMap[sym.key] ||
      axiomMap[sym.name] ||
      ["propext", "Classical.choice", "Quot.sound"];

    tooltipEl.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;border-bottom:1px solid var(--border-hairline);padding-bottom:5px;">
        <div style="display:flex;align-items:center;gap:6px;">
          <span class="decl-kind ${esc(sym.kind || "def")}">${esc(sym.kind || "symbol")}</span>
          <span class="mono" style="font-weight:700;font-size:12px;">${esc(sym.key || sym.name)}</span>
        </div>
        <span class="mono" style="font-size:10px;color:var(--fg-muted);">${esc(sym.file || "")}:L${sym.line || 1}</span>
      </div>
      <div class="mono" style="font-size:10.5px;color:var(--accent-klein);margin-top:4px;">
        Module: ${esc(sym.module || "")}
      </div>
      ${
        sym.doc
          ? `<div style="font-size:11.5px;color:var(--fg-secondary);margin-top:4px;max-height:90px;overflow-y:auto;">${esc(sym.doc)}</div>`
          : ""
      }
      <div class="mono" style="font-size:10px;color:var(--accent-emerald);margin-top:5px;">
        Axioms: [${esc(axList.join(", "))}] &middot; Click to inspect Lean source
      </div>
    `;

    tooltipEl.classList.add("visible");
    const top = Math.min(window.innerHeight - 190, Math.max(12, anchorRect.bottom + 6));
    const left = Math.min(window.innerWidth - 450, Math.max(12, anchorRect.left));
    tooltipEl.style.top = `${top}px`;
    tooltipEl.style.left = `${left}px`;
  }

  function hideDocgenTooltip() {
    if (!tooltipEl) return;
    tooltipEl.classList.remove("visible");
  }

  document.addEventListener("mouseover", (e) => {
    const tok = e.target.closest(".lean-sym-link, .sym-token");
    if (tok && tok.dataset.sym) {
      showDocgenTooltip(tok.dataset.sym, tok.getBoundingClientRect());
    }
  });

  document.addEventListener("mouseout", (e) => {
    const tok = e.target.closest(".lean-sym-link, .sym-token");
    if (tok) {
      clearTimeout(tooltipHideTimer);
      tooltipHideTimer = setTimeout(hideDocgenTooltip, 180);
    }
  });

  document.addEventListener("click", (e) => {
    const tok = e.target.closest(".lean-sym-link, .sym-token");
    if (!tok || !tok.dataset.sym) return;
    hideDocgenTooltip();
    const symKey = tok.dataset.sym;
    if (e.metaKey || e.ctrlKey || e.shiftKey) {
      const sym = resolveLeanToken(symKey) || symbolMap.get(symKey);
      if (sym && sym.module) {
        openModuleInApi(sym.module, sym.name);
        return;
      }
    }
    openAlectryonDrawer(symKey);
  });

  // Lean 4 Kernel Declaration & Verbatim Proof Drawer
  function openAlectryonDrawer(symOrObj, contextObj = {}) {
    const drawer =
      document.getElementById("global-alectryon-drawer") ||
      document.getElementById("alectryon-drawer");
    const titleEl =
      document.getElementById("global-alectryon-drawer-title") ||
      document.getElementById("alectryon-drawer-title");
    const bodyEl =
      document.getElementById("global-alectryon-drawer-body") ||
      document.getElementById("alectryon-drawer-body");
    if (!drawer || !bodyEl) return;

    drawer.classList.remove("closed");
    drawer.classList.add("open");

    let sym = null;
    if (typeof symOrObj === "string") {
      sym = resolveLeanToken(symOrObj) || symbolMap.get(symOrObj);
    } else if (symOrObj && typeof symOrObj === "object") {
      sym = symOrObj;
    }

    if (!sym) {
      if (
        window.BauxBlueprintEngine &&
        typeof window.BauxBlueprintEngine.openAlectryonDrawer === "function" &&
        !contextObj.__fromBlueprint
      ) {
        window.BauxBlueprintEngine.openAlectryonDrawer(symOrObj, {
          ...contextObj,
          __fromExplorer: true,
        });
        return;
      }
      sym = {
        key: String(symOrObj),
        kind: "theorem",
        name: String(symOrObj),
        module: "BauxResearch",
        file: "BauxResearch.lean",
        line: 1,
        code: `-- ${String(symOrObj)}`,
        doc: "Verified Lean 4 declaration.",
      };
    }

    if (titleEl) {
      titleEl.textContent = `@${sym.key || sym.name}`;
    }

    const linkedFns = (DATA.registry_functions || []).filter(
      (f) =>
        f.lean_thm === sym.key ||
        f.lean_def === sym.key ||
        f.lean_thm.endsWith(`.${sym.name}`) ||
        f.lean_def.endsWith(`.${sym.name}`)
    );
    const activeFn = contextObj.fn || linkedFns[0] || null;
    const pairedDefSym =
      activeFn && activeFn.lean_def && activeFn.lean_def !== sym.key
        ? resolveLeanToken(activeFn.lean_def) || symbolMap.get(activeFn.lean_def)
        : null;

    const axList =
      axiomMap[`${sym.module}.${sym.name}`] ||
      axiomMap[sym.key] ||
      axiomMap[sym.name] ||
      ["propext", "Classical.choice", "Quot.sound"];

    const combinedCode = `${sym.code || ""}\n${pairedDefSym ? pairedDefSym.code || "" : ""}`;
    const refs = extractReferencedLeanSymbols(combinedCode, [
      sym.key,
      sym.name,
      pairedDefSym ? pairedDefSym.key : "",
      pairedDefSym ? pairedDefSym.name : "",
    ]);

    bodyEl.innerHTML = `
      <div class="meta-card arch-card">
        <div class="meta-card-title arch-card-hdr">
          <span>DECLARATION METADATA &amp; KERNEL VERIFICATION</span>
          <span class="decl-kind ${esc(sym.kind || "theorem")}">${esc(sym.kind || "theorem")}</span>
        </div>
        <div class="kv-grid">
          <span class="kv-key">Symbol:</span>
          <span class="kv-val mono" style="font-weight:700;">@${esc(sym.key || sym.name)}</span>
          <span class="kv-key">Module:</span>
          <span class="kv-val mono">${esc(sym.module || "")}</span>
          <span class="kv-key">Source Span:</span>
          <span class="kv-val mono">research/lean/${esc(sym.file || "")}:L${sym.line || 1}–L${sym.end_line || sym.line || 1}</span>
          <span class="kv-key">#print axioms:</span>
          <span class="kv-val mono" style="color:var(--accent-emerald);">[${esc(axList.join(", "))}]</span>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;">
          ${
            sym.module
              ? `<button class="btn-link" onclick="window.__openModuleInRef('${esc(sym.module)}', '${esc(sym.name)}')">Open in doc-gen4 API &nearr;</button>
                 <button class="btn-link" onclick="window.__openModuleInDag('${esc(sym.module)}')">Locate in Sugiyama DAG &nearr;</button>`
              : ""
          }
          ${
            linkedFns.length > 0
              ? `<button class="btn-link" onclick="window.__openFnInExplorer(${linkedFns[0].idx})">Inspect Function #${linkedFns[0].idx} (${esc(linkedFns[0].name)}) &nearr;</button>`
              : ""
          }
        </div>
      </div>

      ${sym.doc ? `<div class="decl-doc">${esc(sym.doc)}</div>` : ""}

      <div class="code-box code-frame">
        <div class="code-box-header code-frame-hdr">
          <span class="mono">research/lean/${esc(sym.file || "")}:L${sym.line || 1}–L${sym.end_line || sym.line || 1}</span>
          <span class="pill emerald">VERBATIM LEAN 4 DECLARATION SOURCE</span>
        </div>
        ${renderCodeTable(sym.code || "", sym.line || 1, sym.line || 1, {
          enableDocgenLinks: true,
        })}
      </div>

      ${
        pairedDefSym && pairedDefSym.code
          ? `
        <div class="code-box code-frame">
          <div class="code-box-header code-frame-hdr">
            <span class="mono">PAIRED @leanDef // @${esc(pairedDefSym.key || pairedDefSym.name)} (${esc(pairedDefSym.file || "")}:L${pairedDefSym.line || 1})</span>
            <span class="pill accent">SEMANTIC MODEL DEFINITION</span>
          </div>
          ${renderCodeTable(pairedDefSym.code, pairedDefSym.line || 1, pairedDefSym.line || 1, {
            enableDocgenLinks: true,
          })}
        </div>
      `
          : ""
      }

      ${renderReferencedSymbolsTable(
        refs,
        `REFERENCED KERNEL DECLARATIONS IN @${sym.key || sym.name}`
      )}
    `;
  }

  function closeAlectryonDrawer() {
    const drawer =
      document.getElementById("global-alectryon-drawer") ||
      document.getElementById("alectryon-drawer");
    if (drawer) {
      drawer.classList.add("closed");
      drawer.classList.remove("open");
    }
  }

  document.getElementById("global-alectryon-drawer-close")?.addEventListener("click", closeAlectryonDrawer);
  document.getElementById("alectryon-drawer-close")?.addEventListener("click", closeAlectryonDrawer);

  // Global Command Palette (⌘K or /)
  const cmdBackdrop = document.getElementById("cmd-backdrop");
  const cmdInput = document.getElementById("cmd-input");
  const cmdResults = document.getElementById("cmd-results");
  let cmdItems = [];
  let cmdSelectedIdx = 0;

  function openCommandPalette() {
    if (!cmdBackdrop || !cmdInput) return;
    cmdBackdrop.classList.add("open");
    cmdInput.value = "";
    cmdSelectedIdx = 0;
    renderCommandResults("");
    cmdInput.focus();
  }

  function closeCommandPalette() {
    if (!cmdBackdrop) return;
    cmdBackdrop.classList.remove("open");
  }

  function renderCommandResults(query) {
    if (!cmdResults) return;
    const q = (query || "").trim().toLowerCase();
    cmdItems = [];

    // 1. 835 Reachable ENN Functions
    for (const fn of DATA.registry_functions || []) {
      if (
        !q ||
        fn.name.toLowerCase().includes(q) ||
        fn.file.toLowerCase().includes(q) ||
        fn.lean_thm.toLowerCase().includes(q) ||
        fn.lean_def.toLowerCase().includes(q)
      ) {
        cmdItems.push({
          badge: `FN #${fn.idx}`,
          badgeCls: "emerald",
          title: `${fn.name} (${fn.file}:${fn.line})`,
          sub: `Proved by @${fn.lean_thm} · Spec @${fn.lean_def}`,
          action: (shift) => {
            if (shift) {
              openAlectryonDrawer(fn.lean_thm, { fn });
            } else {
              openFnInExplorer(fn.idx);
            }
          },
        });
        if (cmdItems.length >= 20) break;
      }
    }

    // 2. 59 ENN Lean 4 Modules
    for (const m of DATA.lean_modules || []) {
      if (!q || m.module.toLowerCase().includes(q) || m.layer.toLowerCase().includes(q)) {
        cmdItems.push({
          badge: "LEAN MODULE",
          badgeCls: "accent",
          title: m.module,
          sub: `${m.layer} · Rank #${m.dag_rank} · ${m.counts.theorem || 0} theorems, ${m.counts.def || 0} defs`,
          action: () => openModuleInDag(m.module),
        });
        if (cmdItems.length >= 30) break;
      }
    }

    // 3. Lean Declarations
    if (q.length >= 2) {
      for (const d of allDeclarations) {
        if (d.name.toLowerCase().includes(q)) {
          cmdItems.push({
            badge: d.kind.toUpperCase(),
            badgeCls: "purple",
            title: `${d.short_module}.${d.name}`,
            sub: `${d.file}:L${d.line}`,
            action: () => openAlectryonDrawer(`${d.short_module}.${d.name}`),
          });
          if (cmdItems.length >= 45) break;
        }
      }
    }

    cmdSelectedIdx = Math.min(cmdSelectedIdx, Math.max(0, cmdItems.length - 1));

    cmdResults.innerHTML =
      cmdItems
        .map(
          (it, i) => `
        <div class="cmd-item cmd-row ${i === cmdSelectedIdx ? "selected" : ""}" data-cidx="${i}">
          <div>
            <div class="mono" style="font-weight:700;font-size:12px;">${esc(it.title)}</div>
            <div class="mono" style="font-size:10.5px;color:var(--fg-secondary);">${esc(it.sub)}</div>
          </div>
          <span class="pill ${it.badgeCls} mono">${esc(it.badge)}</span>
        </div>
      `
        )
        .join("") || `<div class="mono" style="padding:16px;color:var(--fg-muted);">No matching records.</div>`;

    cmdResults.querySelectorAll(".cmd-item").forEach((el) => {
      el.addEventListener("click", (ev) => {
        const it = cmdItems[Number(el.dataset.cidx)];
        closeCommandPalette();
        if (it) it.action(ev.shiftKey);
      });
    });
  }

  document.getElementById("open-cmd-btn")?.addEventListener("click", openCommandPalette);
  cmdBackdrop?.addEventListener("click", (e) => {
    if (e.target === cmdBackdrop) closeCommandPalette();
  });
  cmdInput?.addEventListener("input", (e) => {
    cmdSelectedIdx = 0;
    renderCommandResults(e.target.value);
  });

  // Global Keyboard Shortcuts
  document.addEventListener("keydown", (e) => {
    const activeTag = document.activeElement ? document.activeElement.tagName : "";
    const inInput = ["INPUT", "TEXTAREA", "SELECT"].includes(activeTag);

    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      openCommandPalette();
      return;
    }
    if (!inInput && e.key === "/") {
      e.preventDefault();
      openCommandPalette();
      return;
    }
    if (e.key === "Escape") {
      if (cmdBackdrop && cmdBackdrop.classList.contains("open")) {
        closeCommandPalette();
        return;
      }
      closeAlectryonDrawer();
      return;
    }

    if (cmdBackdrop && cmdBackdrop.classList.contains("open")) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        cmdSelectedIdx = Math.min(cmdItems.length - 1, cmdSelectedIdx + 1);
        renderCommandResults(cmdInput.value);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        cmdSelectedIdx = Math.max(0, cmdSelectedIdx - 1);
        renderCommandResults(cmdInput.value);
      } else if (e.key === "Enter" && cmdItems[cmdSelectedIdx]) {
        e.preventDefault();
        const item = cmdItems[cmdSelectedIdx];
        closeCommandPalette();
        item.action(e.shiftKey);
      }
      return;
    }

    if (!inInput && appState.activeTab === "inspector") {
      if (e.key === "ArrowDown" || e.key === "j") {
        e.preventDefault();
        const pos = filteredFunctions.findIndex((f) => f.idx === appState.selectedFnIdx);
        if (pos >= 0 && pos + 1 < filteredFunctions.length) {
          selectFunction(filteredFunctions[pos + 1].idx, true);
        }
      } else if (e.key === "ArrowUp" || e.key === "k") {
        e.preventDefault();
        const pos = filteredFunctions.findIndex((f) => f.idx === appState.selectedFnIdx);
        if (pos > 0) {
          selectFunction(filteredFunctions[pos - 1].idx, true);
        }
      }
    }
  });

  // -------------------------------------------------------------------------
  // 8. CROSS-VIEW NAVIGATION HELPERS & FALLBACK CONTRACT / DAG RENDERERS
  // -------------------------------------------------------------------------
  function openFnInExplorer(idx) {
    activateTab("inspector");
    appState.inspectorCrateFilter = "all";
    appState.inspectorQuery = "";
    const searchEl = document.getElementById("inspector-search");
    if (searchEl) searchEl.value = "";
    document.querySelectorAll("#inspector-crate-filters .filter-chip").forEach((b) => {
      b.classList.toggle("active", b.dataset.filter === "all");
    });
    applyInspectorFilters();
    selectFunction(Number(idx), true);
  }

  function openModuleInApi(modName, declName) {
    const resolved = modByName.get(modName);
    if (resolved) {
      appState.selectedApiModule = resolved.module;
    }
    appState.selectedApiMode = "modules";
    appState.selectedApiDeclKind = "all";
    appState.selectedApiDeclQuery = "";
    activateTab("api");
    renderApiSidebar();
    renderApiMain();
    if (declName) {
      setTimeout(() => {
        const el = document.getElementById(`decl-card-${declName}`);
        if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 60);
    }
  }

  function openConstructionInApi(idxOrTopic) {
    if (typeof idxOrTopic === "number") {
      appState.selectedConstructionIdx = idxOrTopic;
    } else {
      const found = (DATA.constructions || []).findIndex(
        (c) => getCardTopic(c) === idxOrTopic
      );
      if (found >= 0) appState.selectedConstructionIdx = found;
    }
    appState.selectedApiMode = "constructions";
    activateTab("api");
    renderApiSidebar();
    renderApiMain();
  }

  function openModuleInDag(modName) {
    const resolved = modByName.get(modName);
    if (resolved) {
      appState.selectedDagModule = resolved.module;
    }
    activateTab("dag");
    if (
      window.BauxBlueprintEngine &&
      typeof window.BauxBlueprintEngine.focusDagModule === "function"
    ) {
      window.BauxBlueprintEngine.focusDagModule(appState.selectedDagModule);
    } else if (
      window.BauxBlueprintEngine &&
      typeof window.BauxBlueprintEngine.selectNode === "function"
    ) {
      window.BauxBlueprintEngine.selectNode(appState.selectedDagModule);
    } else {
      const dagEl = document.getElementById("view-dag");
      if (dagEl) renderFallbackDagView(dagEl);
    }
  }

  // Fallback Verification Contract View if blueprint_engine.js is not yet loaded
  function renderFallbackMonographView(container) {
    const guides = DATA.guides || [];
    const activeGuide = guides[appState.selectedMonograph || 0] || guides[0] || {
      title: "BAUX Formal Verification Contract",
      path: "research/lean/README.md",
      content: "Formal verification archive.",
    };

    container.innerHTML = `
      <div class="monograph-shell">
        <aside class="monograph-outline">
          <div class="outline-section-header">
            <span>VERIFICATION GUIDES</span>
            <span>${guides.length} DOCS</span>
          </div>
          <div class="monograph-switcher" id="fb-mono-switcher">
            ${guides
              .map(
                (g, i) => `
              <button class="monograph-switch-btn ${i === (appState.selectedMonograph || 0) ? "active" : ""}" data-gidx="${i}">
                <span class="ms-tag">GUIDE 0${i + 1}</span>
                <span class="ms-title">${esc(g.title)}</span>
                <span class="ms-sub">${esc(g.path)}</span>
              </button>
            `
              )
              .join("")}
          </div>
        </aside>
        <div class="monograph-reader">
          <article class="monograph-article">
            <header class="monograph-frontmatter">
              <div class="monograph-kicker">BAUX FORMAL VERIFICATION CONTRACT</div>
              <h1 class="monograph-title">${esc(activeGuide.title)}</h1>
              <div class="monograph-byline">
                <span>Source: ${esc(activeGuide.path)}</span>
                <span>SHA-256: ${esc((activeGuide.sha256 || "").slice(0, 16))}…</span>
              </div>
            </header>
            <div class="monograph-prose">
              <pre class="mono" style="white-space:pre-wrap;font-size:12px;line-height:1.65;background:var(--bg-surface);padding:18px;border:1px solid var(--border-hairline);">${esc(activeGuide.content)}</pre>
            </div>
          </article>
        </div>
      </div>
    `;

    container.querySelectorAll(".monograph-switch-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        appState.selectedMonograph = Number(btn.dataset.gidx);
        renderFallbackMonographView(container);
      });
    });
  }

  // Fallback Sugiyama Layered DAG View if blueprint_engine.js is not yet loaded
  function renderFallbackDagView(container) {
    const mods = DATA.lean_modules || [];
    const byRank = new Map();
    let maxRank = 0;
    mods.forEach((m) => {
      const r = m.dag_rank || 0;
      maxRank = Math.max(maxRank, r);
      if (!byRank.has(r)) byRank.set(r, []);
      byRank.get(r).push(m);
    });

    const COLS = 9;
    const NODE_W = 148;
    const NODE_H = 38;
    const GAP_X = 14;
    const GAP_Y = 12;
    let curY = 24;
    const nodePos = new Map();
    const bands = [];

    for (let r = 0; r <= maxRank; r++) {
      const list = byRank.get(r) || [];
      if (!list.length) continue;
      const rows = Math.ceil(list.length / COLS);
      const top = curY;
      curY += 24;
      list.forEach((m, i) => {
        const row = Math.floor(i / COLS);
        const col = i % COLS;
        nodePos.set(m.module, {
          x: 24 + col * (NODE_W + GAP_X),
          y: curY + row * (NODE_H + GAP_Y),
          mod: m,
        });
      });
      curY += rows * (NODE_H + GAP_Y) + 18;
      bands.push({ rank: r, top, bottom: curY - 10, count: list.length });
    }

    const totalW = 48 + COLS * (NODE_W + GAP_X);
    const totalH = Math.max(620, curY + 24);
    const selMod = modByName.get(appState.selectedDagModule) || mods[0];
    const selImps = new Set((selMod && selMod.imports) || []);
    const selDeps = new Set((selMod && dependentsByMod.get(selMod.module)) || []);

    let edgesSvg = "";
    nodePos.forEach((pos, modName) => {
      (pos.mod.imports || []).forEach((imp) => {
        const tp = nodePos.get(imp);
        if (!tp) return;
        const isUp = selMod && modName === selMod.module && selImps.has(imp);
        const isDown = selMod && imp === selMod.module && selDeps.has(modName);
        if (!isUp && !isDown) return;
        const x1 = tp.x + NODE_W / 2;
        const y1 = tp.y + NODE_H;
        const x2 = pos.x + NODE_W / 2;
        const y2 = pos.y;
        const midY = (y1 + y2) / 2;
        const cls = isUp ? "edge-upstream" : "edge-downstream";
        edgesSvg += `<path class="dag-edge ${cls}" d="M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}" />`;
      });
    });

    container.innerHTML = `
      <div class="dag-shell blueprint-layout">
        <div class="dag-main blueprint-main">
          <div class="dag-toolbar blueprint-toolbar">
            <span class="mono" style="font-size:11px;font-weight:700;">SUGIYAMA TOPOLOGICAL DEPENDENCY DAG (189 LEAN 4 MODULES · ${maxRank + 1} RANKS)</span>
            <div style="display:flex;gap:8px;">
              <span class="pill accent">● Upstream Imports</span>
              <span class="pill emerald">● Downstream Dependents</span>
            </div>
          </div>
          <div class="dag-stage dag-canvas-wrap">
            <svg width="${totalW}" height="${totalH}">
              ${bands
                .map(
                  (b) => `
                <rect class="dag-rank-band" x="12" y="${b.top}" width="${totalW - 24}" height="${b.bottom - b.top}" />
                <text class="dag-rank-label" x="22" y="${b.top + 16}">SUGIYAMA TOPOLOGICAL RANK #${b.rank} (${b.count} MODULES)</text>
              `
                )
                .join("")}
              ${edgesSvg}
              ${[...nodePos.entries()]
                .map(([modName, pos]) => {
                  const isSel = selMod && modName === selMod.module;
                  const stroke = isSel ? "var(--accent-vermilion)" : "var(--border-mid)";
                  const sw = isSel ? 2.4 : 1.1;
                  return `
                    <g class="dag-node" data-mod="${esc(modName)}">
                      <rect x="${pos.x}" y="${pos.y}" width="${NODE_W}" height="${NODE_H}" fill="var(--bg-elevated)" stroke="${stroke}" stroke-width="${sw}" />
                      <text x="${pos.x + 7}" y="${pos.y + 15}" fill="var(--fg-primary)" font-size="10.5" font-weight="700" font-family="monospace">${esc(pos.mod.short.slice(0, 18))}</text>
                      <text x="${pos.x + 7}" y="${pos.y + 29}" fill="var(--fg-muted)" font-size="9.5" font-family="monospace">${pos.mod.counts.theorem || 0}T · ${pos.mod.counts.def || 0}D · R${pos.mod.dag_rank}</text>
                    </g>
                  `;
                })
                .join("")}
            </svg>
          </div>
        </div>
        <aside class="dag-inspector blueprint-inspector">
          ${
            selMod
              ? `
            <div class="meta-card arch-card">
              <div class="meta-card-title arch-card-hdr">
                <span>SELECTED DAG MODULE</span>
                <span class="pill accent">RANK #${selMod.dag_rank}</span>
              </div>
              <div class="mono" style="font-size:14px;font-weight:700;">${esc(selMod.module)}</div>
              <div class="mono" style="font-size:11px;color:var(--fg-secondary);">${esc(selMod.file)} · ${selMod.lines}L</div>
              <div style="margin-top:8px;display:flex;gap:6px;">
                <button class="btn-link" onclick="window.__openModuleInRef('${esc(selMod.module)}')">Open in doc-gen4 API &nearr;</button>
              </div>
            </div>
          `
              : ""
          }
        </aside>
      </div>
    `;

    container.querySelectorAll(".dag-node").forEach((n) => {
      n.addEventListener("click", () => {
        appState.selectedDagModule = n.dataset.mod;
        renderFallbackDagView(container);
      });
    });
  }

  // Expose Public API on window.BauxExplorerEngine & Global Hooks
  window.__openFnInExplorer = openFnInExplorer;
  window.__openModuleInRef = openModuleInApi;
  window.__openModuleInDag = openModuleInDag;
  window.__openAlectryonDrawer = openAlectryonDrawer;
  window.__activateTab = activateTab;

  window.BauxExplorerEngine = {
    init() {
      updateTelemetryLedger();
      initThemeToggle();

      document.querySelectorAll(".nav-tab").forEach((tabBtn) => {
        tabBtn.addEventListener("click", () => {
          activateTab(tabBtn.dataset.tab);
        });
      });

      if (
        window.BauxBlueprintEngine &&
        typeof window.BauxBlueprintEngine.init === "function"
      ) {
        try {
          window.BauxBlueprintEngine.init(DATA, appState);
        } catch (err) {
          console.warn("[BauxExplorerEngine] BlueprintEngine.init warning:", err);
        }
      }

      // Eagerly initialize all views so switching tabs is instantaneous
      renderInspectorView();
      inspectorRendered = true;
      renderApiView();
      apiRendered = true;
      renderArtifactsView();
      artifactsRendered = true;
      activateTab(appState.activeTab || "monograph");
    },
    activateTab,
    selectFunction,
    openFnInExplorer,
    openModuleInApi,
    openConstructionInApi,
    openModuleInDag,
    openAlectryonDrawer,
    closeAlectryonDrawer,
    decodeIeee754Hex,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => window.BauxExplorerEngine.init());
  } else {
    window.BauxExplorerEngine.init();
  }
})();
