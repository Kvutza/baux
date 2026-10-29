use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};

use charming::Chart;
use charming::{component, element, series};

use baux::artifact::json::{self, Value};
use baux::artifact::runs::plot;
use baux::artifact::runs::{Record, atomic, directory, field};

/// Colorblind-safe Okabe-Ito publication palette.
const OKABE_ITO: &[&str] = &[
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#332288", "#882255",
    "#117733", "#999933", "#44AA99", "#AA4499",
];

struct RunView {
    idx: usize,
    label: String,
    color: &'static str,
    ctx: &'static str,
    cohort: &'static str,
    optimizer: String,
    summary_only: bool,
    first_nll: Option<f64>,
    last_nll: Option<f64>,
    delta_nll: Option<f64>,
    wall_ms: Option<f64>,
}

/// Write a publication-grade + W&B-style interactive HTML dashboard powered by
/// `charming` (Apache ECharts for Rust with SVG vector rendering) and `jj`
/// optimizer code diffs.
pub fn write_html(
    root: &Path,
    records: &[Record],
    _comparisons: &Value,
    diffs: &Value,
) -> Result<PathBuf, String> {
    let out_dir = directory(root)?.join("comparisons");
    fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;

    let nll_keys: &[&str] = &["incumbent_nll", "nll_after", "best_nll", "mean_nll"];
    let val_keys: &[&str] = &["nll", "mean_nll"];
    let wall_keys: &[&str] = &["round_wall_ms", "wall_ms"];
    let has_full_vocab = records.iter().any(|r| {
        plot::extract_series(r, false, nll_keys)
            .and_then(|s| s.first().copied())
            .is_some_and(|(_, y)| y > 11.0)
    });

    let views = select_runs(records, has_full_vocab, nll_keys, wall_keys);
    let specs: &[(&str, &str, &str, &str, bool)] = &[
        ("delta_nll", "ΔNLL", "step", "ΔNLL", false),
        ("nll", "NLL", "step", "NLL", false),
        ("validation_nll", "Val ΔNLL", "step", "val ΔNLL", false),
        ("nll_by_time", "ΔNLL vs. Time", "time (s)", "ΔNLL", false),
        ("wall_ms", "ms/round", "step", "ms/round", true),
    ];

    let mut chart_configs = Vec::new();
    for &(metric, title, x_label, y_label, log_y) in specs {
        if let Some(cfg_json) = charming_chart(
            records,
            &views,
            metric,
            title,
            x_label,
            y_label,
            log_y,
            has_full_vocab,
            nll_keys,
            val_keys,
            wall_keys,
        ) {
            chart_configs.push((metric, title, cfg_json));
        }
    }
    if let Some(summary_json) = summary_chart(&views) {
        chart_configs.push(("summary_bar", "−ΔNLL Summary", summary_json));
    }

    let mut html = String::from(HEADER);
    append_sidebar(&mut html, &views);
    html.push_str("<main class=\"wb-main\">\n");
    html.push_str("<div class=\"wb-charts-grid\">\n");
    for (metric, title, _) in &chart_configs {
        html.push_str(&format!(
            "<div class=\"wb-panel\"><div class=\"wb-panel-head\"><div \
             class=\"wb-fig-title\">{}</div></div><div id=\"chart-{metric}\" \
             class=\"wb-chart\"></div></div>\n",
            escape(title),
        ));
    }
    html.push_str("</div>\n");

    html.push_str("<section class=\"wb-diffs\"><h2>Diffs</h2>\n");
    let diff_seq = diffs.as_seq();
    for v in &views {
        let r = &records[v.idx];
        append_card(&mut html, v, r, diff_seq.and_then(|s| s.get(v.idx)));
    }
    html.push_str("</section></main></div>\n");
    append_echarts(&mut html, &views, &chart_configs);
    html.push_str("</body></html>\n");

    let index_path = out_dir.join("index.html");
    atomic(&index_path, html.as_bytes())?;
    Ok(index_path)
}

fn select_runs(
    records: &[Record],
    has_full_vocab: bool,
    nll_keys: &[&str],
    wall_keys: &[&str],
) -> Vec<RunView> {
    let has_deep_runs = records
        .iter()
        .any(|r| plot::extract_series(r, false, nll_keys).is_some_and(|s| s.len() >= 10));
    let mut views = Vec::new();
    let mut seen_sigs = Vec::new();
    for (idx, r) in records.iter().enumerate() {
        let raw_pts = plot::extract_series(r, false, nll_keys);
        let points = raw_pts.as_ref().map_or(0, Vec::len);
        let summary_only = plot::summary_only(r) || (has_deep_runs && points < 8);
        let nll_pts = if summary_only {
            raw_pts
        } else {
            plot::incumbent_series(r, nll_keys)
        };
        if let Some(ref pts) = nll_pts
            && has_full_vocab
            && pts.first().is_some_and(|(_, y)| *y < 11.0)
        {
            continue;
        }
        let first_nll = nll_pts.as_ref().and_then(|s| s.first().map(|p| p.1));
        let last_nll = nll_pts.as_ref().and_then(|s| s.last().map(|p| p.1));
        let delta_nll = match (first_nll, last_nll) {
            (Some(f), Some(l)) => Some(l - f),
            _ => None,
        };
        let wall_pts = plot::extract_series(r, false, wall_keys);
        let wall_ms = wall_pts.as_ref().and_then(|s| {
            if s.is_empty() {
                None
            } else {
                Some(s.iter().map(|(_, w)| *w).sum::<f64>() / s.len() as f64)
            }
        });
        if nll_pts.is_none() && wall_pts.is_none() {
            continue;
        }
        let sig = format!(
            "{}:{}:{:.5}:{:.5}:{}",
            r.optimizer,
            r.recipe,
            first_nll.unwrap_or(0.0),
            last_nll.unwrap_or(0.0),
            points
        );
        if seen_sigs.contains(&sig) {
            continue;
        }
        seen_sigs.push(sig);
        let color = OKABE_ITO[views.len() % OKABE_ITO.len()];
        views.push(RunView {
            idx,
            label: plot::run_label(r),
            color,
            ctx: plot::context_tag(&r.recipe),
            cohort: plot::cohort_tag(r),
            optimizer: r.optimizer.clone(),
            summary_only,
            first_nll,
            last_nll,
            delta_nll,
            wall_ms,
        });
        if views.len() >= 12 {
            break;
        }
    }
    views
}

fn base_chart(metric: &str, x_label: &str, y_label: &str, log_y: bool) -> Chart {
    Chart::new()
        .grid(
            component::Grid::new()
                .left("84")
                .right("24")
                .top("28")
                .bottom("78"),
        )
        .tooltip(
            element::Tooltip::new()
                .trigger(element::Trigger::Axis)
                .axis_pointer(element::AxisPointer::new().type_(element::AxisPointerType::Cross)),
        )
        .legend(component::Legend::new().show(false))
        .toolbox(
            component::Toolbox::new().right("10").top("0").feature(
                component::Feature::new()
                    .data_zoom(component::ToolboxDataZoom::new().y_axis_index("none"))
                    .restore(component::Restore::new())
                    .save_as_image(
                        component::SaveAsImage::new()
                            .type_(component::SaveAsImageType::Svg)
                            .name(format!("baux_{metric}")),
                    ),
            ),
        )
        .data_zoom(
            component::DataZoom::new()
                .type_(component::DataZoomType::Inside)
                .x_axis_index(0)
                .filter_mode(component::FilterMode::Filter),
        )
        .data_zoom(
            component::DataZoom::new()
                .type_(component::DataZoomType::Slider)
                .x_axis_index(0)
                .filter_mode(component::FilterMode::Filter)
                .bottom(8.0),
        )
        .x_axis(
            component::Axis::new()
                .type_(element::AxisType::Log)
                .name(x_label)
                .name_location(element::NameLocation::Middle)
                .name_gap(28.0)
                .scale(true),
        )
        .y_axis(
            component::Axis::new()
                .type_(if log_y {
                    element::AxisType::Log
                } else {
                    element::AxisType::Value
                })
                .name(y_label)
                .name_location(element::NameLocation::Middle)
                .name_gap(60.0)
                .scale(true),
        )
}

#[allow(clippy::too_many_arguments)]
fn charming_chart(
    records: &[Record],
    views: &[RunView],
    metric: &str,
    _title: &str,
    x_label: &str,
    y_label: &str,
    log_y: bool,
    has_full_vocab: bool,
    nll_keys: &[&str],
    val_keys: &[&str],
    wall_keys: &[&str],
) -> Option<String> {
    let mut chart = base_chart(metric, x_label, y_label, log_y);

    let is_step_metric = matches!(metric, "delta_nll" | "nll" | "nll_by_time");
    let is_val_metric = metric == "validation_nll";
    let mut count = 0;

    for v in views {
        if v.summary_only {
            continue;
        }
        let r = &records[v.idx];
        let Some(pts) = plot::comparison_curve(
            r,
            metric,
            has_full_vocab,
            false,
            nll_keys,
            val_keys,
            wall_keys,
        ) else {
            continue;
        };
        if pts.is_empty() {
            continue;
        }
        let data: Vec<Vec<f64>> = pts.into_iter().map(|(x, y)| vec![x, y]).collect();
        let mut line = series::Line::new()
            .name(v.label.clone())
            .show_symbol(is_val_metric || data.len() <= 12)
            .symbol(element::Symbol::Circle)
            .symbol_size(if is_val_metric { 6.0 } else { 4.0 })
            .line_style(element::LineStyle::new().width(2.4).color(v.color))
            .item_style(element::ItemStyle::new().color(v.color))
            .data(data);
        if is_step_metric {
            line = line.step(element::Step::End);
        }
        chart = chart.series(line);
        count += 1;

        if metric == "nll"
            && let Some(cand_pts) = plot::candidate_series(r)
            && cand_pts.len() >= 5
        {
            let cand_data: Vec<Vec<f64>> = cand_pts.into_iter().map(|(x, y)| vec![x, y]).collect();
            let scatter = series::Scatter::new()
                .name(format!("{} [proposals]", v.label))
                .symbol(element::Symbol::Circle)
                .symbol_size(3.5)
                .item_style(element::ItemStyle::new().color(v.color).opacity(0.24))
                .data(cand_data);
            chart = chart.series(scatter);
        }
    }
    if count == 0 {
        None
    } else {
        Some(chart.to_string())
    }
}

fn summary_chart(views: &[RunView]) -> Option<String> {
    let mut categories = Vec::new();
    let mut gains = Vec::new();
    for v in views {
        let Some(d) = v.delta_nll else {
            continue;
        };
        if v.optimizer == "muon" || v.last_nll.unwrap_or(12.5) < 12.1 {
            continue;
        }
        let short_label = v.label.replace("TuRBO-ENN ", "");
        categories.push(short_label);
        gains.push((-d).max(0.0));
    }
    if categories.is_empty() {
        return None;
    }
    let chart = Chart::new()
        .grid(
            component::Grid::new()
                .left("16")
                .right("28")
                .top("20")
                .bottom("52"),
        )
        .tooltip(element::Tooltip::new().trigger(element::Trigger::Axis))
        .toolbox(
            component::Toolbox::new().right("10").top("0").feature(
                component::Feature::new().save_as_image(
                    component::SaveAsImage::new()
                        .type_(component::SaveAsImageType::Svg)
                        .name("baux_summary_bar"),
                ),
            ),
        )
        .x_axis(
            component::Axis::new()
                .type_(element::AxisType::Value)
                .name("−ΔNLL")
                .name_location(element::NameLocation::Middle)
                .name_gap(28.0),
        )
        .y_axis(
            component::Axis::new()
                .type_(element::AxisType::Category)
                .data(categories),
        )
        .series(
            series::Bar::new()
                .name("−ΔNLL")
                .item_style(element::ItemStyle::new().color("#0072B2"))
                .data(gains),
        );
    Some(chart.to_string())
}

fn append_sidebar(html: &mut String, views: &[RunView]) {
    html.push_str("<div class=\"wb-shell\">\n<aside class=\"wb-sidebar\">\n");
    html.push_str("<div class=\"wb-brand\"><h1>BAUX Runs</h1></div>\n");
    html.push_str("<div class=\"wb-controls\">\n");

    html.push_str("<div class=\"wb-ctrl-row\">\n");
    html.push_str(
        "<div><div class=\"wb-btn-row\"><button class=\"wb-btn active\" id=\"btn-theme-paper\" \
         onclick=\"wbSetTheme('paper')\">Paper</button><button class=\"wb-btn\" \
         id=\"btn-theme-dark\" onclick=\"wbSetTheme('dark')\">Dark</button></div></div>\n",
    );
    html.push_str(
        "<div><div class=\"wb-btn-row\"><button class=\"wb-btn active\" id=\"btn-x-log\" \
         onclick=\"wbSetXScale('log')\">Log</button><button class=\"wb-btn\" id=\"btn-x-lin\" \
         onclick=\"wbSetXScale('value')\">Lin</button></div></div>\n",
    );
    html.push_str(
        "<div><div class=\"wb-btn-row\"><button class=\"wb-btn active\" id=\"btn-cand\" \
         onclick=\"wbToggleProposals()\">Cloud</button></div></div>\n",
    );
    html.push_str("</div>\n");

    html.push_str("<div class=\"wb-btn-row\">\n");
    html.push_str(
        "<button class=\"wb-btn active\" id=\"btn-f-bo\" onclick=\"wbFilter('bo', \
         this)\">BO</button>\n",
    );
    html.push_str(
        "<button class=\"wb-btn\" id=\"btn-f-1m\" onclick=\"wbFilter('1M', this)\">1M</button>\n",
    );
    html.push_str("<button class=\"wb-btn\" onclick=\"wbFilter('577K', this)\">577K</button>\n");
    html.push_str("<button class=\"wb-btn\" onclick=\"wbFilter('64K', this)\">64K</button>\n");
    html.push_str("<button class=\"wb-btn\" onclick=\"wbFilter('all', this)\">All</button>\n");
    html.push_str("</div>\n");

    html.push_str(
        "<div><div class=\"wb-ctrl-label\">Smooth: <span \
         id=\"wb-smooth-val\">0.00</span></div><input type=\"range\" id=\"wb-smooth\" min=\"0\" \
         max=\"0.95\" step=\"0.05\" value=\"0\" oninput=\"wbSetSmooth(this.value)\" \
         style=\"width:100%\"></div>\n",
    );
    html.push_str("</div>\n");

    html.push_str("<div class=\"wb-run-list\">\n");
    for (i, v) in views.iter().enumerate() {
        if v.summary_only {
            continue;
        }
        sidebar_item(html, i, v);
    }
    if views.iter().any(|v| v.summary_only) {
        html.push_str("<div class=\"wb-sec-hdr\" style=\"margin-top:6px\">Summary</div>\n");
        for (i, v) in views.iter().enumerate() {
            if !v.summary_only {
                continue;
            }
            sidebar_item(html, i, v);
        }
    }
    html.push_str("</div></aside>\n");
}

fn sidebar_item(html: &mut String, i: usize, v: &RunView) {
    let nll_str = match (v.first_nll, v.last_nll) {
        (Some(f), Some(l)) => format!("{f:.3}&rarr;<b>{l:.3}</b>"),
        _ => "n/a".into(),
    };
    let delta_str = match v.delta_nll {
        Some(d) if d < 0.0 => format!("<span class=\"wb-good\">{d:+.4}</span>"),
        Some(d) => format!("<span class=\"wb-bad\">{d:+.4}</span>"),
        None => String::new(),
    };
    let ms_str = v.wall_ms.map(|w| format!("{w:.0}ms")).unwrap_or_default();
    html.push_str(&format!(
        "<div class=\"wb-run-item\" onmouseenter=\"wbHighlight({i}, true)\" \
         onmouseleave=\"wbHighlight({i}, false)\"><label class=\"wb-run-check\"><input \
         type=\"checkbox\" checked data-idx=\"{i}\" onchange=\"wbToggleRun({i}, \
         this.checked)\"><span class=\"wb-swatch\" style=\"background:{}\"></span></label><div \
         class=\"wb-run-info\"><div class=\"wb-run-title\"><span class=\"wb-run-name\">{}</span> \
         <a href=\"#run-diff-{}\" onclick=\"wbOpenDiff({})\" \
         class=\"wb-diff-link\">diff</a></div><div \
         class=\"wb-run-metrics\"><span>{nll_str}</span> {delta_str} \
         <span>{ms_str}</span></div></div></div>\n",
        v.color,
        escape(&v.label),
        v.idx,
        v.idx,
    ));
}

fn append_echarts(html: &mut String, views: &[RunView], chart_configs: &[(&str, &str, String)]) {
    let meta: Vec<Value> = views
        .iter()
        .map(|v| {
            json::json!({
                "label": v.label,
                "ctx": v.ctx,
                "cohort": v.cohort,
                "optimizer": v.optimizer,
                "color": v.color,
                "summary_only": v.summary_only,
                "last_nll": v.last_nll.unwrap_or(12.5),
            })
        })
        .collect();
    let meta_json = json::to_string(&json::json!(meta)).unwrap_or_else(|_| "[]".into());
    html.push_str(&format!(
        "<script src=\"https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js\"></script>\n\
        <script>\n\
        const WB_RUNS = {meta_json};\n\
        let WB_RAW_CHARTS = {{}};\n\
        const WB_INSTANCES = {{}};\n"
    ));
    for (metric, _, cfg) in chart_configs {
        html.push_str(&format!("WB_RAW_CHARTS[{metric:?}] = {cfg};\n"));
    }
    html.push_str(
        r#"
const WB_VISIBLE = WB_RUNS.map(() => true);
let WB_SMOOTH = 0.0;
let WB_THEME = 'paper';
let WB_SHOW_PROPOSALS = true;

function wbStyleOptionForTheme(opt, theme, metric) {
  const isPaper = (theme === 'paper');
  const fg = isPaper ? '#0f172a' : '#e2e8f0';
  const sub = isPaper ? '#334155' : '#94a3b8';
  const gridLine = isPaper ? '#e2e8f0' : '#222738';
  const bg = isPaper ? '#ffffff' : 'transparent';
  opt.backgroundColor = bg;
  opt.textStyle = { fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif', color: fg };
  if (metric === 'summary_bar' && Array.isArray(opt.grid) && opt.grid[0]) {
    opt.grid[0].containLabel = true;
  }
  if (Array.isArray(opt.dataZoom)) {
    opt.dataZoom.forEach(dz => {
      if (dz.type === 'slider') {
        dz.height = 16;
        dz.bottom = 8;
        dz.showDetail = false;
      }
    });
  }
  const styleAxis = (ax, isY) => {
    if (!ax) return;
    ax.nameTextStyle = { color: fg, fontSize: 12, fontWeight: 600 };
    ax.axisLabel = { color: sub, fontSize: 11, margin: 8 };
    if (isY && ax.type === 'value') {
      ax.axisLabel.formatter = (v) => (typeof v === 'number' ? v.toFixed(3) : v);
    }
    ax.axisLine = { show: true, lineStyle: { color: gridLine } };
    ax.splitLine = { show: true, lineStyle: { color: gridLine, type: 'dashed' } };
  };
  styleAxis(opt.xAxis, false);
  styleAxis(opt.yAxis, true);
}

function wbInit() {
  if (typeof echarts === 'undefined') return;
  for (const [metric, rawOpt] of Object.entries(WB_RAW_CHARTS)) {
    if (!rawOpt._origSeries) {
      rawOpt._origSeries = rawOpt.series.map(s => ({
        name: s.name,
        type: s.type,
        step: s.step,
        data: Array.isArray(s.data) ? s.data.map(pt => Array.isArray(pt) ? [pt[0], pt[1]] : pt) : []
      }));
    }
    const el = document.getElementById('chart-' + metric);
    if (!el) continue;
    if (WB_INSTANCES[metric]) {
      WB_INSTANCES[metric].dispose();
    }
    wbStyleOptionForTheme(rawOpt, WB_THEME, metric);
    const chart = echarts.init(el, WB_THEME === 'dark' ? 'dark' : null, { renderer: 'svg' });
    chart.setOption(rawOpt);
    WB_INSTANCES[metric] = chart;
  }
  const syncCharts = Object.entries(WB_INSTANCES)
    .filter(([k]) => k !== 'summary_bar')
    .map(([, c]) => c);
  echarts.connect(syncCharts);
  wbFilter('bo', document.getElementById('btn-f-bo'));
}

function wbSetTheme(theme) {
  WB_THEME = theme;
  document.body.classList.toggle('theme-dark', theme === 'dark');
  document.getElementById('btn-theme-paper').classList.toggle('active', theme === 'paper');
  document.getElementById('btn-theme-dark').classList.toggle('active', theme === 'dark');
  wbInit();
}

function wbApplyVisibility() {
  const selected = {};
  WB_RUNS.forEach((r, i) => {
    selected[r.label] = WB_VISIBLE[i];
    selected[r.label + ' [proposals]'] = WB_VISIBLE[i] && WB_SHOW_PROPOSALS;
  });
  for (const [metric, chart] of Object.entries(WB_INSTANCES)) {
    if (metric === 'summary_bar') continue;
    chart.setOption({ legend: { selected } });
  }
}

function wbToggleProposals() {
  WB_SHOW_PROPOSALS = !WB_SHOW_PROPOSALS;
  const btn = document.getElementById('btn-cand');
  btn.classList.toggle('active', WB_SHOW_PROPOSALS);
  wbApplyVisibility();
}

function wbToggleRun(idx, checked) {
  WB_VISIBLE[idx] = checked;
  wbApplyVisibility();
}

function wbFilter(mode, btn) {
  if (btn) {
    btn.parentElement.querySelectorAll('.wb-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }
  WB_RUNS.forEach((r, i) => {
    if (mode === 'all') WB_VISIBLE[i] = true;
    else if (mode === 'bo') WB_VISIBLE[i] = (r.optimizer !== 'muon' && r.last_nll >= 12.1);
    else WB_VISIBLE[i] = (r.ctx === mode && r.last_nll >= 12.1);
  });
  document.querySelectorAll('.wb-run-check input').forEach((cb, i) => {
    cb.checked = WB_VISIBLE[i];
  });
  wbApplyVisibility();
}

function wbSetXScale(scaleType) {
  document.getElementById('btn-x-log').classList.toggle('active', scaleType === 'log');
  document.getElementById('btn-x-lin').classList.toggle('active', scaleType === 'value');
  for (const [metric, chart] of Object.entries(WB_INSTANCES)) {
    if (metric === 'summary_bar') continue;
    chart.setOption({ xAxis: { type: scaleType, scale: true } });
  }
}

function wbSetSmooth(val) {
  WB_SMOOTH = parseFloat(val) || 0.0;
  document.getElementById('wb-smooth-val').textContent = WB_SMOOTH.toFixed(2);
  for (const [metric, chart] of Object.entries(WB_INSTANCES)) {
    if (metric === 'summary_bar') continue;
    const rawOpt = WB_RAW_CHARTS[metric];
    if (!rawOpt || !rawOpt._origSeries) continue;
    const updated = rawOpt._origSeries.map(s => {
      if (s.type !== 'line' || WB_SMOOTH <= 0.001 || s.data.length <= 2) {
        return { name: s.name, step: s.step, data: s.data };
      }
      let ema = s.data[0][1];
      const smoothed = s.data.map(([x, y], idx) => {
        if (idx === 0) return [x, y];
        ema = WB_SMOOTH * ema + (1.0 - WB_SMOOTH) * y;
        return [x, ema];
      });
      return { name: s.name, step: false, data: smoothed };
    });
    chart.setOption({ series: updated });
  }
}

function wbHighlight(idx, on) {
  const name = WB_RUNS[idx]?.label;
  if (!name) return;
  for (const [metric, chart] of Object.entries(WB_INSTANCES)) {
    if (metric === 'summary_bar') continue;
    chart.dispatchAction({ type: on ? 'highlight' : 'downplay', seriesName: name });
  }
}

function wbOpenDiff(idx) {
  const card = document.getElementById('run-diff-' + idx);
  if (!card) return;
  const det = card.querySelector('details');
  if (det) det.open = true;
}

window.addEventListener('resize', () => {
  Object.values(WB_INSTANCES).forEach(c => c.resize());
});
window.addEventListener('DOMContentLoaded', wbInit);
</script>
"#,
    );
}

/// Serve the comparison dashboard and SVGs on `http://127.0.0.1:<port>/`.
pub fn serve(root: &Path, port: u16) -> Result<(), String> {
    let out_dir = directory(root)?.join("comparisons");
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    let addr = listener.local_addr().map_err(|e| e.to_string())?;
    println!("http://{addr}/");
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else {
            continue;
        };
        let mut buf = [0u8; 4096];
        let n = stream.read(&mut buf).unwrap_or(0);
        let req = String::from_utf8_lossy(&buf[..n]);
        let req_path = req
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().nth(1))
            .unwrap_or("/");
        let clean = req_path
            .split('?')
            .next()
            .unwrap_or("/")
            .trim_start_matches('/');
        let (status, content_type, body) = if clean.is_empty() || clean == "index.html" {
            match fs::read(out_dir.join("index.html")) {
                Ok(b) => ("200 OK", "text/html; charset=utf-8", b),
                Err(_) => ("404 Not Found", "text/plain", b"Not Found".to_vec()),
            }
        } else if clean.ends_with(".svg") && !clean.contains("..") && !clean.contains('/') {
            match fs::read(out_dir.join(clean)) {
                Ok(b) => ("200 OK", "image/svg+xml", b),
                Err(_) => ("404 Not Found", "text/plain", b"Not Found".to_vec()),
            }
        } else {
            ("404 Not Found", "text/plain", b"Not Found".to_vec())
        };
        let header = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: \
             {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
            body.len()
        );
        let _ = stream.write_all(header.as_bytes());
        let _ = stream.write_all(&body);
    }
    Ok(())
}

fn append_traces(html: &mut String, seq: &[Value]) {
    if seq.is_empty() {
        return;
    }
    html.push_str("<div class=\"traces\">");
    for item in seq {
        let y = field(item, "y");
        let pts = item["points"].as_u64().unwrap_or(0);
        let f = item["first"]
            .as_seq()
            .and_then(|a| a.get(1))
            .and_then(|v| v.as_f64());
        let l = item["last"]
            .as_seq()
            .and_then(|a| a.get(1))
            .and_then(|v| v.as_f64());
        if let (Some(f), Some(l)) = (f, l) {
            html.push_str(&format!(
                "<span class=\"trace-pill\"><b>{}</b>: {:.6} &rarr; {:.6} ({} pts)</span>",
                escape(y),
                f,
                l,
                pts
            ));
        }
    }
    html.push_str("</div>");
}

fn append_patch(html: &mut String, patch: &str) {
    if patch.is_empty() {
        return;
    }
    html.push_str("<pre class=\"diff\">");
    for line in patch.lines() {
        let cls = if line.starts_with("+++") || line.starts_with("---") || line.starts_with("diff ")
        {
            "diff-hdr"
        } else if line.starts_with('+') {
            "diff-add"
        } else if line.starts_with('-') {
            "diff-del"
        } else if line.starts_with("@@") {
            "diff-hunk"
        } else {
            ""
        };
        if cls.is_empty() {
            html.push_str(&escape(line));
        } else {
            html.push_str(&format!("<span class=\"{cls}\">{}</span>", escape(line)));
        }
        html.push('\n');
    }
    html.push_str("</pre>");
}

fn append_diff(html: &mut String, d: &Value) {
    let from = field(d, "compared_from_commit");
    let files: Vec<&str> = d["optimizer_files_changed"]
        .as_seq()
        .map(|s| s.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    let patch = field(d, "optimizer_patch");
    if files.is_empty() && patch.is_empty() {
        return;
    }
    html.push_str(&format!(
        "<details><summary>JJ Optimizer Code Diff from <code>{}</code> ({} files \
         changed)</summary>",
        escape(from.get(..8).unwrap_or(from)),
        files.len()
    ));
    if !files.is_empty() {
        html.push_str(&format!(
            "<div class=\"file-list\">{}</div>",
            escape(&files.join(", "))
        ));
    }
    append_patch(html, patch);
    html.push_str("</details>");
}

fn append_card(html: &mut String, v: &RunView, r: &Record, diff: Option<&Value>) {
    let short_commit = r.commit.get(..12).unwrap_or(&r.commit);
    let short_change = r.change.get(..12).unwrap_or(&r.change);
    html.push_str(&format!(
        "<div class=\"run-card\" id=\"run-diff-{}\"><div class=\"run-head\"><span \
         class=\"wb-swatch\" style=\"background:{}\"></span><strong>{}</strong> <span \
         class=\"badge\">{}</span> <span class=\"mono\">commit {} ({})</span></div>",
        v.idx,
        v.color,
        escape(&v.label),
        escape(v.cohort),
        escape(short_commit),
        escape(short_change),
    ));
    html.push_str(&format!(
        "<div class=\"run-meta\">artifacts: <code>{}</code></div>",
        escape(&r.output)
    ));
    if let Some(seq) = r.measurements.as_seq() {
        append_traces(html, seq);
    }
    if let Some(d) = diff {
        append_diff(html, d);
    }
    html.push_str("</div>\n");
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

const HEADER: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BAUX — Publication &amp; W&amp;B Run Explorer</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif; background: #f8fafc; color: #0f172a; margin: 0; }
  body.theme-dark { background: #0b0d13; color: #e2e8f0; }
  .wb-shell { display: flex; min-height: 100vh; }
  .wb-sidebar { width: 356px; min-width: 356px; background: #ffffff; border-right: 1px solid #e2e8f0; display: flex; flex-direction: column; height: 100vh; position: sticky; top: 0; }
  body.theme-dark .wb-sidebar { background: #11141d; border-right-color: #222738; }
  .wb-brand { padding: 14px 16px; border-bottom: 1px solid #e2e8f0; }
  body.theme-dark .wb-brand { border-bottom-color: #222738; }
  .wb-brand h1 { margin: 0 0 4px 0; font-size: 15px; color: #0f172a; }
  body.theme-dark .wb-brand h1 { color: #f8fafc; }
  .wb-sub { font-size: 11px; color: #64748b; }
  .wb-controls { padding: 12px 16px; border-bottom: 1px solid #e2e8f0; display: flex; flex-direction: column; gap: 9px; }
  body.theme-dark .wb-controls { border-bottom-color: #222738; }
  .wb-ctrl-label { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; font-weight: 700; }
  .wb-ctrl-row { display: grid; grid-template-columns: repeat(3, auto); justify-content: space-between; align-items: end; gap: 8px; }
  .wb-btn-row { display: flex; flex-wrap: wrap; gap: 5px; }
  .wb-btn { background: #f1f5f9; border: 1px solid #cbd5e1; color: #334155; border-radius: 5px; padding: 4px 8px; font-size: 11px; cursor: pointer; font-weight: 600; line-height: 1.2; }
  body.theme-dark .wb-btn { background: #1e2436; border-color: #2e364f; color: #cbd5e1; }
  .wb-btn.active { background: #0072B2; border-color: #0072B2; color: #ffffff; }
  .wb-sec-hdr { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; padding: 4px 4px 2px 4px; }
  .wb-run-list { flex: 1; overflow-y: auto; padding: 8px 10px; display: flex; flex-direction: column; gap: 6px; }
  .wb-run-item { display: flex; gap: 8px; padding: 8px 10px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; align-items: flex-start; }
  body.theme-dark .wb-run-item { background: #161a26; border-color: #23293d; }
  .wb-run-item:hover { border-color: #0072B2; }
  .wb-run-check { display: flex; align-items: center; gap: 6px; cursor: pointer; margin-top: 2px; }
  .wb-swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; flex-shrink: 0; }
  .wb-run-info { flex: 1; min-width: 0; }
  .wb-run-title { display: flex; align-items: center; gap: 6px; font-size: 11px; margin-bottom: 3px; }
  .wb-ctx { background: #e0f2fe; color: #0369a1; padding: 1px 5px; border-radius: 3px; font-family: ui-monospace, monospace; font-weight: 700; font-size: 10px; }
  body.theme-dark .wb-ctx { background: #1e293b; color: #38bdf8; }
  .wb-mod-sum { background: #fef3c7; color: #92400e; padding: 1px 5px; border-radius: 3px; font-size: 10px; font-weight: 600; }
  body.theme-dark .wb-mod-sum { background: #3f2e12; color: #fbbf24; }
  .wb-diff-link { margin-left: auto; color: #0072B2; text-decoration: none; font-family: ui-monospace, monospace; font-size: 11px; font-weight: 600; }
  body.theme-dark .wb-diff-link { color: #60a5fa; }
  .wb-run-name { font-size: 12px; color: #0f172a; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  body.theme-dark .wb-run-name { color: #f1f5f9; }
  .wb-run-metrics { font-size: 11px; font-family: ui-monospace, monospace; font-variant-numeric: tabular-nums; color: #475569; margin-top: 4px; display: grid; grid-template-columns: 1fr auto auto; gap: 8px; align-items: baseline; }
  body.theme-dark .wb-run-metrics { color: #cbd5e1; }
  .wb-good { color: #009E73; font-weight: 700; }
  .wb-bad { color: #D55E00; font-weight: 700; }
  .wb-main { flex: 1; padding: 16px 22px; overflow-y: auto; }
  .wb-ontology { background: #ffffff; border: 1px solid #cbd5e1; border-left: 4px solid #0072B2; border-radius: 6px; padding: 10px 14px; margin-bottom: 16px; font-size: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
  body.theme-dark .wb-ontology { background: #131722; border-color: #222738; border-left-color: #38bdf8; }
  .wb-onto-right { color: #475569; font-size: 11px; }
  body.theme-dark .wb-onto-right { color: #94a3b8; }
  .wb-charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 16px; }
  .wb-panel { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
  body.theme-dark .wb-panel { background: #131722; border-color: #222738; box-shadow: none; }
  .wb-panel-head { display: flex; justify-content: space-between; align-items: flex-start; padding: 0 4px 6px 4px; border-bottom: 1px solid #f1f5f9; margin-bottom: 6px; }
  body.theme-dark .wb-panel-head { border-bottom-color: #1e2436; }
  .wb-fig-title { font-size: 13px; color: #0f172a; font-weight: 700; }
  body.theme-dark .wb-fig-title { color: #f8fafc; }
  .wb-fig-sub { font-size: 11px; color: #64748b; margin-top: 2px; }
  .wb-hint { font-size: 10px; color: #64748b; font-family: ui-monospace, monospace; white-space: nowrap; }
  .wb-chart { width: 100%; height: 370px; }
  .wb-diffs h2 { margin: 26px 0 12px 0; font-size: 16px; border-bottom: 1px solid #cbd5e1; padding-bottom: 8px; }
  body.theme-dark .wb-diffs h2 { border-bottom-color: #222738; }
  .run-card { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; }
  body.theme-dark .run-card { background: #131722; border-color: #222738; }
  .run-head { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; font-size: 14px; }
  .badge { background: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-family: ui-monospace, monospace; font-weight: 600; }
  body.theme-dark .badge { background: #1e293b; color: #38bdf8; }
  .mono { color: #64748b; font-size: 12px; font-family: ui-monospace, monospace; }
  .run-meta { margin-top: 6px; font-size: 12px; color: #64748b; }
  .traces { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
  .trace-pill { background: #f1f5f9; border: 1px solid #cbd5e1; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-family: ui-monospace, monospace; }
  body.theme-dark .trace-pill { background: #1a2030; border-color: #2a334d; color: #cbd5e1; }
  details { margin-top: 10px; border-top: 1px solid #e2e8f0; padding-top: 8px; }
  body.theme-dark details { border-top-color: #222736; }
  summary { cursor: pointer; font-size: 13px; color: #0072B2; font-weight: 600; }
  body.theme-dark summary { color: #60a5fa; }
  .file-list { font-size: 12px; font-family: ui-monospace, monospace; color: #64748b; margin: 6px 0; }
  pre.diff { background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 10px; overflow-x: auto; font-size: 12px; line-height: 1.45; font-family: ui-monospace, monospace; color: #e2e8f0; margin: 6px 0 0 0; }
  .diff-hdr { color: #94a3b8; font-weight: bold; }
  .diff-hunk { color: #38bdf8; }
  .diff-add { color: #4ade80; }
  .diff-del { color: #f87171; }
</style>
</head>
<body>
"#;
