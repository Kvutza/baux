use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};

use charming::Chart;
use charming::{component, element, series};

use baux::artifact::json::Value;
use baux::artifact::runs::plot;
use baux::artifact::runs::{Record, atomic, directory};

#[path = "dashboard/html.rs"]
mod html;
use html::{HEADER, RunView, append_card, append_echarts, append_sidebar, escape};

/// Colorblind-safe Okabe-Ito publication palette.
const OKABE_ITO: &[&str] = &[
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#332288", "#882255",
    "#117733", "#999933", "#44AA99", "#AA4499",
];

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
