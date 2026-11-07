//! BAUX Paper Crawler & Deduplication Engine (backed by pgrust)
//!
//! Zero Python, 100% memory-safe Rust using only stdlib.
//! Compiles with `rustc -O tools/paper-crawl.rs -o tools/bin/paper-crawl`

use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug, Clone)]
pub struct MathBlock {
    pub kind: String,
    pub name: Option<String>,
    pub label: Option<String>,
    pub statement: String,
    pub proof: Option<String>,
    pub file_path: String,
    pub line_number: usize,
}

fn pg_sql(sql: &str) -> Result<String, String> {
    let pgrustw = Path::new("tools/pgrustw");
    if !pgrustw.exists() {
        return Err("tools/pgrustw not found".to_string());
    }

    let mut child = Command::new(pgrustw)
        .arg("psql")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn pgrustw: {e}"))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(sql.as_bytes())
            .map_err(|e| format!("Failed to write SQL to psql: {e}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("Failed to wait for psql: {e}"))?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(format!("SQL execution error: {err}"));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn sql_escape(s: &str) -> String {
    s.replace('\'', "''")
}

fn clean_arxiv(id: &str) -> Option<String> {
    let trimmed = id.trim();
    let cleaned = trimmed
        .trim_start_matches("arxiv:")
        .trim_start_matches("arXiv:");
    // Match YYMM.NNNNN or YYMM.NNNN
    let parts: Vec<&str> = cleaned.split('.').collect();
    if parts.len() == 2 {
        let yymm = parts[0];
        let rest = parts[1].split('v').next().unwrap_or(parts[1]); // strip version v1, v2
        if (yymm.len() == 4)
            && yymm.chars().all(|c| c.is_ascii_digit())
            && (rest.len() == 4 || rest.len() == 5)
            && rest.chars().all(|c| c.is_ascii_digit())
        {
            return Some(format!("{yymm}.{rest}"));
        }
    }
    None
}

fn extract_ids(content: &str) -> HashSet<String> {
    let mut ids = HashSet::new();
    for line in content.lines() {
        let bytes = line.as_bytes();
        let len = bytes.len();
        if len < 9 {
            continue;
        }
        for i in 0..len {
            if bytes[i] == b'.' && i >= 4 && i + 4 < len {
                let prefix = &bytes[i - 4..i];
                if prefix.iter().all(|b| b.is_ascii_digit()) {
                    let mut end = i + 1;
                    while end < len && bytes[end].is_ascii_digit() && (end - (i + 1)) < 5 {
                        end += 1;
                    }
                    let num_digits = end - (i + 1);
                    if num_digits == 4 || num_digits == 5 {
                        if let Ok(cand) = std::str::from_utf8(&bytes[i - 4..end]) {
                            if let Some(valid) = clean_arxiv(cand) {
                                ids.insert(valid);
                            }
                        }
                    }
                }
            }
        }
    }
    ids
}

fn extract_theorems(content: &str, file_path: &str) -> Vec<MathBlock> {
    let targets = [
        "theorem",
        "lemma",
        "proposition",
        "prop",
        "corollary",
        "definition",
        "defn",
        "conjecture",
        "assumption",
    ];
    let lines: Vec<&str> = content.lines().collect();
    let num_lines = lines.len();
    let mut blocks = Vec::new();
    let mut i = 0;

    while i < num_lines {
        let line = lines[i].trim();
        // Check for \begin{target}
        let mut matched_kind = None;
        for t in &targets {
            let pattern = format!("\\begin{{{t}}}");
            if line.contains(&pattern) {
                matched_kind = Some((*t).to_string());
                break;
            }
        }

        if let Some(kind) = matched_kind {
            let start_line = i + 1;
            let end_pattern = format!("\\end{{{kind}}}");
            let mut name = None;
            let mut label = None;
            let mut stmt_lines = Vec::new();

            // Extract name from [Bracket] if present on begin line
            if let Some(open) = line.find('[') {
                if let Some(close) = line[open..].find(']') {
                    name = Some(line[open + 1..open + close].trim().to_string());
                }
            }

            // Collect statement lines
            let mut found_end = false;
            while i < num_lines {
                let cur = lines[i];
                if cur.contains("\\label{") {
                    if let Some(l_start) = cur.find("\\label{") {
                        if let Some(l_end) = cur[l_start + 7..].find('}') {
                            label = Some(cur[l_start + 7..l_start + 7 + l_end].trim().to_string());
                        }
                    }
                }
                if cur.contains(&end_pattern) {
                    // Collect up to end pattern
                    if let Some(pos) = cur.find(&end_pattern) {
                        stmt_lines.push(&cur[..pos]);
                    }
                    found_end = true;
                    i += 1;
                    break;
                } else {
                    stmt_lines.push(cur);
                }
                i += 1;
            }

            if found_end {
                let statement = stmt_lines.join("\n").trim().to_string();

                // Look ahead for \begin{proof} ... \end{proof}
                let mut proof = None;
                let mut p_idx = i;
                // skip comments and empty lines
                while p_idx < num_lines
                    && (lines[p_idx].trim().is_empty() || lines[p_idx].trim().starts_with('%'))
                {
                    p_idx += 1;
                }

                if p_idx < num_lines && lines[p_idx].contains("\\begin{proof}") {
                    let mut proof_lines = Vec::new();
                    while p_idx < num_lines {
                        let p_cur = lines[p_idx];
                        if p_cur.contains("\\end{proof}") {
                            if let Some(pos) = p_cur.find("\\end{proof}") {
                                proof_lines.push(&p_cur[..pos]);
                            }
                            i = p_idx + 1;
                            break;
                        } else {
                            proof_lines.push(p_cur);
                        }
                        p_idx += 1;
                    }
                    let p_str = proof_lines.join("\n").trim().to_string();
                    if !p_str.is_empty() {
                        proof = Some(p_str);
                    }
                }

                blocks.push(MathBlock {
                    kind,
                    name,
                    label,
                    statement,
                    proof,
                    file_path: file_path.to_string(),
                    line_number: start_line,
                });
                continue;
            }
        }
        i += 1;
    }

    blocks
}

fn download_arxiv(id: &str, out_dir: &Path) -> Result<bool, String> {
    if out_dir.exists()
        && fs::read_dir(out_dir)
            .map(|mut d| d.next().is_some())
            .unwrap_or(false)
    {
        return Ok(false);
    }

    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let tar_path = format!("/tmp/crawl_{id}.tar.gz");

    let status = Command::new("curl")
        .args([
            "-sL",
            &format!("https://arxiv.org/src/{id}"),
            "-o",
            &tar_path,
        ])
        .status()
        .map_err(|e| e.to_string())?;

    if !status.success() {
        return Err(format!("curl failed for {id}"));
    }

    let tar_status = Command::new("tar")
        .args(["-xzf", &tar_path, "-C", out_dir.to_str().unwrap()])
        .status();

    let success = match tar_status {
        Ok(s) if s.success() => true,
        _ => {
            let out_file = out_dir.join("main.tex");
            let gz_out = Command::new("gunzip")
                .args(["-c", &tar_path])
                .stdout(fs::File::create(&out_file).map_err(|e| e.to_string())?)
                .status();
            gz_out.map(|s| s.success()).unwrap_or(false)
        }
    };

    let _ = fs::remove_file(&tar_path);
    if success {
        Ok(true)
    } else {
        Err(format!("Failed to unpack source archive for {id}"))
    }
}

fn fetch_linxiv(id: &str) -> (String, String, Option<i32>, String) {
    let linxiv = Path::new("tools/bin/linxiv-cli");
    if !linxiv.exists() {
        return (format!("arXiv:{id}"), String::new(), None, String::new());
    }

    let output = Command::new(linxiv).args(["fetch", id]).output();

    if let Ok(out) = output {
        let text = String::from_utf8_lossy(&out.stdout);
        let mut title = String::new();
        let mut authors = String::new();
        let mut year = None;
        let mut abstract_text = String::new();

        let mut in_abstract = false;
        for line in text.lines() {
            if line.starts_with("# ") && title.is_empty() {
                title = line.trim_start_matches("# ").trim().to_string();
            } else if line.starts_with("Authors:") {
                authors = line.trim_start_matches("Authors:").trim().to_string();
            } else if line.starts_with("Published:") {
                let pub_date = line.trim_start_matches("Published:").trim();
                if let Some(y_str) = pub_date.split('-').next() {
                    year = y_str.parse::<i32>().ok();
                }
            } else if line.starts_with("## Abstract") {
                in_abstract = true;
            } else if in_abstract {
                abstract_text.push_str(line);
                abstract_text.push(' ');
            }
        }

        if !title.is_empty() {
            return (title, authors, year, abstract_text.trim().to_string());
        }
    }

    (format!("arXiv:{id}"), String::new(), None, String::new())
}

fn parse_theorems(out_dir: &Path) -> Vec<MathBlock> {
    let mut blocks = Vec::new();
    fn visit(dir: &Path, blocks: &mut Vec<MathBlock>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    visit(&p, blocks);
                } else if p.extension().and_then(|s| s.to_str()) == Some("tex") {
                    if let Ok(content) = fs::read_to_string(&p) {
                        let path_str = p.to_string_lossy().to_string();
                        let found = extract_theorems(&content, &path_str);
                        blocks.extend(found);
                    }
                }
            }
        }
    }
    visit(out_dir, &mut blocks);
    blocks
}

fn ingest_paper(id: &str, depth: i32) -> Result<HashSet<String>, String> {
    let clean_id = clean_arxiv(id).ok_or_else(|| format!("Invalid arxiv ID: {id}"))?;
    println!(">>> Processing paper: {clean_id} (depth {depth})");

    let out_dir = PathBuf::from("papers/library").join(&clean_id);
    let downloaded = download_arxiv(&clean_id, &out_dir)?;
    if downloaded {
        println!(
            "    Downloaded and unpacked source bundle to {}",
            out_dir.display()
        );
    } else {
        println!("    Source already present at {}", out_dir.display());
    }

    // Metadata
    let (title, authors, year, abstract_txt) = fetch_linxiv(&clean_id);
    let year_val = year
        .map(|y| y.to_string())
        .unwrap_or_else(|| "NULL".to_string());

    // Scan for citations
    let mut cited_ids = HashSet::new();
    let mut has_bib = false;
    let mut figure_count = 0;

    if out_dir.exists() {
        if let Ok(entries) = fs::read_dir(&out_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    if ext == "bbl" || ext == "bib" {
                        has_bib = true;
                        if let Ok(content) = fs::read_to_string(&path) {
                            for cid in extract_ids(&content) {
                                if cid != clean_id {
                                    cited_ids.insert(cid);
                                }
                            }
                        }
                    } else if ext == "tex" {
                        if let Ok(content) = fs::read_to_string(&path) {
                            if content.contains("\\begin{thebibliography}") {
                                has_bib = true;
                                for cid in extract_ids(&content) {
                                    if cid != clean_id {
                                        cited_ids.insert(cid);
                                    }
                                }
                            }
                        }
                    } else if ext == "pdf" || ext == "eps" || ext == "png" {
                        figure_count += 1;
                    }
                }
            }
        }
    }

    // Extract theorems & proofs
    let math_blocks = parse_theorems(&out_dir);
    println!(
        "    Extracted {} unique cited arXiv IDs, {} mathematical theorems/definitions \
         (figures={figure_count})",
        cited_ids.len(),
        math_blocks.len()
    );

    // Insert into pgrust
    let mut sql = format!(
        "INSERT INTO papers (arxiv_id, title, authors, year, abstract, tex_source_path, \
         has_source, has_bib, figure_count) VALUES ('{clean_id}', '{title_esc}', '{authors_esc}', \
         {year_val}, '{abs_esc}', '{path_esc}', TRUE, {has_bib}, {figure_count}) ON CONFLICT \
         (arxiv_id) DO UPDATE SET title = EXCLUDED.title, authors = EXCLUDED.authors, has_source \
         = TRUE, has_bib = EXCLUDED.has_bib, figure_count = EXCLUDED.figure_count;\n",
        title_esc = sql_escape(&title),
        authors_esc = sql_escape(&authors),
        abs_esc = sql_escape(&abstract_txt),
        path_esc = sql_escape(&out_dir.display().to_string()),
    );

    for cid in &cited_ids {
        sql.push_str(&format!(
            "INSERT INTO citations (source_arxiv_id, target_arxiv_id) VALUES ('{clean_id}', \
             '{cid}') ON CONFLICT DO NOTHING;\n"
        ));
        sql.push_str(&format!(
            "INSERT INTO crawl_frontier (arxiv_id, depth, discovered_from, status) VALUES \
             ('{cid}', {next_depth}, '{clean_id}', 'pending') ON CONFLICT (arxiv_id) DO NOTHING;\n",
            next_depth = depth + 1
        ));
    }

    // Delete prior parsed theorems for this paper so re-ingest doesn't
    // duplicate
    sql.push_str(&format!(
        "DELETE FROM theorems WHERE arxiv_id = '{clean_id}';\n"
    ));

    for b in &math_blocks {
        let name_val = b
            .name
            .as_ref()
            .map(|n| format!("'{}'", sql_escape(n)))
            .unwrap_or_else(|| "NULL".to_string());
        let label_val = b
            .label
            .as_ref()
            .map(|l| format!("'{}'", sql_escape(l)))
            .unwrap_or_else(|| "NULL".to_string());
        let proof_val = b
            .proof
            .as_ref()
            .map(|p| format!("'{}'", sql_escape(p)))
            .unwrap_or_else(|| "NULL".to_string());
        sql.push_str(&format!(
            "INSERT INTO theorems (arxiv_id, kind, name, label, statement, proof, file_path, \
             line_number) VALUES ('{clean_id}', '{kind}', {name_val}, {label_val}, '{stmt}', \
             {proof_val}, '{file}', {line});\n",
            kind = sql_escape(&b.kind),
            stmt = sql_escape(&b.statement),
            file = sql_escape(&b.file_path),
            line = b.line_number,
        ));
    }

    sql.push_str(&format!(
        "UPDATE crawl_frontier SET status = 'downloaded' WHERE arxiv_id = '{clean_id}';\n"
    ));

    pg_sql(&sql)?;
    Ok(cited_ids)
}

fn extract_all() -> Result<(), String> {
    println!("=== Parsing Mathematical Blocks Across Downloaded Papers ===");
    let lib_dir = Path::new("papers/library");
    if !lib_dir.exists() {
        return Ok(());
    }

    let mut total_blocks = 0;
    if let Ok(entries) = fs::read_dir(lib_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(id) = path.file_name().and_then(|s| s.to_str()) {
                    if clean_arxiv(id).is_some() {
                        let blocks = parse_theorems(&path);
                        if !blocks.is_empty() {
                            println!("[{id}] Found {} mathematical blocks", blocks.len());
                            let mut sql =
                                format!("DELETE FROM theorems WHERE arxiv_id = '{id}';\n");
                            for b in &blocks {
                                let name_val = b
                                    .name
                                    .as_ref()
                                    .map(|n| format!("'{}'", sql_escape(n)))
                                    .unwrap_or_else(|| "NULL".to_string());
                                let label_val = b
                                    .label
                                    .as_ref()
                                    .map(|l| format!("'{}'", sql_escape(l)))
                                    .unwrap_or_else(|| "NULL".to_string());
                                let proof_val = b
                                    .proof
                                    .as_ref()
                                    .map(|p| format!("'{}'", sql_escape(p)))
                                    .unwrap_or_else(|| "NULL".to_string());
                                sql.push_str(&format!(
                                    "INSERT INTO theorems (arxiv_id, kind, name, label, \
                                     statement, proof, file_path, line_number) VALUES ('{id}', \
                                     '{kind}', {name_val}, {label_val}, '{stmt}', {proof_val}, \
                                     '{file}', {line});\n",
                                    kind = sql_escape(&b.kind),
                                    stmt = sql_escape(&b.statement),
                                    file = sql_escape(&b.file_path),
                                    line = b.line_number,
                                ));
                            }
                            pg_sql(&sql)?;
                            total_blocks += blocks.len();
                        }
                    }
                }
            }
        }
    }
    println!(
        "=== Ingested {total_blocks} mathematical theorems, lemmas, and definitions into pgrust \
         ==="
    );
    Ok(())
}

fn find_theorems(query: &str) -> Result<(), String> {
    let sql = format!(
        "SELECT t.arxiv_id, p.title, t.kind, COALESCE(t.name, t.label, 'unnamed') as name, \
         LEFT(t.statement, 200) as snippet, t.file_path, t.line_number FROM theorems t JOIN \
         papers p ON t.arxiv_id = p.arxiv_id WHERE t.statement ILIKE '%{esc}%' OR t.name ILIKE \
         '%{esc}%' OR t.kind ILIKE '%{esc}%' ORDER BY t.arxiv_id, t.line_number LIMIT 20;",
        esc = sql_escape(query)
    );
    let res = pg_sql(&sql)?;
    println!("{res}");
    Ok(())
}

fn get_theorem(id_or_label: &str) -> Result<(), String> {
    let sql = format!(
        "SELECT t.id, t.arxiv_id, p.title, t.kind, t.name, t.label, t.file_path, t.line_number, \
         t.statement, t.proof FROM theorems t JOIN papers p ON t.arxiv_id = p.arxiv_id WHERE \
         t.label = '{esc}' OR t.id::text = '{esc}' OR t.name ILIKE '%{esc}%' LIMIT 1;",
        esc = sql_escape(id_or_label)
    );
    let res = pg_sql(&sql)?;
    println!("{res}");
    Ok(())
}

fn cmd_status() -> Result<(), String> {
    let sql = "
    SELECT 
        (SELECT count(*) FROM papers) AS total_papers,
        (SELECT count(*) FROM papers WHERE has_source = true) AS papers_with_source,
        (SELECT count(*) FROM citations) AS total_citations,
        (SELECT count(*) FROM theorems) AS total_theorems,
        (SELECT count(*) FROM crawl_frontier WHERE status = 'pending') AS pending_frontier,
        (SELECT count(*) FROM crawl_frontier WHERE status = 'downloaded') AS downloaded_frontier;
    ";
    let res = pg_sql(sql)?;
    println!("{res}");

    let kinds_sql = "
    SELECT kind, count(*) AS count
    FROM theorems
    GROUP BY kind
    ORDER BY count DESC;
    ";
    println!("--- Mathematical Breakdown in Literature Corpus ---");
    let kinds = pg_sql(kinds_sql)?;
    println!("{kinds}");
    Ok(())
}

fn cmd_crawl(max_papers: usize, max_depth: i32) -> Result<(), String> {
    println!("=== Starting Recursive Crawl (limit: {max_papers}, max depth: {max_depth}) ===");

    for step in 0..max_papers {
        let pick_sql = format!(
            "SELECT arxiv_id, depth FROM crawl_frontier WHERE status = 'pending' AND depth <= \
             {max_depth} ORDER BY depth ASC, discovered_at ASC LIMIT 1;"
        );
        let out = pg_sql(&pick_sql)?;
        let lines: Vec<&str> = out
            .lines()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let mut target_id = None;
        let mut target_depth = 0;
        for line in &lines {
            if line.contains('|') && !line.contains("arxiv_id") && !line.starts_with('-') {
                let parts: Vec<&str> = line.split('|').map(|p| p.trim()).collect();
                if parts.len() >= 2 {
                    if let Some(clean) = clean_arxiv(parts[0]) {
                        target_id = Some(clean);
                        target_depth = parts[1].parse::<i32>().unwrap_or(1);
                        break;
                    }
                }
            }
        }

        let id = match target_id {
            Some(id) => id,
            None => {
                println!("Frontier exhausted or no pending papers within depth {max_depth}. Done!");
                break;
            }
        };

        if let Some(prefix) = id.get(0..2).and_then(|p| p.parse::<u32>().ok()) {
            if prefix < 15 && prefix > 30 {
                println!("Skipping {id} (outside 2015-2025 window, prefix: {prefix})");
                let skip_sql = format!(
                    "UPDATE crawl_frontier SET status = 'out_of_scope' WHERE arxiv_id = '{id}';"
                );
                let _ = pg_sql(&skip_sql);
                continue;
            }
        }

        println!(
            "[{}/{}] Crawling frontier paper: {id}",
            step + 1,
            max_papers
        );
        match ingest_paper(&id, target_depth) {
            Ok(_) => (),
            Err(e) => {
                eprintln!("    Failed to ingest {id}: {e}");
                let err_sql = format!(
                    "UPDATE crawl_frontier SET status = 'failed', last_error = '{esc}' WHERE \
                     arxiv_id = '{id}';",
                    esc = sql_escape(&e)
                );
                let _ = pg_sql(&err_sql);
            }
        }
    }

    cmd_status()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: paper-crawl <ingest-seed <id> | crawl [--limit N] [--depth D] | \
             parse-theorems | find <query> | get <label|id> | status>"
        );
        std::process::exit(1);
    }

    match args[1].as_str() {
        "ingest-seed" => {
            if args.len() < 3 {
                eprintln!("Usage: paper-crawl ingest-seed <arxiv_id>");
                std::process::exit(1);
            }
            let id = &args[2];
            if let Err(e) = ingest_paper(id, 0) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        "parse-theorems" => {
            if let Err(e) = extract_all() {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        "find" => {
            if args.len() < 3 {
                eprintln!("Usage: paper-crawl find <query>");
                std::process::exit(1);
            }
            let q = &args[2..].join(" ");
            if let Err(e) = find_theorems(q) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        "get" => {
            if args.len() < 3 {
                eprintln!("Usage: paper-crawl get <label|id|name>");
                std::process::exit(1);
            }
            let target = &args[2];
            if let Err(e) = get_theorem(target) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        "crawl" => {
            let mut limit = 20;
            let mut depth = 2;
            let mut i = 2;
            while i < args.len() {
                if args[i] == "--limit" && i + 1 < args.len() {
                    limit = args[i + 1].parse().unwrap_or(20);
                    i += 2;
                } else if args[i] == "--depth" && i + 1 < args.len() {
                    depth = args[i + 1].parse().unwrap_or(2);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if let Err(e) = cmd_crawl(limit, depth) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        "status" => {
            if let Err(e) = cmd_status() {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        cmd => {
            eprintln!("Unknown command: {cmd}");
            std::process::exit(1);
        }
    }
}
