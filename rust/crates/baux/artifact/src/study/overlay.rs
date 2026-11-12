//! Value overlay, deep-merge, dot-path normalization, and variant-aware
//! cleanup.

use deser_value::{Kind, Map, Value};

/// Deeply normalize dotted keys in map values into nested map hierarchies.
pub fn normalize_paths(value: Value) -> Value {
    let (kind, meta) = value.into_parts();
    let new_kind = match kind {
        Kind::Map(map) => {
            let mut result = Map::new();
            for (k, v) in map.iter() {
                let norm_v = normalize_paths(v.clone());
                if let Some(key_str) = k.as_str() {
                    if key_str.contains('.') {
                        let nested = nested_map(key_str, norm_v);
                        merge_into(&mut result, nested);
                        continue;
                    }
                }
                insert_merge(&mut result, k, norm_v);
            }
            Kind::Map(result)
        }
        Kind::Seq(seq) => {
            let items: Vec<Value> = seq
                .iter()
                .map(|item| normalize_paths(item.clone()))
                .collect();
            Kind::Seq(deser_value::Seq::from(items))
        }
        other => other,
    };
    let mut out = Value::new(new_kind);
    if let Some(m) = meta {
        out.set_meta(Some(m));
    }
    out
}

fn nested_map(key_str: &str, val: Value) -> Map {
    let parts: Vec<&str> = key_str.split('.').collect();
    let mut current = val;
    for &part in parts.iter().rev() {
        let mut map = Map::new();
        map.insert(part, current);
        current = Value::from(map);
    }
    match current.into_kind() {
        Kind::Map(m) => m,
        _ => Map::new(),
    }
}

fn insert_merge(target: &mut Map, key: &Value, val: Value) {
    if let Some(existing) = target.get_mut(key) {
        if existing.is_map() && val.is_map() {
            if let Kind::Map(src_map) = val.into_kind()
                && let Some(dst_map) = existing.as_map_mut()
            {
                merge_maps(dst_map, src_map);
            }
            return;
        }
        *existing = val;
    } else {
        target.insert(key.clone(), val);
    }
}

fn merge_into(target: &mut Map, source: Map) {
    merge_maps(target, source);
}

fn merge_maps(target: &mut Map, source: Map) {
    for (k, v) in source.into_iter() {
        insert_merge(target, &k, v);
    }
}

/// Deeply merge an overlay Value into a base Value.
pub fn deep_merge(base: &mut Value, overlay: &Value) {
    if !base.is_map() || !overlay.is_map() {
        *base = overlay.clone();
        return;
    }
    let overlay_map = match overlay.as_map() {
        Some(m) => m,
        None => return,
    };
    let base_map = match base.as_map_mut() {
        Some(m) => m,
        None => return,
    };

    for (k, v) in overlay_map.iter() {
        if v.as_str() == Some("$delete") {
            base_map.remove(k);
        } else if let Some(existing) = base_map.get_mut(k) {
            if existing.is_map() && v.is_map() {
                deep_merge(existing, v);
            } else {
                *existing = v.clone();
            }
        } else {
            base_map.insert(k.clone(), v.clone());
        }
    }
}

/// Recursively prune any keys whose value is "$delete".
pub fn prune_deletes(val: &mut Value) {
    if let Some(map) = val.as_map_mut() {
        let keys_to_remove: Vec<Value> = map
            .iter()
            .filter_map(|(k, v)| (v.as_str() == Some("$delete")).then(|| k.clone()))
            .collect();
        for k in keys_to_remove {
            map.remove(&k);
        }
        for v in map.values_mut() {
            prune_deletes(v);
        }
    } else if let Some(seq) = val.as_seq_mut() {
        for v in seq.iter_mut() {
            prune_deletes(v);
        }
    }
}

/// Clean up simulator and surrogate variant fields based on merged
/// configuration state.
pub fn clean_variants(leaf: &mut Value, overlay: &Value) {
    if get_str(leaf, "backend") == Some("resident-metal")
        && !has_path(overlay, "optimizer.perturbation")
    {
        remove_path(leaf, "optimizer.perturbation");
    }

    if get_path(leaf, "optimizer.surrogate").is_some() {
        if get_str(leaf, "optimizer.surrogate.fit") == Some("fixed") {
            if !has_path(overlay, "optimizer.surrogate.fit-candidates") {
                remove_path(leaf, "optimizer.surrogate.fit-candidates");
            }
            if !has_path(overlay, "optimizer.surrogate.fit-samples") {
                remove_path(leaf, "optimizer.surrogate.fit-samples");
            }
        }

        if let Some(metric) = get_str(leaf, "optimizer.surrogate.metric") {
            if metric == "euclidean" {
                remove_path(leaf, "optimizer.surrogate.local-neighbors");
            } else if matches!(metric, "conformal" | "hyperbolic" | "wasserstein")
                && !has_path(leaf, "optimizer.surrogate.local-neighbors")
            {
                set_path(
                    leaf,
                    "optimizer.surrogate.local-neighbors",
                    Value::from(4u64),
                );
            }
        }
    }
}

/// Query a value by dot-separated path.
pub fn get_path<'a>(val: &'a Value, path: &str) -> Option<&'a Value> {
    let mut cur = val;
    for part in path.split('.') {
        cur = cur.as_map()?.get(part)?;
    }
    Some(cur)
}

/// Query a string value by dot-separated path.
pub fn get_str<'a>(val: &'a Value, path: &str) -> Option<&'a str> {
    get_path(val, path).and_then(|v| v.as_str())
}

/// Check if a non-null value exists at the specified dot-separated path.
pub fn has_path(val: &Value, path: &str) -> bool {
    matches!(get_path(val, path), Some(v) if !v.is_null())
}

/// Set a value at a dot-separated path, creating parent maps as needed.
pub fn set_path(root: &mut Value, path: &str, new_val: Value) {
    if !root.is_map() {
        *root = Value::from(Map::new());
    }
    let parts: Vec<&str> = path.split('.').collect();
    let mut cur = root;
    for &part in &parts[..parts.len() - 1] {
        let needs_new_map = cur
            .as_map()
            .and_then(|m| m.get(part))
            .map_or(true, |v| !v.is_map());
        if needs_new_map {
            if let Some(m) = cur.as_map_mut() {
                m.insert(part, Map::new());
            }
        }
        cur = cur.as_map_mut().unwrap().get_mut(part).unwrap();
    }
    let last = parts[parts.len() - 1];
    if new_val.as_str() == Some("$delete") {
        if let Some(m) = cur.as_map_mut() {
            m.remove(last);
        }
    } else if let Some(m) = cur.as_map_mut() {
        m.insert(last, new_val);
    }
}

/// Remove a value at a dot-separated path.
pub fn remove_path(root: &mut Value, path: &str) {
    let parts: Vec<&str> = path.split('.').collect();
    let mut cur = root;
    for &part in &parts[..parts.len() - 1] {
        match cur.as_map_mut().and_then(|m| m.get_mut(part)) {
            Some(next) if next.is_map() => cur = next,
            _ => return,
        }
    }
    if let Some(map) = cur.as_map_mut() {
        map.remove(parts[parts.len() - 1]);
    }
}

/// Parse a raw CLI scalar string into a strongly-typed Value.
pub fn parse_value(raw: &str) -> Value {
    let s = raw.trim();
    if s == "true" {
        return Value::from(true);
    }
    if s == "false" {
        return Value::from(false);
    }
    if let Ok(n) = s.parse::<u64>() {
        return Value::from(n);
    }
    if let Ok(n) = s.parse::<i64>() {
        return Value::from(n);
    }
    if (s.contains('.') || s.contains('e') || s.contains('E'))
        && let Ok(f) = s.parse::<f64>()
    {
        return Value::from(f);
    }
    if (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
    {
        return Value::from(&s[1..s.len() - 1]);
    }
    if (s.starts_with('[') && s.ends_with(']')) || (s.starts_with('{') && s.ends_with('}')) {
        if let Ok(v) = crate::artifact::toml::from_str::<Value>(s) {
            return v;
        }
    }
    Value::from(s)
}

/// Split a string on commas that appear outside quotes or brackets.
pub fn split_commas(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut bracket_depth = 0usize;

    for ch in s.chars() {
        match ch {
            '\'' if !in_double => {
                in_single = !in_single;
                current.push(ch);
            }
            '"' if !in_single => {
                in_double = !in_double;
                current.push(ch);
            }
            '[' if !in_single && !in_double => {
                bracket_depth += 1;
                current.push(ch);
            }
            ']' if !in_single && !in_double && bracket_depth > 0 => {
                bracket_depth -= 1;
                current.push(ch);
            }
            ',' if !in_single && !in_double && bracket_depth == 0 => {
                parts.push(current.trim().to_string());
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }
    if !current.trim().is_empty() || !parts.is_empty() {
        parts.push(current.trim().to_string());
    }
    parts
}
