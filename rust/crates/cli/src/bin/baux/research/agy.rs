use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const USER: u32 = 14;
const ASSISTANT: u32 = 15;
const SUMMARY: u32 = 23;
const TOOL: u32 = 132;

pub(super) fn import(
    conversation: &str,
    full: bool,
    tools: bool,
    output: Option<&Path>,
) -> Result<(), String> {
    if !valid_id(conversation) {
        return Err(format!(
            "invalid Anti-Gravity conversation ID `{conversation}`; expected a UUID"
        ));
    }
    let db = db_path(conversation)?;
    if !db.is_file() {
        return Err(format!(
            "Anti-Gravity conversation `{conversation}` was not found at {}",
            db.display()
        ));
    }
    let (start, rows) = read_stable(&db, full, tools)?;
    let text = render(conversation, &db, start, rows, tools)?;
    if let Some(path) = output {
        fs::write(path, text)
            .map_err(|error| format!("could not write {}: {error}", path.display()))?;
        println!(
            "baux: imported Anti-Gravity conversation to {}",
            path.display()
        );
    } else {
        print!("{text}");
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct Stamp {
    length: u64,
    modified: Option<std::time::SystemTime>,
}

fn file_stamp(path: &Path) -> Result<Stamp, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("could not inspect {}: {error}", path.display()))?;
    Ok(Stamp {
        length: metadata.len(),
        modified: metadata.modified().ok(),
    })
}

fn read_stable(path: &Path, full: bool, tools: bool) -> Result<(Option<u64>, Vec<Row>), String> {
    for _ in 0..3 {
        let before = file_stamp(path)?;
        let start = if full { None } else { last_summary(path)? };
        let rows = read_rows(path, start, tools)?;
        if before == file_stamp(path)? {
            return Ok((start, rows));
        }
    }
    Err(
        "Anti-Gravity updated the conversation during import; retry when the current turn is idle"
            .to_owned(),
    )
}

#[derive(Debug, PartialEq, Eq)]
struct Row {
    index: u64,
    kind: u32,
    payload: Vec<u8>,
}

fn db_path(conversation: &str) -> Result<PathBuf, String> {
    let home = env::var_os("HOME").ok_or_else(|| "HOME is not set".to_owned())?;
    Ok(PathBuf::from(home)
        .join(".gemini/antigravity-cli/conversations")
        .join(format!("{conversation}.db")))
}

fn db_uri(path: &Path) -> Result<OsString, String> {
    let path = path
        .to_str()
        .ok_or_else(|| format!("database path is not UTF-8: {}", path.display()))?;
    if path.contains('?') || path.contains('#') {
        return Err(format!(
            "database path contains a URI control character: {path}"
        ));
    }
    Ok(OsString::from(format!("file:{path}?immutable=1")))
}

fn sqlite(path: &Path, query: &str) -> Result<String, String> {
    let result = Command::new("sqlite3")
        .args(["-readonly", "-batch", "-noheader", "-separator", "\t"])
        .arg(db_uri(path)?)
        .arg(query)
        .output()
        .map_err(|error| format!("could not start sqlite3: {error}"))?;
    if !result.status.success() {
        let error = String::from_utf8_lossy(&result.stderr);
        return Err(format!(
            "could not read {}: {}",
            path.display(),
            error.trim()
        ));
    }
    String::from_utf8(result.stdout).map_err(|error| format!("sqlite3 returned non-UTF-8: {error}"))
}

fn last_summary(path: &Path) -> Result<Option<u64>, String> {
    let value = sqlite(
        path,
        "SELECT MAX(idx) FROM steps WHERE step_type = 23 AND status = 3;",
    )?;
    let value = value.trim();
    if value.is_empty() {
        Ok(None)
    } else {
        value
            .parse()
            .map(Some)
            .map_err(|error| format!("invalid summary index `{value}`: {error}"))
    }
}

fn read_rows(path: &Path, start: Option<u64>, tools: bool) -> Result<Vec<Row>, String> {
    let floor = start.map_or_else(String::new, |index| format!(" AND idx >= {index}"));
    let kinds = if tools {
        "14, 15, 23, 132"
    } else {
        "14, 15, 23"
    };
    let query = format!(
        "SELECT idx, step_type, hex(step_payload) FROM steps WHERE status IN (3, 5) AND step_type \
         IN ({kinds}){floor} ORDER BY idx;"
    );
    sqlite(path, &query)?
        .lines()
        .map(parse_row)
        .collect::<Result<Vec<_>, _>>()
}

fn parse_row(line: &str) -> Result<Row, String> {
    let mut cells = line.splitn(3, '\t');
    let index = cells
        .next()
        .ok_or_else(|| "SQLite row has no index".to_owned())?
        .parse()
        .map_err(|error| format!("invalid step index: {error}"))?;
    let kind = cells
        .next()
        .ok_or_else(|| format!("step {index} has no type"))?
        .parse()
        .map_err(|error| format!("invalid type for step {index}: {error}"))?;
    let payload = decode_hex(
        cells
            .next()
            .ok_or_else(|| format!("step {index} has no payload"))?,
    )?;
    Ok(Row {
        index,
        kind,
        payload,
    })
}

fn render(
    conversation: &str,
    db: &Path,
    start: Option<u64>,
    rows: Vec<Row>,
    tools: bool,
) -> Result<String, String> {
    let mut text = format!(
        "# Anti-Gravity research context\n\nconversation: {conversation}\nsource: {}\nmode: {}\n",
        db.display(),
        if start.is_some() { "context" } else { "full" }
    );
    let mut count = 0_usize;
    for row in rows {
        let item = match row.kind {
            USER => field_text(&row.payload, &[19, 2])?.map(|body| ("User", body)),
            ASSISTANT => field_text(&row.payload, &[20, 1])?.map(|body| ("Assistant", body)),
            SUMMARY if start == Some(row.index) => field_text(&row.payload, &[30, 5])?
                .map(clean_summary)
                .map(|body| ("Context", body)),
            TOOL if tools => tool_text(&row.payload)?.map(|body| ("Tool", body)),
            _ => None,
        };
        let Some((role, body)) = item else {
            continue;
        };
        if body.trim().is_empty() {
            continue;
        }
        text.push_str(&format!(
            "\n## {role} · step {}\n\n{}\n",
            row.index,
            body.trim()
        ));
        count += 1;
    }
    if count == 0 {
        return Err("conversation contains no importable steps".to_owned());
    }
    Ok(text)
}

fn tool_text(payload: &[u8]) -> Result<Option<String>, String> {
    let Some(name) = field_text(payload, &[5, 4, 2])? else {
        return Ok(None);
    };
    let args = field_text(payload, &[5, 4, 3])?.unwrap_or_default();
    let label = field_text(payload, &[5, 30])?.unwrap_or_default();
    let mut text = name;
    if !label.is_empty() {
        text.push_str(&format!(" — {label}"));
    }
    if !args.is_empty() {
        text.push_str(&format!("\n\n```json\n{args}\n```"));
    }
    Ok(Some(text))
}

fn clean_summary(text: String) -> String {
    text.strip_prefix("<summary>")
        .and_then(|text| text.strip_suffix("</summary>"))
        .unwrap_or(&text)
        .trim()
        .to_owned()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Value<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
    Fixed32(&'a [u8]),
    Fixed64(&'a [u8]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Field<'a> {
    number: u32,
    value: Value<'a>,
}

fn parse_proto(mut input: &[u8]) -> Result<Vec<Field<'_>>, String> {
    let mut fields = Vec::new();
    while !input.is_empty() {
        let (key, used) = read_varint(input)?;
        input = &input[used..];
        let number = u32::try_from(key >> 3).map_err(|_| "protobuf field number overflow")?;
        if number == 0 {
            return Err("protobuf field number is zero".to_owned());
        }
        let wire = key & 7;
        let (value, used) = match wire {
            0 => {
                let (value, used) = read_varint(input)?;
                (Value::Varint(value), used)
            }
            1 => (Value::Fixed64(take(input, 8)?), 8),
            2 => {
                let (length, head) = read_varint(input)?;
                let length = usize::try_from(length)
                    .map_err(|_| "protobuf byte field is too large".to_owned())?;
                let bytes = take(&input[head..], length)?;
                (Value::Bytes(bytes), head + length)
            }
            5 => (Value::Fixed32(take(input, 4)?), 4),
            _ => return Err(format!("unsupported protobuf wire type {wire}")),
        };
        input = &input[used..];
        fields.push(Field { number, value });
    }
    Ok(fields)
}

fn field_bytes<'a>(input: &'a [u8], path: &[u32]) -> Result<Option<&'a [u8]>, String> {
    let Some((&head, tail)) = path.split_first() else {
        return Ok(Some(input));
    };
    for field in parse_proto(input)? {
        if field.number != head {
            continue;
        }
        let Value::Bytes(bytes) = field.value else {
            continue;
        };
        if tail.is_empty() {
            return Ok(Some(bytes));
        }
        if let Some(value) = field_bytes(bytes, tail)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn field_text(input: &[u8], path: &[u32]) -> Result<Option<String>, String> {
    field_bytes(input, path)?
        .map(|bytes| {
            String::from_utf8(bytes.to_vec())
                .map_err(|error| format!("protobuf text is not UTF-8: {error}"))
        })
        .transpose()
}

fn read_varint(input: &[u8]) -> Result<(u64, usize), String> {
    let mut value = 0_u64;
    for (index, byte) in input.iter().copied().take(10).enumerate() {
        if index == 9 && byte > 1 {
            return Err("protobuf varint overflow".to_owned());
        }
        value |= u64::from(byte & 0x7f) << (index * 7);
        if byte & 0x80 == 0 {
            return Ok((value, index + 1));
        }
    }
    Err("truncated protobuf varint".to_owned())
}

fn take(input: &[u8], length: usize) -> Result<&[u8], String> {
    input
        .get(..length)
        .ok_or_else(|| "truncated protobuf field".to_owned())
}

fn decode_hex(text: &str) -> Result<Vec<u8>, String> {
    if !text.len().is_multiple_of(2) {
        return Err("SQLite returned odd-length hexadecimal data".to_owned());
    }
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = hex_digit(pair[0])?;
            let low = hex_digit(pair[1])?;
            Ok((high << 4) | low)
        })
        .collect()
}

fn hex_digit(byte: u8) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(format!("invalid hexadecimal digit `{}`", char::from(byte))),
    }
}

fn valid_id(value: &str) -> bool {
    value.len() == 36
        && value.bytes().enumerate().all(|(index, byte)| {
            if matches!(index, 8 | 13 | 18 | 23) {
                byte == b'-'
            } else {
                byte.is_ascii_hexdigit()
            }
        })
}

#[cfg(test)]
mod tests {
    use super::{Row, clean_summary, decode_hex, field_text, parse_row, read_varint, valid_id};

    #[test]
    fn nested_text() {
        let payload = [0x9a, 0x01, 0x07, 0x12, 0x05, b'h', b'e', b'l', b'l', b'o'];
        assert_eq!(
            field_text(&payload, &[19, 2]).unwrap(),
            Some("hello".to_owned())
        );
    }

    #[test]
    fn invalid_varint() {
        assert!(read_varint(&[0x80]).is_err());
        assert!(read_varint(&[0xff; 10]).is_err());
    }

    #[test]
    fn sqlite_row() {
        assert_eq!(
            parse_row("7\t14\t9A0107120568656C6C6F").unwrap(),
            Row {
                index: 7,
                kind: 14,
                payload: decode_hex("9A0107120568656C6C6F").unwrap(),
            }
        );
    }

    #[test]
    fn checks_uuid() {
        assert!(valid_id("056ca082-9db3-4d19-ab75-09d668c8fb3c"));
        assert!(!valid_id("../../conversation"));
    }

    #[test]
    fn summary_tags() {
        assert_eq!(
            clean_summary("<summary>\nstate\n</summary>".to_owned()),
            "state"
        );
    }
}
