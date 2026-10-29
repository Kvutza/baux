//! Shared terminal rendering for BAUX developer and experiment commands.

use std::io::{self, IsTerminal, Write};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[90m";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Theme {
    #[default]
    Dark,
    Light,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tag {
    Orchestration,
    Tuning,
    Kernel,
    Metric,
    Body,
    Warning,
    Error,
    Heartbeat,
}

#[derive(Clone, Copy)]
struct Palette {
    error: &'static str,
    warning: &'static str,
    tag: &'static str,
    accent: &'static str,
    body: &'static str,
}

const DARK: Palette = Palette {
    error: "\x1b[38;2;224;122;95m",
    warning: "\x1b[38;2;245;158;66m",
    tag: "\x1b[38;2;110;113;142m",
    accent: "\x1b[38;2;129;178;154m",
    body: "\x1b[38;2;235;235;235m",
};

const LIGHT: Palette = Palette {
    error: "\x1b[38;2;179;78;61m",
    warning: "\x1b[38;2;196;98;44m",
    tag: "\x1b[38;2;55;57;72m",
    accent: "\x1b[38;2;77;118;98m",
    body: "\x1b[38;2;24;24;26m",
};

impl Tag {
    const fn code(self) -> &'static str {
        match self {
            Self::Orchestration => "o",
            Self::Tuning => "t",
            Self::Kernel => "k",
            Self::Metric => "m",
            Self::Body => "b",
            Self::Warning => "w",
            Self::Error => "e",
            Self::Heartbeat => "h",
        }
    }
}

pub fn theme() -> Theme {
    match std::env::var("BAUX_TERMINAL_THEME").as_deref() {
        Ok("light") => Theme::Light,
        _ => Theme::Dark,
    }
}

pub fn tag_prefix(who: &str) -> String {
    let separator = if matches!(who, "t" | "m" | "b") {
        " "
    } else {
        ""
    };
    format!("{who}|{separator}")
}

pub fn render(tag: Tag, text: &str, color: bool, theme: Theme) -> String {
    let palette = match theme {
        Theme::Dark => DARK,
        Theme::Light => LIGHT,
    };
    let prefix = tag_prefix(tag.code());
    text.lines().fold(String::new(), |mut output, line| {
        if line.is_empty() {
            return output;
        }
        if color {
            let body = match tag {
                Tag::Warning => palette.warning,
                Tag::Error => palette.error,
                Tag::Heartbeat | Tag::Body => DIM,
                Tag::Metric => palette.accent,
                _ => palette.body,
            };
            output.push_str(palette.tag);
            output.push_str(&prefix);
            output.push_str(RESET);
            output.push_str(body);
            output.push_str(line);
            output.push_str(RESET);
        } else {
            output.push_str(&prefix);
            output.push_str(line);
        }
        output.push('\n');
        output
    })
}

pub fn stderr(tag: Tag, text: &str) {
    let rendered = render(tag, text, io::stderr().is_terminal(), theme());
    let _ = io::stderr().write_all(rendered.as_bytes());
}

/// Report a command failure using the shared terminal format and exit status.
pub fn finish(result: Result<(), String>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            stderr(Tag::Error, &format!("baux: {error}"));
            ExitCode::FAILURE
        }
    }
}

pub fn stdout(tag: Tag, text: &str) {
    let rendered = render(tag, text, io::stdout().is_terminal(), theme());
    let _ = io::stdout().write_all(rendered.as_bytes());
}

pub fn write_log(mut output: impl Write, line: &[u8]) -> io::Result<()> {
    for part in line.split_inclusive(|byte| *byte == b'\n') {
        let payload = part.strip_suffix(b"\n").unwrap_or(part);
        output.write_all(timestamp().as_bytes())?;
        output.write_all(b" ")?;
        output.write_all(payload)?;
        output.write_all(b"\n")?;
    }
    Ok(())
}

fn timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let days = seconds.div_euclid(86_400);
    let day_seconds = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_date(days);
    let hour = day_seconds / 3_600;
    let minute = day_seconds % 3_600 / 60;
    let second = day_seconds % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn civil_date(days_since_epoch: i64) -> (i64, i64, i64) {
    let z = days_since_epoch + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_spacing() {
        assert_eq!(tag_prefix("o"), "o|");
        assert_eq!(tag_prefix("t"), "t| ");
        assert_eq!(tag_prefix("m"), "m| ");
    }

    #[test]
    fn palette() {
        assert!(render(Tag::Error, "bad", true, Theme::Dark).contains("38;2;224;122;95"));
        assert!(render(Tag::Error, "bad", true, Theme::Light).contains("38;2;179;78;61"));
        assert!(render(Tag::Tuning, "run", true, Theme::Dark).contains("38;2;110;113;142"));
    }

    #[test]
    fn plain_output() {
        let output = render(Tag::Metric, "nll 3.2", false, Theme::Dark);
        assert_eq!(output, "m| nll 3.2\n");
        assert!(!output.contains("\x1b["));
    }

    #[test]
    fn epoch_date() {
        assert_eq!(civil_date(0), (1970, 1, 1));
        assert_eq!(civil_date(20_000), (2024, 10, 4));
    }
}
