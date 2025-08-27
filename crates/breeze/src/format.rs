use std::borrow::Cow;

use crate::app::{Project, Task};
use crate::cli::OutputMode;
use comfy_table::{Cell, ContentArrangement, Table, presets::UTF8_FULL};
use humantime_fmt;

pub trait ColumnSpec: Sized + Copy + 'static {
  type Entity;

  fn defaults() -> &'static [Self];

  /// Parse a vector of user-provided column strings into typed columns.
  /// Returns an error if any column is unknown.
  fn parse_many(cols: &[String]) -> Result<Vec<Self>, String>;

  /// The set of allowed machine-readable column keys (lowercase)
  fn allowed() -> &'static [&'static str];

  fn key(col: Self) -> &'static str;

  fn title(col: Self) -> &'static str;

  fn cell(entity: &Self::Entity, col: Self) -> Cow<'_, str>;
}

pub enum Formatter {
  Table,
  Tsv,
  Json { pretty: bool },
}

pub struct RenderOptions {
  pub headers: bool,
}

impl Formatter {
  pub fn render<C: ColumnSpec>(
    &self,
    items: &[C::Entity],
    cols: &[C],
    opts: &RenderOptions,
  ) -> Result<String, String>
  where
    C::Entity: serde::Serialize,
  {
    match self {
      Formatter::Tsv => {
        let mut out = String::new();
        // header uses machine keys (lowercase)
        if opts.headers {
          for (i, &c) in cols.iter().enumerate() {
            if i > 0 {
              out.push('\t');
            }
            out.push_str(C::key(c));
          }
          out.push('\n');
        }
        for item in items {
          for (i, &c) in cols.iter().enumerate() {
            if i > 0 {
              out.push('\t');
            }
            let v = C::cell(item, c);
            out.push_str(&v);
          }
          out.push('\n');
        }
        Ok(out)
      }
      Formatter::Table => {
        let mut table = Table::new();
        table
          .load_preset(UTF8_FULL)
          .set_content_arrangement(ContentArrangement::Dynamic);
        if opts.headers {
          table.set_header(
            cols
              .iter()
              .map(|&c| Cell::new(C::title(c).to_uppercase()))
              .collect::<Vec<_>>(),
          );
        }

        for item in items {
          let row = cols
            .iter()
            .map(|&c| Cell::new(C::cell(item, c)))
            .collect::<Vec<_>>();
          table.add_row(row);
        }

        Ok(table.to_string())
      }
      Formatter::Json { pretty } => {
        if *pretty {
          serde_json::to_string_pretty(items).map_err(|e| e.to_string())
        } else {
          serde_json::to_string(items).map_err(|e| e.to_string())
        }
      }
    }
  }
}

pub fn make_formatter(mode: OutputMode) -> Formatter {
  match mode {
    OutputMode::Table => Formatter::Table,
    OutputMode::Tsv => Formatter::Tsv,
    OutputMode::Json => Formatter::Json { pretty: true },
  }
}

// Project columns
#[derive(Clone, Copy)]
pub enum ProjectCol {
  Id,
  Name,
  Directory,
  Description,
}

impl std::str::FromStr for ProjectCol {
  type Err = String;
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.to_ascii_lowercase().as_str() {
      "id" => Ok(Self::Id),
      "name" => Ok(Self::Name),
      "directory" | "dir" | "path" => Ok(Self::Directory),
      "description" | "desc" => Ok(Self::Description),
      other => Err(format!("unknown project column: {}", other)),
    }
  }
}

impl ColumnSpec for ProjectCol {
  type Entity = Project;

  fn defaults() -> &'static [Self] {
    const COLS: &[ProjectCol] = &[
      ProjectCol::Id,
      ProjectCol::Name,
      ProjectCol::Directory,
      ProjectCol::Description,
    ];
    COLS
  }

  fn parse_many(cols: &[String]) -> Result<Vec<Self>, String> {
    if cols.is_empty() {
      return Ok(Self::defaults().to_vec());
    }
    let mut out = Vec::with_capacity(cols.len());
    for c in cols {
      match c.parse() {
        Ok(v) => out.push(v),
        Err(_) => {
          return Err(format!(
            "Unknown project column: '{}'. Allowed: {}",
            c,
            Self::allowed().join(",")
          ));
        }
      }
    }
    Ok(out)
  }

  fn allowed() -> &'static [&'static str] {
    &["id", "name", "directory", "description"]
  }

  fn key(col: Self) -> &'static str {
    match col {
      Self::Id => "id",
      Self::Name => "name",
      Self::Directory => "directory",
      Self::Description => "description",
    }
  }

  fn title(col: Self) -> &'static str {
    match col {
      Self::Id => "ID",
      Self::Name => "Name",
      Self::Directory => "Directory",
      Self::Description => "Description",
    }
  }

  fn cell(p: &Self::Entity, col: Self) -> Cow<'_, str> {
    match col {
      Self::Id => Cow::Owned(p.id.to_string()),
      Self::Name => Cow::Borrowed(&p.name),
      Self::Directory => Cow::Borrowed(&p.directory),
      Self::Description => Cow::Borrowed(p.description.as_deref().unwrap_or("")),
    }
  }
}

// Task columns
#[derive(Clone, Copy)]
pub enum TaskCol {
  Id,
  Project,
  Status,
  Created,
  Files,
}

impl std::str::FromStr for TaskCol {
  type Err = String;
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.to_ascii_lowercase().as_str() {
      "id" => Ok(Self::Id),
      "project" | "project_id" => Ok(Self::Project),
      "status" => Ok(Self::Status),
      "created" | "created_at" => Ok(Self::Created),
      "files" | "files_indexed" => Ok(Self::Files),
      other => Err(format!("unknown task column: {}", other)),
    }
  }
}

impl ColumnSpec for TaskCol {
  type Entity = Task;

  fn defaults() -> &'static [Self] {
    const COLS: &[TaskCol] = &[
      TaskCol::Id,
      TaskCol::Project,
      TaskCol::Status,
      TaskCol::Created,
      TaskCol::Files,
    ];
    COLS
  }

  fn parse_many(cols: &[String]) -> Result<Vec<Self>, String> {
    if cols.is_empty() {
      return Ok(Self::defaults().to_vec());
    }
    let mut out = Vec::with_capacity(cols.len());
    for c in cols {
      match c.parse() {
        Ok(v) => out.push(v),
        Err(_) => {
          return Err(format!(
            "Unknown task column: '{}'. Allowed: {}",
            c,
            Self::allowed().join(",")
          ));
        }
      }
    }
    Ok(out)
  }

  fn allowed() -> &'static [&'static str] {
    &["id", "project", "status", "created", "files"]
  }

  fn key(col: Self) -> &'static str {
    match col {
      Self::Id => "id",
      Self::Project => "project",
      Self::Status => "status",
      Self::Created => "created",
      Self::Files => "files",
    }
  }

  fn title(col: Self) -> &'static str {
    match col {
      Self::Id => "ID",
      Self::Project => "Project",
      Self::Status => "Status",
      Self::Created => "Created",
      Self::Files => "Files",
    }
  }

  fn cell(t: &Self::Entity, col: Self) -> Cow<'_, str> {
    match col {
      Self::Id => Cow::Owned(t.id.to_string()),
      Self::Project => Cow::Owned(t.project_id.to_string()),
      Self::Status => Cow::Owned(t.status.to_string()),
      Self::Created => Cow::Owned(humantime_fmt::format_relative(
        t.created_at.and_utc().into(),
      )),
      Self::Files => Cow::Owned(
        t.files_indexed
          .map(|v| v.to_string())
          .unwrap_or_else(|| "-".to_string()),
      ),
    }
  }
}

pub fn render_projects(
  projects: &[Project],
  columns: &[String],
  mode: OutputMode,
  headers: bool,
) -> Result<String, String> {
  let cols = ProjectCol::parse_many(columns)?;
  let fmt = make_formatter(mode);
  let opts = RenderOptions { headers };
  fmt.render::<ProjectCol>(projects, &cols, &opts)
}

pub fn render_tasks(
  tasks: &[Task],
  columns: &[String],
  mode: OutputMode,
  headers: bool,
) -> Result<String, String> {
  let cols = TaskCol::parse_many(columns)?;
  let fmt = make_formatter(mode);
  let opts = RenderOptions { headers };
  fmt.render::<TaskCol>(tasks, &cols, &opts)
}
