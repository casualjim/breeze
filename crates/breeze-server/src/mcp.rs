use std::any::TypeId;
use std::collections::HashMap;
use std::sync::Arc;

use breeze_indexer::Indexer;
use breeze_indexer::SearchGranularity as IndexerSearchGranularity;
use breeze_indexer::SearchOptions;

use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::tool_handler;
use rmcp::tool_router;
use rmcp::transport::streamable_http_server::{
  StreamableHttpService, session::local::LocalSessionManager,
};
use rmcp::{ErrorData, RoleServer, ServerHandler, model::*, service::RequestContext, tool};
use schemars::JsonSchema;
use schemars::generate::SchemaSettings;
use schemars::transform::AddNullable;
use serde::Deserialize;
use tokio::sync::RwLock;
use tracing::info;
use uuid::Uuid;

use crate::types::*;

/// A shortcut for generating a JSON schema for a type.
pub fn schema_for_type<T: JsonSchema>() -> JsonObject {
  let settings = SchemaSettings::draft07().with_transform(AddNullable::default());
  let generator = settings.into_generator();
  let schema = generator.into_root_schema_for::<T>();
  let object = serde_json::to_value(schema).expect("failed to serialize schema");
  match object {
    serde_json::Value::Object(mut object) => {
      // Remove meta-schema URL to prevent draft meta-schema validation
      object.remove("$schema");
      object
    }
    _ => panic!(
      "Schema serialization produced non-object value: expected JSON object but got {:?}",
      object
    ),
  }
}

/// Call [`schema_for_type`] with a cache
pub fn cached_schema_for_type<T: JsonSchema + std::any::Any>() -> Arc<JsonObject> {
  thread_local! {
      static CACHE_FOR_TYPE: std::sync::RwLock<HashMap<TypeId, Arc<JsonObject>>> = Default::default();
  };
  CACHE_FOR_TYPE.with(|cache| {
    if let Some(x) = cache
      .read()
      .expect("schema cache lock poisoned")
      .get(&TypeId::of::<T>())
    {
      x.clone()
    } else {
      let schema = schema_for_type::<T>();
      let schema = Arc::new(schema);
      cache
        .write()
        .expect("schema cache lock poisoned")
        .insert(TypeId::of::<T>(), schema.clone());
      schema
    }
  })
}

#[derive(Clone)]
pub struct BreezeService {
  indexer: Arc<Indexer>,
  tool_router: ToolRouter<BreezeService>,
  // Per-session defaults: project scope and canonical root
  session_project_id: Arc<RwLock<Option<Uuid>>>,
  session_root: Arc<RwLock<Option<String>>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ProjectPathSet {
  path: String,
}

#[tool_router]
impl BreezeService {
  pub fn new(indexer: Arc<Indexer>) -> Self {
    Self {
      indexer,
      tool_router: Self::tool_router(),
      session_project_id: Arc::new(RwLock::new(None)),
      session_root: Arc::new(RwLock::new(None)),
    }
  }

  #[tool(description = "Create a new project for indexing", input_schema = cached_schema_for_type::<CreateProject>())]
  async fn create_project(
    &self,
    Parameters(CreateProject {
      name,
      path,
      description,
    }): Parameters<CreateProject>,
  ) -> Result<CallToolResult, ErrorData> {
    let project_path = std::path::PathBuf::from(path);
    let name = name.unwrap_or_else(|| {
      project_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
    });
    info!("Creating project: {} at {}", name, project_path.display());

    match self
      .indexer
      .project_manager()
      .create_project(
        name,
        project_path.to_string_lossy().to_string(),
        description,
        None,
      )
      .await
    {
      Ok(project) => match self.indexer.index_project(project.id).await {
        Ok(_) => Ok(CallToolResult::success(vec![Content::text(format!(
          "Successfully created project '{}' with ID: {}",
          project.name, project.id
        ))])),
        Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
          "Failed to create project: {}",
          e
        ))])),
      },
      Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
        "Failed to create project: {}",
        e
      ))])),
    }
  }

  #[tool(description = "Search code using semantic understanding. Provide an absolute project path in the 'path' field to bound the search (e.g., /Users/alice/workspace/myproj). Relative paths are not accepted.", input_schema = cached_schema_for_type::<SimpleSearchRequest>())]
  async fn search_code(
    &self,
    Parameters(search_req): Parameters<SimpleSearchRequest>,
  ) -> Result<CallToolResult, ErrorData> {
    let limit: usize = match search_req.limit {
      Some(l) if l >= 0 => l as usize,
      _ => 10,
    };

    // Keep non-empty path only
    let req_path = search_req
      .path
      .as_ref()
      .and_then(|p| (!p.trim().is_empty()).then_some(p.as_str()));

    info!("Searching for '{}' with limit {}", search_req.query, limit);

    // Prefer explicit path (absolute or relative resolved via session_root); else use session project
    let project_id: Option<Uuid> = if let Some(_p) = req_path {
      let abs = match self.resolve_path_abs(req_path).await {
        Ok(v) => v,
        Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
      };
      if let Some(abs) = abs {
        match self.indexer.project_manager().find_by_path(&abs).await {
          Ok(Some(project)) => Some(project.id),
          _ => None,
        }
      } else {
        None
      }
    } else {
      self.get_session_project().await
    };

    if req_path.is_none() && project_id.is_none() {
      return Ok(CallToolResult::error(vec![Content::text(
        "No path provided and no session project set. Call project_path.set first.",
      )]));
    }

    let options = SearchOptions {
      file_limit: limit,
      chunks_per_file: 3,
      granularity: IndexerSearchGranularity::Chunk,
      ..Default::default()
    };

    match self
      .indexer
      .search(&search_req.query, options, project_id)
      .await
    {
      Ok(results) => {
        if results.is_empty() {
          return Ok(CallToolResult::success(vec![Content::text(
            "No results found.".to_string(),
          )]));
        }
        let mut report = String::new();
        // Normalize relevance by top score to make it intuitive (percentage-like)
        let max_score = results
          .iter()
          .map(|r| r.relevance_score)
          .fold(0.0f32, f32::max);
        for r in results {
          use std::fmt::Write as _;
          let rel_norm = if max_score > 0.0 {
            (r.relevance_score / max_score) as f64
          } else {
            0.0
          };
          let rel_pct = rel_norm * 100.0;
          let _ = writeln!(report, "## {}\n", r.file_path);
          let _ = writeln!(report, "Relevance: {:.1}%\n", rel_pct);

          for ch in r.chunks {
            let lang = if ch.language.trim().is_empty() {
              ""
            } else {
              ch.language.as_str()
            };
            // Fenced code block
            let _ = writeln!(report, "```{}\n{}\n```\n", lang, ch.content);
          }
        }
        Ok(CallToolResult::success(vec![Content::text(report)]))
      }
      Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
        "Search failed: {}",
        e
      ))])),
    }
  }

  #[tool(description = "Set default project root and scope for this session", input_schema = cached_schema_for_type::<ProjectPathSet>())]
  async fn set_project_path(
    &self,
    Parameters(ProjectPathSet { path }): Parameters<ProjectPathSet>,
  ) -> Result<CallToolResult, ErrorData> {
    // Resolve (supports relative via existing session root)
    let abs = match self.resolve_path_abs(Some(&path)).await {
      Ok(v) => v,
      Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
    };

    let Some(abs) = abs else {
      return Ok(CallToolResult::error(vec![Content::text(
        "Empty resolved path",
      )]));
    };

    // Validate directory exists
    match std::fs::metadata(&abs) {
      Ok(md) if md.is_dir() => {}
      _ => {
        return Ok(CallToolResult::error(vec![Content::text(
          "Path does not exist or is not a directory",
        )]));
      }
    }

    // Find project by path and store both id and canonical root (from DB)
    match self.indexer.project_manager().find_by_path(&abs).await {
      Ok(Some(project)) => {
        *self.session_project_id.write().await = Some(project.id);
        *self.session_root.write().await = Some(project.directory.clone());
        Ok(CallToolResult::success(vec![Content::json(
          serde_json::json!({
            "project_id": project.id,
            "project_root": project.directory
          }),
        )?]))
      }
      Ok(None) => Ok(CallToolResult::error(vec![Content::text(
        "No indexed project covers that path",
      )])),
      Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
        "Failed to resolve project: {}",
        e
      ))])),
    }
  }

  #[tool(description = "Get current session project root")]
  async fn get_project_path(&self) -> Result<CallToolResult, ErrorData> {
    Ok(CallToolResult::success(vec![Content::json(
      serde_json::json!({
        "project_id": self.get_session_project().await,
        "project_root": self.session_root.read().await.clone()
      }),
    )?]))
  }
}

impl BreezeService {
  // Internal helpers for session state and path resolution
  async fn get_session_project(&self) -> Option<Uuid> {
    *self.session_project_id.read().await
  }

  async fn resolve_path_abs(&self, maybe_path: Option<&str>) -> Result<Option<String>, String> {
    // If an explicit path is provided
    if let Some(p) = maybe_path {
      let path = std::path::Path::new(p);
      if path.is_absolute() {
        return canonicalize_nearest_ancestor(path)
          .map(Some)
          .map_err(|e| e.to_string());
      }
      // Relative path requires session_root
      let base = self
        .session_root
        .read()
        .await
        .clone()
        .ok_or_else(|| "No session root set; call project_path.set first".to_string())?;
      let joined = std::path::Path::new(&base).join(path);
      return canonicalize_nearest_ancestor(&joined)
        .map(Some)
        .map_err(|e| e.to_string());
    }

    // No path provided: return session_root (already canonical)
    Ok(self.session_root.read().await.clone())
  }
}

fn canonicalize_nearest_ancestor(p: &std::path::Path) -> std::io::Result<String> {
  let mut ancestor = Some(p);
  while let Some(a) = ancestor {
    if a.exists() {
      break;
    }
    ancestor = a.parent();
  }
  let normalized = if let Some(a) = ancestor {
    let ac = a.canonicalize().unwrap_or_else(|_| a.to_path_buf());
    let tail = p.strip_prefix(a).unwrap_or(p);
    ac.join(tail)
  } else {
    p.to_path_buf()
  };
  let mut s = normalized.to_string_lossy().to_string();
  if s.ends_with('/') && s.len() > 1 {
    s.pop();
  }
  Ok(s)
}

#[tool_handler]
impl ServerHandler for BreezeService {
  fn get_info(&self) -> ServerInfo {
    let title = env!("CARGO_PKG_DESCRIPTION");
    let title = if title.trim().is_empty() {
      None
    } else {
      Some(title.to_string())
    };

    let website_url = env!("CARGO_PKG_HOMEPAGE");
    let website_url = if website_url.trim().is_empty() {
      None
    } else {
      Some(website_url.to_string())
    };

    ServerInfo {
            protocol_version: ProtocolVersion::LATEST,
            capabilities: ServerCapabilities::builder()
                .enable_prompts()
                .enable_resources()
                .enable_prompts_list_changed()
                .enable_resources_list_changed()
                .enable_resources_subscribe()
                .enable_tools()
                .build(),
            server_info: Implementation {
              name:"breeze-mcp".to_string(),
              version:env!("CARGO_PKG_VERSION").to_string(),
              title,
              icons: None,
              website_url,
            },
            instructions: Some("Breeze MCP server provides semantic code search and indexing capabilities. Use 'search_code' to search through indexed code and 'create_project' to index new directories.".to_string()),
        }
  }

  async fn initialize(
    &self,
    _request: InitializeRequestParam,
    context: RequestContext<RoleServer>,
  ) -> Result<InitializeResult, rmcp::ErrorData> {
    if let Some(http_request_part) = context.extensions.get::<axum::http::request::Parts>() {
      let initialize_headers = &http_request_part.headers;
      let initialize_uri = &http_request_part.uri;
      info!(?initialize_headers, %initialize_uri, "MCP initialize from http server");
    }
    Ok(self.get_info())
  }
}

/// Create an HTTP streamable MCP service
pub fn create_http_service(indexer: Arc<Indexer>) -> StreamableHttpService<BreezeService> {
  StreamableHttpService::new(
    move || Ok(BreezeService::new(indexer.clone())),
    LocalSessionManager::default().into(),
    Default::default(),
  )
}
