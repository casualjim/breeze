use crate::App;
use crate::Config;
use crate::app::SearchRequest;
#[cfg(feature = "perfprofiling")]
use crate::cli::DebugCommands;
use crate::cli::{OutputMode, ProjectCommands, SearchGranularity, TaskCommands};
use humantime_fmt;
use std::path::PathBuf;
use syntastica::language_set::SupportedLanguage;
use tracing::{error, info};

pub async fn handle_project_commands(app: App, cmd: ProjectCommands) -> Result<(), i32> {
  match cmd {
    ProjectCommands::Create {
      name,
      directory,
      description,
    } => handle_project_create(app, name, directory, description).await,
    ProjectCommands::List {
      output,
      no_headers,
      columns,
    } => handle_project_list(app, output, no_headers, columns).await,
    ProjectCommands::Show { id } => handle_project_show(app, id).await,
    ProjectCommands::Update {
      id,
      name,
      description,
    } => handle_project_update(app, id, name, description).await,
    ProjectCommands::Delete { id } => handle_project_delete(app, id).await,
    ProjectCommands::Index { id } => handle_project_index(app, id).await,
  }
}

pub async fn handle_task_commands(app: App, cmd: TaskCommands) -> Result<(), i32> {
  match cmd {
    TaskCommands::Show { id } => handle_task_show(app, id).await,
    TaskCommands::List {
      limit,
      output,
      no_headers,
      columns,
    } => handle_task_list(app, limit, output, no_headers, columns).await,
  }
}

async fn handle_project_show(app: App, id: String) -> Result<(), i32> {
  match app.get_project(&id).await {
    Ok(project) => {
      println!("ID: {}", project.id);
      println!("Name: {}", project.name);
      println!("Directory: {}", project.directory);
      if let Some(desc) = &project.description {
        println!("Description: {}", desc);
      }
      println!("Created: {}", humantime_fmt::format_relative(project.created_at.and_utc().into()));
      println!("Updated: {}", humantime_fmt::format_relative(project.updated_at.and_utc().into()));
      Ok(())
    }
    Err(e) => {
      error!("Failed to get project: {}", e);
      Err(1)
    }
  }
}

async fn handle_task_show(app: App, id: String) -> Result<(), i32> {
  match app.get_task(&id).await {
    Ok(task) => {
      println!("Task ID: {}", task.id);
      println!("Project ID: {}", task.project_id);
      println!("Path: {}", task.path);
      println!("Status: {}", task.status);
      println!("Created: {}", humantime_fmt::format_relative(task.created_at.and_utc().into()));
      if let Some(started) = task.started_at {
        println!("Started: {}", humantime_fmt::format_relative(started.and_utc().into()));
      }
      if let Some(completed) = task.completed_at {
        println!("Completed: {}", humantime_fmt::format_relative(completed.and_utc().into()));
      }
      if let Some(error) = &task.error {
        println!("Error: {}", error);
      }
      if let Some(files) = task.files_indexed {
        println!("Files indexed: {}", files);
      }
      Ok(())
    }
    Err(e) => {
      error!("Failed to get task: {}", e);
      Err(1)
    }
  }
}

async fn handle_project_update(
  app: App,
  id: String,
  name: Option<String>,
  description: Option<String>,
) -> Result<(), i32> {
  match app.update_project(&id, name, description).await {
    Ok(project) => {
      println!("Project updated successfully!");
      println!("ID: {}", project.id);
      println!("Name: {}", project.name);
      if let Some(desc) = &project.description {
        println!("Description: {}", desc);
      }
      Ok(())
    }
    Err(e) => {
      error!("Failed to update project: {}", e);
      Err(1)
    }
  }
}

async fn handle_project_delete(app: App, id: String) -> Result<(), i32> {
  match app.delete_project(&id).await {
    Ok(_) => {
      println!("Project deleted successfully!");
      Ok(())
    }
    Err(e) => {
      error!("Failed to delete project: {}", e);
      Err(1)
    }
  }
}

async fn handle_project_index(app: App, id: String) -> Result<(), i32> {
  match app.index_project(&id).await {
    Ok(resp) => {
      println!("Indexing task submitted successfully!");
      println!("Task ID: {}", resp.task_id);
      println!("Status: {}", resp.status);
      Ok(())
    }
    Err(e) => {
      error!("Failed to index project: {}", e);
      Err(1)
    }
  }
}

async fn handle_project_create(
  app: App,
  name: String,
  directory: PathBuf,
  description: Option<String>,
) -> Result<(), i32> {
  match app
    .create_project(name, directory.to_string_lossy().to_string(), description)
    .await
  {
    Ok(project) => {
      println!("Project created successfully!");
      println!("ID: {}", project.id);
      println!("Name: {}", project.name);
      println!("Directory: {}", project.directory);
      if let Some(desc) = &project.description {
        println!("Description: {}", desc);
      }
      Ok(())
    }
    Err(e) => {
      error!("Failed to create project: {}", e);
      Err(1)
    }
  }
}

async fn handle_project_list(
  app: App,
  output: OutputMode,
  no_headers: bool,
  columns: Option<Vec<String>>,
) -> Result<(), i32> {
  match app.list_projects().await {
    Ok(projects) => {
      if projects.is_empty() {
        // No output when empty; keep output modes consistent
        println!("No projects found.");
      } else {
        use crate::tables::projects_render;
        match projects_render(
          &projects,
          columns.as_deref().unwrap_or(&[]),
          output,
          !no_headers,
        ) {
          Ok(s) => print!("{}", s),
          Err(e) => {
            error!("{}", e);
            return Err(1);
          }
        }
      }
      Ok(())
    }
    Err(e) => {
      error!("Failed to list projects: {}", e);
      Err(1)
    }
  }
}

async fn handle_task_list(
  app: App,
  limit: Option<usize>,
  output: OutputMode,
  no_headers: bool,
  columns: Option<Vec<String>>,
) -> Result<(), i32> {
  match app.list_tasks(limit).await {
    Ok(tasks) => {
      if tasks.is_empty() {
        // No output when empty; keep output modes consistent
        println!("No tasks found.");
      } else {
        use crate::tables::tasks_render;
        match tasks_render(
          &tasks,
          columns.as_deref().unwrap_or(&[]),
          output,
          !no_headers,
        ) {
          Ok(s) => print!("{}", s),
          Err(e) => {
            error!("{}", e);
            return Err(1);
          }
        }
      }
      Ok(())
    }
    Err(e) => {
      error!("Failed to list tasks: {}", e);
      Err(1)
    }
  }
}

#[derive(Debug, Clone)]
pub struct SearchCommandArgs {
  pub query: String,
  pub limit: Option<usize>,
  pub chunks_per_file: Option<usize>,
  pub languages: Option<Vec<String>>,
  pub granularity: Option<SearchGranularity>,
  pub node_types: Option<Vec<String>>,
  pub node_name_pattern: Option<String>,
  pub parent_context_pattern: Option<String>,
  pub scope_depth: Option<(usize, usize)>,
  pub has_definitions: Option<Vec<String>>,
  pub has_references: Option<Vec<String>>,
  pub path: Option<PathBuf>,
}

pub async fn handle_search_command(app: App, args: SearchCommandArgs) -> Result<(), i32> {
  info!("Starting search for: \"{}\"", args.query);

  let req = SearchRequest {
    project_id: None, // No project ID for CLI search
    path: args.path.map(|p| p.display().to_string()),
    query: args.query,
    limit: args.limit,
    chunks_per_file: args.chunks_per_file,
    languages: args.languages,
    granularity: args
      .granularity
      .map(|g| match g {
        SearchGranularity::Document => breeze_server::types::SearchGranularity::Document,
        SearchGranularity::Chunk => breeze_server::types::SearchGranularity::Chunk,
      })
      .or(Some(breeze_server::types::SearchGranularity::Chunk)),
    node_types: args.node_types,
    node_name_pattern: args.node_name_pattern,
    parent_context_pattern: args.parent_context_pattern,
    scope_depth: args.scope_depth,
    has_definitions: args.has_definitions,
    has_references: args.has_references,
  };

  match app.search(req).await {
    Ok(results) => {
      if results.is_empty() {
        println!("No results found.");
      } else {
        let theme = syntastica_themes::catppuccin::mocha();
        let mut renderer = syntastica::renderer::TerminalRenderer::default();
        let language_set = syntastica_parsers::LanguageSetImpl::default();
        language_set.preload_all().unwrap();
        println!("\nFound {} results:", results.len());
        for (idx, result) in results.iter().enumerate() {
          println!("\n[{}] {}", idx + 1, result.file_path);
          println!(
            "   Score: {:.4} | Chunks: {}",
            result.relevance_score, result.chunk_count
          );

          if !result.chunks.is_empty() {
            println!("   Found {} relevant chunks:", result.chunks.len());
            for (chunk_idx, chunk) in result.chunks.iter().enumerate() {
              println!(
                "\n   Chunk {} (lines {}-{}, score: {:.4}):",
                chunk_idx + 1,
                chunk.start_line,
                chunk.end_line,
                chunk.relevance_score
              );

              // Determine language from file extension
              let language =
                syntastica_parsers::Lang::for_name(chunk.language.to_lowercase(), &language_set)
                  .ok();

              if let Some(language) = language {
                let highlighted = syntastica::highlight(
                  &chunk.content,
                  language,
                  &language_set,
                  &mut renderer,
                  theme.clone(),
                )
                .unwrap();

                // Print highlighted lines with indentation
                for line in highlighted.lines() {
                  println!("      {}", line);
                }
              } else {
                println!("{}", &chunk.content);
              }
            }
          }
          println!("{}", "-".repeat(80));
        }
      }
      Ok(())
    }
    Err(e) => {
      error!("Search failed: {}", e);
      Err(1)
    }
  }
}

pub async fn handle_init_command(force: bool) -> Result<(), i32> {
  match Config::default_config_path() {
    Ok(config_path) => {
      // Check if config already exists
      if config_path.exists() && !force {
        error!(
          "Configuration file already exists at: {}\nUse --force to overwrite",
          config_path.display()
        );
        return Err(1);
      }

      // Create config directory if it doesn't exist
      if let Some(parent) = config_path.parent()
        && let Err(e) = tokio::fs::create_dir_all(parent).await
      {
        error!("Failed to create config directory: {}", e);
        return Err(1);
      }

      // Write the commented config file
      let config_content = Config::generate_commented_config();
      if let Err(e) = tokio::fs::write(&config_path, config_content).await {
        error!("Failed to write config file: {}", e);
        return Err(1);
      }

      println!(
        "✅ Configuration file created at: {}",
        config_path.display()
      );
      println!("\nYou can now:");
      println!("  1. Edit the config file to customize settings");
      println!("  2. Run 'breeze serve' to start the API server");
      println!("  3. Create a project with 'breeze project create <name> <path>'");
      println!("  4. Index it with 'breeze project index <project-id>'");
      println!("\nThe config file is heavily commented to help you get started!");
      Ok(())
    }
    Err(e) => {
      error!("Failed to determine config path: {}", e);
      Err(1)
    }
  }
}

pub async fn handle_config_command(config: &Config, defaults: bool) -> Result<(), i32> {
  if defaults {
    // Show the default configuration
    println!("{}", toml::to_string_pretty(&Config::default()).unwrap());
  } else {
    // Show the actual loaded configuration (which already factors in the config file)
    println!("{}", toml::to_string_pretty(config).unwrap());
  }
  Ok(())
}

#[cfg(feature = "perfprofiling")]
pub async fn handle_debug_command(command: DebugCommands) -> Result<(), i32> {
  match command {
    #[cfg(feature = "perfprofiling")]
    DebugCommands::ChunkDirectory {
      path,
      max_chunk_size,
      max_file_size,
      max_parallel,
      tokenizer,
    } => {
      use breeze_chunkers::{Tokenizer, WalkOptions, walk_project};
      use futures_util::StreamExt;
      use std::collections::BTreeMap;
      use std::sync::Arc;
      use tokio::sync::Mutex;

      info!("Starting chunk analysis of: {}", path.display());

      // Parse tokenizer string
      let tokenizer_obj = if tokenizer == "characters" {
        Tokenizer::Characters
      } else if let Some(encoding) = tokenizer.strip_prefix("tiktoken:") {
        Tokenizer::Tiktoken(encoding.to_string())
      } else if let Some(model) = tokenizer.strip_prefix("hf:") {
        Tokenizer::HuggingFace(model.to_string())
      } else {
        error!(
          "Invalid tokenizer format: {}. Use 'characters', 'tiktoken:cl100k_base', or 'hf:org/repo'",
          tokenizer
        );
        return Err(1);
      };

      let start = std::time::Instant::now();

      let chunker = walk_project(
        &path,
        WalkOptions {
          max_chunk_size,
          tokenizer: tokenizer_obj,
          max_parallel,
          max_file_size,
          large_file_threads: 4,
          ..Default::default()
        },
      );

      let file_chunk_counts = Arc::new(Mutex::new(BTreeMap::<String, usize>::new()));
      #[cfg(feature = "perfprofiling")]
      let file_start_times = Arc::new(Mutex::new(BTreeMap::<String, std::time::Instant>::new()));
      let total_chunks = Arc::new(std::sync::atomic::AtomicUsize::new(0));

      // Process chunks concurrently with a buffer
      let concurrent_limit = max_parallel * 2; // Process more chunks than walker threads

      chunker
        .map(|chunk_result| {
          let file_chunk_counts = Arc::clone(&file_chunk_counts);
          #[cfg(feature = "perfprofiling")]
          let file_start_times = Arc::clone(&file_start_times);
          let total_chunks = Arc::clone(&total_chunks);

          async move {
            match chunk_result {
              Ok(project_chunk) => {
                let file_path = project_chunk.file_path.clone();

                // Track the first time we see a chunk from this file
                #[cfg(feature = "perfprofiling")]
                {
                  let mut start_times = file_start_times.lock().await;
                  start_times
                    .entry(file_path.clone())
                    .or_insert_with(std::time::Instant::now);
                }

                // Check if this is an EOF chunk
                if let breeze_chunkers::Chunk::EndOfFile {
                  file_path: eof_path,
                  ..
                } = &project_chunk.chunk
                {
                  // Record the file processing time
                  #[cfg(feature = "perfprofiling")]
                  {
                    let start_time = {
                      let mut start_times = file_start_times.lock().await;
                      start_times.remove(eof_path)
                    };

                    if let Some(start_time) = start_time {
                      let duration = start_time.elapsed();
                      let file_size = tokio::fs::metadata(eof_path)
                        .await
                        .map(|m| m.len())
                        .unwrap_or(0);

                      // Detect language from the file path
                      let language = if let Ok(Some(detection)) =
                        hyperpolyglot::detect(std::path::Path::new(eof_path))
                      {
                        detection.language().to_string()
                      } else {
                        "unknown".to_string()
                      };

                      // Record to performance tracker
                      breeze_chunkers::performance::get_tracker().record_file_processing(
                        eof_path.clone(),
                        language,
                        file_size,
                        duration,
                        "chunking_complete",
                      );
                    }
                  }
                }

                // Update counts
                {
                  let mut counts = file_chunk_counts.lock().await;
                  *counts.entry(file_path).or_insert(0) += 1;
                }

                let count = total_chunks.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if count % 1000 == 0 {
                  info!("Processed {} chunks so far...", count);
                }
              }
              Err(e) => {
                error!("Error processing chunk: {}", e);
              }
            }
          }
        })
        .buffer_unordered(concurrent_limit)
        .collect::<Vec<_>>()
        .await;

      let elapsed = start.elapsed();

      // Get final values
      let final_counts = file_chunk_counts.lock().await;
      let final_total = total_chunks.load(std::sync::atomic::Ordering::Relaxed);

      // Print summary statistics
      println!("\n=== Chunking Summary ===");
      println!("Directory: {}", path.display());
      println!("Total files processed: {}", final_counts.len());
      println!("Total chunks: {}", final_total);
      println!("Time elapsed: {:.2}s", elapsed.as_secs_f64());
      println!(
        "Chunks per second: {:.0}",
        final_total as f64 / elapsed.as_secs_f64()
      );

      if !final_counts.is_empty() {
        let avg_chunks_per_file = final_total as f64 / final_counts.len() as f64;
        println!("Average chunks per file: {:.1}", avg_chunks_per_file);
      }

      Ok(())
    }
  }
}

pub async fn handle_serve_command(
  config: Config,
  shutdown_token: tokio_util::sync::CancellationToken,
) -> Result<(), i32> {
  info!(
    "Starting Breeze API server on ports HTTP:{} HTTPS:{}",
    config.server.ports.http, config.server.ports.https
  );

  // Additional startup configuration details
  // Max file size (configured)
  if let Some(size) = config.indexer.limits.file_size.as_deref() {
    info!("Max file size: {} bytes", size);
  } else {
    info!("Max file size: 5MB");
  }

  // Max chunk size (configured)
  match config.indexer.limits.max_chunk_size {
    Some(sz) => info!("Max chunk size (configured): {} tokens", sz),
    None => info!("Max chunk size (configured): not set"),
  }

  // Embedding provider/model/dimensions (configured)
  let provider_name = config.embeddings.provider.clone();
  let (model_name, embedding_dim) = match provider_name.as_str() {
    "local" => {
      let model = config
        .embeddings
        .local
        .as_ref()
        .map(|l| l.model.clone())
        .unwrap_or_else(|| "(default) BAAI/bge-small-en-v1.5".to_string());
      let dim = config
        .embeddings
        .local
        .as_ref()
        .and_then(|l| l.embedding_dim)
        .map(|d| d.to_string())
        .unwrap_or_else(|| "unknown".to_string());
      (model, dim)
    }
    "voyage" => {
      let model = config
        .embeddings
        .voyage
        .as_ref()
        .map(|v| v.model.clone())
        .unwrap_or_else(|| "(unset)".to_string());
      let dim = config
        .embeddings
        .voyage
        .as_ref()
        .and_then(|v| v.embedding_dim)
        .map(|d| d.to_string())
        .unwrap_or_else(|| "unknown".to_string());
      (model, dim)
    }
    other => {
      // OpenAI-like (custom) provider
      if let Some(p) = config.embeddings.providers.get(other) {
        let dim = p
          .embedding_dim
          .map(|d| d.to_string())
          .unwrap_or_else(|| "unknown".to_string());
        (p.model.clone(), dim)
      } else {
        ("(unset)".to_string(), "unknown".to_string())
      }
    }
  };
  info!("Embedding provider: {}", provider_name);
  info!("Embedding model: {}", model_name);
  info!("Embedding dimensions: {}", embedding_dim);

  // Convert indexer config
  let indexer_config = match config.to_indexer_config() {
    Ok(cfg) => cfg,
    Err(e) => {
      error!("Failed to convert indexer config: {}", e);
      return Err(1);
    }
  };

  // Log effective (optimal) chunk size based on provider/tier
  let effective_chunk = indexer_config.optimal_chunk_size();
  info!("Max chunk size (effective): {} tokens", effective_chunk);

  // Create server config from breeze config
  let server_config = breeze_server::Config {
    tls_enabled: !config.server.tls.disabled,
    domains: config
      .server
      .tls
      .letsencrypt
      .as_ref()
      .map(|le| le.domains.clone())
      .unwrap_or_default(),
    email: config
      .server
      .tls
      .letsencrypt
      .as_ref()
      .map(|le| le.emails.clone())
      .unwrap_or_default(),
    cache: config
      .server
      .tls
      .letsencrypt
      .as_ref()
      .map(|le| le.cert_dir.clone()),
    production: config
      .server
      .tls
      .letsencrypt
      .as_ref()
      .map(|le| le.production)
      .unwrap_or(false),
    tls_key: config
      .server
      .tls
      .keypair
      .as_ref()
      .map(|kp| PathBuf::from(&kp.tls_key)),
    tls_cert: config
      .server
      .tls
      .keypair
      .as_ref()
      .map(|kp| PathBuf::from(&kp.tls_cert)),
    https_port: config.server.ports.https,
    http_port: config.server.ports.http,
    indexer: indexer_config,
  };

  // Run the server
  match breeze_server::run(server_config, Some(shutdown_token)).await {
    Ok(_) => {
      tracing::debug!("Breeze server stopped successfully");
      Ok(())
    }
    Err(e) => {
      error!("Server error: {}", e);
      Err(1)
    }
  }
}
