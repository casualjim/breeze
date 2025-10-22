use crate::{
  Tokenizer,
  chunker::InnerChunker,
  languages::get_language,
  types::{Chunk, ChunkError, ProjectChunk},
};
use blake3::Hasher;
use futures::{Stream, StreamExt};
use ignore::{DirEntry, WalkBuilder, overrides::OverrideBuilder};

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
  collections::BTreeSet,
  path::{Path, PathBuf},
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

/// Default ignore patterns embedded from extra-ignores file
const DEFAULT_IGNORE_PATTERNS: &str = include_str!("../../../extra-ignores");

use std::sync::OnceLock;

/// Get or create the default ignore file path
fn get_default_ignore_file() -> &'static std::path::PathBuf {
  static DEFAULT_IGNORE_FILE: OnceLock<std::path::PathBuf> = OnceLock::new();

  DEFAULT_IGNORE_FILE.get_or_init(|| {
    // Create a temporary file with our default patterns
    let temp_dir = std::env::temp_dir();
    let ignore_path = temp_dir.join("breeze-default-ignore");

    // Write the default patterns to the file
    if let Err(e) = std::fs::write(&ignore_path, DEFAULT_IGNORE_PATTERNS) {
      error!("Failed to create default ignore file: {}", e);
    }

    ignore_path
  })
}

/// Options for walking a project directory
#[derive(Debug, Clone)]
pub struct WalkOptions {
  pub max_chunk_size: usize,
  pub tokenizer: Tokenizer,
  pub max_parallel: usize,
  pub max_file_size: Option<u64>,
  pub large_file_threads: usize,
  pub existing_hashes: std::collections::BTreeMap<PathBuf, [u8; 32]>,
  pub cancel_token: Option<CancellationToken>,
}

impl Default for WalkOptions {
  fn default() -> Self {
    Self {
      max_chunk_size: 1000,
      tokenizer: Tokenizer::Characters,
      max_parallel: 4,
      max_file_size: Some(5 * 1024 * 1024), // 5MB default
      large_file_threads: 4,
      existing_hashes: std::collections::BTreeMap::new(),
      cancel_token: None,
    }
  }
}

/// Walk a project directory with options
pub fn walk_project(
  path: impl AsRef<Path>,
  options: WalkOptions,
) -> impl Stream<Item = Result<ProjectChunk, ChunkError>> {
  let path = path.as_ref().to_owned();
  let max_file_size = options.max_file_size;
  let cancel_token = options.cancel_token.clone();

  // Create a channel for streaming results
  let (tx, rx) = mpsc::channel::<Result<ProjectChunk, ChunkError>>(options.max_parallel * 2);

  // Create the chunker upfront
  let chunker = match InnerChunker::new(options.max_chunk_size, options.tokenizer) {
    Ok(c) => Arc::new(c),
    Err(e) => {
      // Send error and return early
      let tx_clone = tx.clone();
      tokio::spawn(async move {
        let _ = tx_clone.send(Err(e)).await;
      });
      return ReceiverStream::new(rx);
    }
  };

  tokio::spawn(async move {
    // Phase 1: Collect all files with their sizes
    let file_entries = match collect_files_with_sizes(&path, max_file_size).await {
      Ok(entries) => entries,
      Err(e) => {
        let _ = tx.send(Err(ChunkError::IoError(e))).await;
        return;
      }
    };

    if let Some(cancel) = &cancel_token
      && cancel.is_cancelled()
    {
      debug!("walk_project cancelled before processing files");
      return;
    }

    if file_entries.is_empty() {
      debug!("No files found to process");
      return;
    }

    info!("Collected {} files for processing", file_entries.len());

    // Find deleted files by comparing existing_hashes with collected files
    if !options.existing_hashes.is_empty() {
      let collected_paths: BTreeSet<PathBuf> =
        file_entries.iter().map(|(path, _)| path.clone()).collect();
      let tx_clone = tx.clone();
      let existing_hashes = options.existing_hashes.clone();
      let cancel_clone = cancel_token.clone();

      // Emit delete chunks in a separate task to avoid blocking
      tokio::spawn(async move {
        let mut deleted_count = 0;

        for (existing_path, _) in existing_hashes {
          if let Some(cancel) = &cancel_clone
            && cancel.is_cancelled()
          {
            debug!("walk_project deletion emission cancelled");
            break;
          }
          if !collected_paths.contains(&existing_path) {
            // This file was in the index but no longer exists
            deleted_count += 1;
            let delete_chunk = ProjectChunk {
              file_path: existing_path.to_string_lossy().to_string(),
              chunk: Chunk::Delete {
                file_path: existing_path.to_string_lossy().to_string(),
              },
              file_size: 0, // File no longer exists
            };
            if tx_clone.send(Ok(delete_chunk)).await.is_err() {
              break;
            }
          }
        }

        if deleted_count > 0 {
          info!(
            "Detected {} deleted files to remove from index",
            deleted_count
          );
        }
      });
    }

    // Sort by size (largest first)
    let mut file_entries = file_entries;
    file_entries.sort_by_key(|(_, size)| std::cmp::Reverse(*size));

    // Log size distribution
    let total_size: u64 = file_entries.iter().map(|(_, size)| size).sum();
    let largest_size = file_entries.first().map(|(_, size)| *size).unwrap_or(0);
    let smallest_size = file_entries.last().map(|(_, size)| *size).unwrap_or(0);
    info!(
      "File size distribution: {} files, total: {} MB, largest: {} KB, smallest: {} bytes",
      file_entries.len(),
      total_size / (1024 * 1024),
      largest_size / 1024,
      smallest_size
    );

    // Phase 2: Process files with dual-pool work-stealing
    process_with_dual_pools(
      file_entries,
      chunker,
      tx,
      options.large_file_threads,
      options.max_parallel,
      Arc::new(options.existing_hashes),
      cancel_token,
    )
    .await;
  });

  ReceiverStream::new(rx)
}

/// Check if an entry should be traversed (for directories) or processed (for files)
fn should_process_entry(entry: &DirEntry) -> bool {
  // Always traverse directories
  if entry.file_type().is_some_and(|ft| ft.is_dir()) {
    return true;
  }

  // For files, apply our filtering logic
  if !entry.file_type().is_some_and(|ft| ft.is_file()) {
    return false;
  }

  let path = entry.path();

  // Skip empty files
  if let Ok(metadata) = entry.metadata()
    && metadata.len() == 0
  {
    debug!("Skipping empty file: {}", path.display());
    return false;
  }

  // Skip binary files
  let is_text_file = if let Ok(Some(file_type)) = infer::get_from_path(path) {
    file_type.matcher_type() == infer::MatcherType::Text
  } else {
    // If infer can't determine, check with hyperpolyglot
    hyperpolyglot::detect(path).is_ok()
  };

  if !is_text_file {
    debug!("Skipping binary file: {}", path.display());
    return false;
  }

  true
}

/// Decide if a single path would be included by the walker, using the same
/// ignore/VCS pruning semantics. This is intended for file watcher events.
pub fn walker_includes_path(
  project_root: impl AsRef<Path>,
  path: impl AsRef<Path>,
  max_file_size: Option<u64>,
) -> bool {
  let project_root = project_root.as_ref();
  let path = path.as_ref();

  // If the path is outside the project, exclude it
  if let Ok(abs) = std::fs::canonicalize(path)
    && let Ok(root) = std::fs::canonicalize(project_root)
    && !abs.starts_with(&root)
  {
    return false;
  }

  // Build an override that only includes the specific path, so the walker
  // applies ignore and VCS pruning but we don't traverse the entire tree.
  let mut ob = OverrideBuilder::new(project_root);
  // Use a relative pattern if possible for portability
  let candidate = match path.strip_prefix(project_root) {
    Ok(rel) => rel,
    Err(_) => path,
  };
  // Override expects Unix-style separators in patterns; Path display is fine
  ob.add(&candidate.to_string_lossy()).ok();
  let overrides = match ob.build() {
    Ok(o) => o,
    Err(_) => return false,
  };

  let mut builder = WalkBuilder::new(project_root);

  // Default and custom ignores, matching walk_project
  let default_ignore = get_default_ignore_file();
  if default_ignore.exists() {
    builder.add_ignore(default_ignore);
  }
  builder
    .overrides(overrides)
    .max_filesize(max_file_size)
    .add_custom_ignore_filename(".breezeignore");

  // Build the iterator and check whether our file appears
  for ent in builder.build().flatten() {
    let p = ent.path();
    if p == path {
      // Only include files; directories are always traversable in walker
      if let Some(ft) = ent.file_type() {
        return ft.is_file();
      }
      return false;
    }
  }
  false
}

/// Process a stream of file paths with options
pub fn walk_files<S>(
  files: S,
  project_root: impl AsRef<Path>,
  options: WalkOptions,
) -> impl Stream<Item = Result<ProjectChunk, ChunkError>>
where
  S: Stream<Item = PathBuf> + Send + 'static,
{
  let project_root = project_root.as_ref().to_owned();
  let max_file_size = options.max_file_size;
  let cancel_token = options.cancel_token.clone();

  // Create a channel for streaming results
  let (tx, rx) = mpsc::channel::<Result<ProjectChunk, ChunkError>>(options.max_parallel * 2);

  // Create the chunker upfront
  let chunker = match InnerChunker::new(options.max_chunk_size, options.tokenizer) {
    Ok(c) => Arc::new(c),
    Err(e) => {
      // Send error and return early
      let tx_clone = tx.clone();
      tokio::spawn(async move {
        let _ = tx_clone.send(Err(e)).await;
      });
      return ReceiverStream::new(rx);
    }
  };

  tokio::spawn(async move {
    // Phase 1: Collect files from stream with their sizes
    let mut file_entries = Vec::new();
    let mut files = Box::pin(files);

    while let Some(path) = files.next().await {
      if let Some(cancel) = &cancel_token
        && cancel.is_cancelled()
      {
        debug!("walk_files cancelled before file collection complete");
        break;
      }
      // First check if file exists
      match tokio::fs::metadata(&path).await {
        Ok(meta) => {
          // File exists, check if walker would include this file (inherit ignores/VCS pruning)
          if walker_includes_path(&project_root, &path, max_file_size) {
            let size = meta.len();
            file_entries.push((path, size));
          }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
          // File doesn't exist - emit a Delete chunk immediately
          let path_str = path.to_string_lossy().to_string();
          let delete_chunk = ProjectChunk {
            file_path: path_str.clone(),
            chunk: Chunk::Delete {
              file_path: path_str,
            },
            file_size: 0, // File doesn't exist
          };
          if let Err(send_err) = tx.send(Ok(delete_chunk)).await {
            debug!("Failed to send delete chunk: {}", send_err);
            return;
          }
        }
        Err(e) => {
          debug!("Failed to get metadata for {}: {}", path.display(), e);
        }
      }
    }

    if file_entries.is_empty() {
      debug!("No valid files found to process");
      return;
    }

    info!("Collected {} files for processing", file_entries.len());

    // Sort by size (largest first) - same as walk_project
    file_entries.sort_by_key(|(_, size)| std::cmp::Reverse(*size));

    // Log size distribution
    let total_size: u64 = file_entries.iter().map(|(_, size)| size).sum();
    let largest_size = file_entries.first().map(|(_, size)| *size).unwrap_or(0);
    let smallest_size = file_entries.last().map(|(_, size)| *size).unwrap_or(0);
    info!(
      "File size distribution: {} files, total: {} MB, largest: {} KB, smallest: {} bytes",
      file_entries.len(),
      total_size / (1024 * 1024),
      largest_size / 1024,
      smallest_size
    );

    // Phase 2: Process files with dual-pool work-stealing (share implementation)
    process_with_dual_pools(
      file_entries,
      chunker,
      tx,
      options.large_file_threads,
      options.max_parallel,
      Arc::new(options.existing_hashes),
      cancel_token,
    )
    .await;
  });

  ReceiverStream::new(rx)
}

/// Collect all files with their sizes (without reading content)
async fn collect_files_with_sizes(
  path: &Path,
  max_file_size: Option<u64>,
) -> Result<Vec<(PathBuf, u64)>, std::io::Error> {
  // Use tokio's spawn_blocking for the walker
  let path = path.to_owned();

  tokio::task::spawn_blocking(move || {
    let mut entries = Vec::new();
    let mut builder = WalkBuilder::new(&path);

    // Add our default ignore patterns
    let default_ignore = get_default_ignore_file();
    if default_ignore.exists() {
      builder.add_ignore(default_ignore);
    }

    // Filter entries using the same logic we use downstream (no binary/empty files)
    builder
      .max_filesize(max_file_size)
      .filter_entry(should_process_entry)
      .add_custom_ignore_filename(".breezeignore");

    for entry in builder.build().flatten() {
      if let Some(file_type) = entry.file_type()
        && file_type.is_file()
        && let Ok(metadata) = entry.metadata()
      {
        let size = metadata.len();
        if size > 0 {
          entries.push((entry.path().to_owned(), size));
        }
      }
    }
    Ok(entries)
  })
  .await?
}

/// Process files using dual-pool work-stealing approach
async fn process_with_dual_pools(
  file_entries: Vec<(PathBuf, u64)>,
  chunker: Arc<InnerChunker>,
  tx: mpsc::Sender<Result<ProjectChunk, ChunkError>>,
  large_file_threads: usize,
  small_file_threads: usize,
  existing_hashes: Arc<std::collections::BTreeMap<PathBuf, [u8; 32]>>,
  cancel_token: Option<CancellationToken>,
) {
  use std::collections::VecDeque;
  use std::sync::Mutex;

  // Create a deque that allows taking from both ends
  let work_queue = Arc::new(Mutex::new(VecDeque::from(file_entries.clone())));

  // Track total work and completed work
  let total_files = file_entries.len();
  let remaining_work = Arc::new(AtomicUsize::new(total_files));
  let skipped_files = Arc::new(AtomicUsize::new(0));
  let skipped_size = Arc::new(AtomicUsize::new(0));

  info!(
    "Starting dual-pool processing: {} total files, {} large file threads, {} small file threads",
    total_files, large_file_threads, small_file_threads
  );

  // Spawn large file workers (take from front)
  let mut handles = Vec::new();
  for i in 0..large_file_threads {
    let work_queue = work_queue.clone();
    let chunker = chunker.clone();
    let tx = tx.clone();
    let remaining = remaining_work.clone();
    let existing_hashes = existing_hashes.clone();
    let skipped_files = skipped_files.clone();
    let skipped_size = skipped_size.clone();
    let cancel_token = cancel_token.clone();

    let handle = tokio::spawn(async move {
      debug!("Large file worker {} started", i);
      let mut processed = 0;
      let mut total_size_processed = 0u64;

      loop {
        if let Some(cancel) = &cancel_token
          && cancel.is_cancelled()
        {
          debug!("Large file worker {} cancelled", i);
          break;
        }
        // Take from the front (largest files)
        let work_item = {
          let mut queue = work_queue.lock().unwrap();
          queue.pop_front()
        };

        match work_item {
          Some((path, size)) => {
            debug!(
              "Large file worker {} processing: {} ({} KB)",
              i,
              path.display(),
              size / 1024
            );
            total_size_processed += size;

            let existing_hash = existing_hashes.get(&path).cloned();

            // Check if file will be skipped
            let mut chunk_count = 0;
            let mut stream = Box::pin(process_file(&path, size, chunker.clone(), existing_hash));
            while let Some(result) = stream.next().await {
              chunk_count += 1;
              if tx.send(result).await.is_err() {
                debug!("Large file worker {} exiting: receiver dropped", i);
                return;
              }
            }

            // If no chunks were produced, the file was skipped
            if chunk_count == 0 && existing_hash.is_some() {
              skipped_files.fetch_add(1, Ordering::Relaxed);
              skipped_size.fetch_add(size as usize, Ordering::Relaxed);
            }

            processed += 1;
            let remaining_count = remaining.fetch_sub(1, Ordering::SeqCst) - 1;
            if remaining_count % 100 == 0 && remaining_count > 0 {
              debug!("Progress: {} files remaining", remaining_count);
            }
          }
          None => {
            debug!("Large file worker {} found empty queue, exiting", i);
            break;
          }
        }
      }

      info!(
        "Large file worker {} completed: processed {} files, {} MB total",
        i,
        processed,
        total_size_processed / (1024 * 1024)
      );
    });
    handles.push(handle);
  }

  // Spawn small file workers (take from back)
  for i in 0..small_file_threads {
    let work_queue = work_queue.clone();
    let chunker = chunker.clone();
    let tx = tx.clone();
    let remaining = remaining_work.clone();
    let existing_hashes = existing_hashes.clone();
    let skipped_files = skipped_files.clone();
    let skipped_size = skipped_size.clone();
    let cancel_token = cancel_token.clone();

    let handle = tokio::spawn(async move {
      debug!("Small file worker {} started", i);
      let mut processed = 0;
      let mut total_size_processed = 0u64;

      loop {
        if let Some(cancel) = &cancel_token
          && cancel.is_cancelled()
        {
          debug!("Small file worker {} cancelled", i);
          break;
        }
        // Take from the back (smallest files)
        let work_item = {
          let mut queue = work_queue.lock().unwrap();
          queue.pop_back()
        };

        match work_item {
          Some((path, size)) => {
            total_size_processed += size;

            let existing_hash = existing_hashes.get(&path).cloned();

            // Check if file will be skipped
            let mut chunk_count = 0;
            let mut stream = Box::pin(process_file(&path, size, chunker.clone(), existing_hash));
            while let Some(result) = stream.next().await {
              chunk_count += 1;
              if tx.send(result).await.is_err() {
                debug!("Small file worker {} exiting: receiver dropped", i);
                return;
              }
            }

            // If no chunks were produced, the file was skipped
            if chunk_count == 0 && existing_hash.is_some() {
              skipped_files.fetch_add(1, Ordering::Relaxed);
              skipped_size.fetch_add(size as usize, Ordering::Relaxed);
            }

            processed += 1;
            let remaining_count = remaining.fetch_sub(1, Ordering::SeqCst) - 1;
            if remaining_count > 0 && (remaining_count % 100 == 0 || remaining_count < 10) {
              debug!("Progress: {} files remaining", remaining_count);
            }
          }
          None => {
            debug!("Small file worker {} found empty queue, exiting", i);
            break;
          }
        }
      }

      info!(
        "Small file worker {} completed: processed {} files, {} KB total",
        i,
        processed,
        total_size_processed / 1024
      );
    });
    handles.push(handle);
  }

  // Wait for all workers to complete
  for handle in handles {
    let _ = handle.await;
  }

  let skipped = skipped_files.load(Ordering::Relaxed);
  let skipped_mb = skipped_size.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
  let processed = total_files - skipped;

  info!(
    "File processing complete: {} total files, {} processed, {} skipped ({:.2} MB saved)",
    total_files, processed, skipped, skipped_mb
  );
}

/// Process a single file and yield chunks as a stream
fn process_file<P: AsRef<Path>>(
  path: P,
  file_size: u64,
  chunker: Arc<InnerChunker>,
  existing_hash: Option<[u8; 32]>,
) -> impl Stream<Item = Result<ProjectChunk, ChunkError>> + Send {
  let path = path.as_ref().to_owned();

  async_stream::try_stream! {
      let path_str = path.to_string_lossy().to_string();


      // First check if it's a text file using infer (this only reads a few bytes)
      let is_text_file = if let Ok(Some(file_type)) = infer::get_from_path(&path) {
          file_type.matcher_type() == infer::MatcherType::Text
      } else {
          // If infer can't determine, check with hyperpolyglot
          hyperpolyglot::detect(&path).is_ok()
      };

      if !is_text_file {
          return; // Skip binary files
      }

      // Now we know it's a text file, read it once
      let content = match tokio::fs::read_to_string(&path).await {
          Ok(content) => content,
          Err(e) => {
              if e.kind() == std::io::ErrorKind::NotFound {
                  // File was deleted between collection and reading
                  debug!("File deleted during processing: {}", path.display());
                  yield ProjectChunk {
                      file_path: path_str.clone(),
                      chunk: Chunk::Delete {
                          file_path: path_str.clone(),
                      },
                      file_size,
                  };
                  return;
              } else {
                  // Propagate other IO errors
                  Err(ChunkError::IoError(e))?;
                  unreachable!()
              }
          }
      };

      if content.is_empty() {
          return;
      }

      // Compute hash of the content
      let mut hasher = Hasher::new();
      hasher.update(content.as_bytes());
      let hash = hasher.finalize();
      let mut content_hash = [0u8; 32];
      content_hash.copy_from_slice(hash.as_bytes());

      // Check if file has changed
      if let Some(existing) = existing_hash
          && existing == content_hash
      {
          // File unchanged, skip it entirely
          debug!("Skipping unchanged file: {}", path.display());
          return;
      }

      // Check if hyperpolyglot detects a supported language
      let detected_language = if let Ok(Some(detection)) = hyperpolyglot::detect(&path) {
          let language = detection.language();
          if get_language(language).is_some() {
              Some(language.to_string())
          } else {
              None
          }
      } else {
          None
      };

      // Clone content once for the EOF marker
      let content_for_eof = content.clone();

      // Try semantic chunking first if we have a supported language
      if let Some(language) = detected_language {
          let mut chunk_stream = Box::pin(chunker.chunk_code(content, language.clone(), Some(path_str.clone())));
          let mut had_success = false;
          let mut chunk_count = 0;

          while let Some(chunk_result) = chunk_stream.next().await {
              match chunk_result {
                  Ok(chunk) => {
                      had_success = true;
                      chunk_count += 1;
                      yield ProjectChunk {
                          file_path: path_str.clone(),
                          chunk,
                          file_size,
                      };
                  }
                  Err(_) => {
                      // If we haven't had any successful chunks, fall through to text chunking
                      if !had_success {
                          break;
                      }
                  }
              }
          }

          // If semantic chunking succeeded, we're done
          if had_success {
              // Emit EOF marker with chunk count
              yield ProjectChunk {
                  file_path: path_str.clone(),
                  chunk: Chunk::EndOfFile {
                      file_path: path_str.clone(),
                      content: content_for_eof,
                      content_hash,
                      expected_chunks: chunk_count,
                  },
                  file_size,
              };
              return;
          }
      }

      // Fall back to text chunking
      // Note: content was already moved into chunk_code if we tried semantic chunking
      let content_for_text = content_for_eof.clone();
      let mut chunk_count = 0;

      let mut chunk_stream = Box::pin(chunker.chunk_text(content_for_text, Some(path_str.clone())));
      while let Some(chunk_result) = chunk_stream.next().await {
          let chunk = chunk_result?;
          chunk_count += 1;
          yield ProjectChunk {
              file_path: path_str.clone(),
              chunk,
              file_size,
          };
      }

      // Emit EOF marker with chunk count
      yield ProjectChunk {
          file_path: path_str.clone(),
          chunk: Chunk::EndOfFile {
              file_path: path_str,
              content: content_for_eof,
              content_hash,
              expected_chunks: chunk_count,
          },
          file_size,
      };
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::types::Chunk;
  use std::fs;
  use tempfile::TempDir;
  use tokio_stream::StreamExt;

  async fn create_test_project() -> TempDir {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();

    // Create a simple project structure
    fs::create_dir_all(base_path.join("src")).unwrap();
    fs::create_dir_all(base_path.join("tests")).unwrap();
    fs::create_dir_all(base_path.join("docs")).unwrap();
    fs::create_dir_all(base_path.join("scripts")).unwrap();
    fs::create_dir_all(base_path.join(".git")).unwrap();

    // Create a Python module structure
    fs::write(
      base_path.join("src/__init__.py"),
      r#"""Main package for the test project."""

__version__ = "0.1.0"
__author__ = "Test Author"

from .calculator import Calculator
from .utils import factorial

__all__ = ["Calculator", "factorial"]
"#,
    )
    .unwrap();

    fs::write(
      base_path.join("src/calculator.py"),
      r#"""Calculator module with basic arithmetic operations."""

class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.memory = 0

    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.memory = result
        return result

    def subtract(self, a, b):
        """Subtract b from a."""
        result = a - b
        self.memory = result
        return result

    def multiply(self, a, b):
        """Multiply two numbers."""
        result = a * b
        self.memory = result
        return result

    def clear_memory(self):
        """Clear the calculator memory."""
        self.memory = 0
"#,
    )
    .unwrap();

    // Create Python file
    fs::write(
      base_path.join("scripts/test.py"),
      r#"#!/usr/bin/env python3
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value += x
        return self.value

if __name__ == "__main__":
    print(f"5! = {factorial(5)}")
"#,
    )
    .unwrap();

    // Create another Python file
    fs::write(
      base_path.join("scripts/data_processor.py"),
      r#"#!/usr/bin/env python3
"""Data processing utilities."""

import json
import csv
from typing import List, Dict, Any

class DataProcessor:
    """Process various data formats."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = []

    def load_json(self, filepath: str) -> List[Dict]:
        """Load data from JSON file."""
        with open(filepath, 'r') as f:
            self.data = json.load(f)
        return self.data

    def process_records(self) -> List[Dict]:
        """Process loaded records."""
        processed = []
        for record in self.data:
            if self._validate_record(record):
                processed.append(self._transform_record(record))
        return processed

    def _validate_record(self, record: Dict) -> bool:
        """Validate a single record."""
        required_fields = self.config.get('required_fields', [])
        return all(field in record for field in required_fields)

    def _transform_record(self, record: Dict) -> Dict:
        """Transform a single record."""
        # Apply transformations based on config
        return record
"#,
    )
    .unwrap();

    // Create a markdown file
    fs::write(
      base_path.join("README.md"),
      r#"# Test Project

This is a test project for the walker functionality.

## Features
- Python modules and packages
- Data processing utilities
- Calculator implementations
- Documentation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src import Calculator

calc = Calculator()
result = calc.add(5, 3)
print(f"Result: {result}")
```

## Testing

Run tests with pytest:

```bash
python -m pytest tests/
```
"#,
    )
    .unwrap();

    // Create a binary file (should be skipped)
    fs::write(
      base_path.join("test.bin"),
      [0u8, 1, 2, 3, 255, 254, 253, 252],
    )
    .unwrap();

    // Create .gitignore
    fs::write(base_path.join(".gitignore"), "target/\n*.log\n").unwrap();

    // Create a file in .git (should be ignored)
    fs::write(
      base_path.join(".git/config"),
      "[core]\nrepositoryformatversion = 0\n",
    )
    .unwrap();

    // Create a requirements file
    fs::write(
      base_path.join("requirements.txt"),
      "pytest>=7.0.0\nnumpy>=1.20.0\npandas>=1.3.0\n",
    )
    .unwrap();

    temp_dir
  }

  #[tokio::test]
  async fn test_walk_project_basic() {
    let temp_dir = create_test_project().await;
    let path = temp_dir.path();

    let mut chunks = Vec::new();
    let mut stream = walk_project(
      path,
      WalkOptions {
        max_chunk_size: 500,
        tokenizer: Tokenizer::Characters,
        max_parallel: 4,
        max_file_size: None,
        large_file_threads: 2,
        existing_hashes: std::collections::BTreeMap::new(),
        cancel_token: None,
      },
    );

    while let Some(result) = stream.next().await {
      match result {
        Ok(chunk) => chunks.push(chunk),
        Err(e) => panic!("Unexpected error: {}", e),
      }
    }

    // Should have found and chunked multiple files
    assert!(!chunks.is_empty(), "Should have found some chunks");

    // Check we got chunks from different files
    let unique_files: std::collections::HashSet<_> = chunks.iter().map(|c| &c.file_path).collect();
    assert!(
      unique_files.len() > 1,
      "Should have chunks from multiple files"
    );

    // Check we have both semantic and text chunks
    let has_semantic = chunks.iter().any(|c| c.is_semantic());
    let has_text = chunks.iter().any(|c| c.is_text());
    assert!(has_semantic, "Should have semantic chunks");
    assert!(has_text, "Should have text chunks");

    // Debug: print all file paths
    let file_paths: Vec<_> = chunks.iter().map(|c| &c.file_path).collect();
    println!("Found files: {:?}", file_paths);

    // Verify .git files were ignored
    assert!(
      !chunks.iter().any(|c| c.file_path.contains(".git")),
      ".git files should be ignored"
    );

    // Verify binary files were skipped
    assert!(
      !chunks.iter().any(|c| c.file_path.contains("test.bin")),
      "Binary files should be skipped"
    );
  }

  #[tokio::test]
  async fn test_walk_project_languages() {
    let temp_dir = create_test_project().await;
    let path = temp_dir.path();

    let mut chunks = Vec::new();
    let mut stream = walk_project(
      path,
      WalkOptions {
        max_chunk_size: 1000,
        tokenizer: Tokenizer::Characters,
        max_parallel: 2,
        max_file_size: None,
        large_file_threads: 2,
        existing_hashes: std::collections::BTreeMap::new(),
        cancel_token: None,
      },
    );

    while let Some(result) = stream.next().await {
      chunks.push(result.unwrap());
    }

    // Check Python files (the only supported language currently)
    let python_chunks: Vec<_> = chunks
      .iter()
      .filter(|c| c.file_path.ends_with(".py") && c.is_semantic())
      .collect();
    assert!(
      !python_chunks.is_empty(),
      "Should have Python semantic chunks"
    );
    assert!(python_chunks.iter().all(|c| match &c.chunk {
      Chunk::Semantic(sc) => sc.metadata.language == "Python",
      _ => false,
    }));

    // Check Markdown files (should be semantic chunks)
    let md_chunks: Vec<_> = chunks
      .iter()
      .filter(|c| c.file_path.ends_with(".md") && !matches!(c.chunk, Chunk::EndOfFile { .. }))
      .collect();
    assert!(!md_chunks.is_empty(), "Should have Markdown chunks");
    assert!(md_chunks.iter().all(|c| c.is_semantic()));
  }

  #[tokio::test]
  async fn test_walk_project_empty_dir() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();

    let mut chunks = Vec::new();
    let mut stream = walk_project(
      path,
      WalkOptions {
        max_chunk_size: 500,
        tokenizer: Tokenizer::Characters,
        max_parallel: 4,
        max_file_size: None,
        large_file_threads: 2,
        existing_hashes: std::collections::BTreeMap::new(),
        cancel_token: None,
      },
    );

    while let Some(result) = stream.next().await {
      chunks.push(result.unwrap());
    }

    assert!(chunks.is_empty(), "Empty directory should yield no chunks");
  }

  #[tokio::test]
  async fn test_walk_project_concurrency() {
    let temp_dir = create_test_project().await;
    let path = temp_dir.path();

    // Test with different concurrency levels
    for max_parallel in [1, 2, 8] {
      let mut chunks = Vec::new();
      let mut stream = walk_project(
        path,
        WalkOptions {
          max_chunk_size: 500,
          tokenizer: Tokenizer::Characters,
          max_parallel,
          max_file_size: None,
          large_file_threads: 2,
          existing_hashes: std::collections::BTreeMap::new(),
          cancel_token: None,
        },
      );

      while let Some(result) = stream.next().await {
        chunks.push(result.unwrap());
      }

      assert!(
        !chunks.is_empty(),
        "Should get chunks with concurrency level {}",
        max_parallel
      );
    }
  }

  #[tokio::test]
  async fn test_process_file_stream() {
    let temp_dir = TempDir::new().unwrap();
    let rust_file = temp_dir.path().join("test.rs");

    fs::write(
      &rust_file,
      r#"
fn main() {
    println!("Test");
}

fn helper() {
    let x = 42;
}
"#,
    )
    .unwrap();

    let chunker = Arc::new(InnerChunker::new(100, Tokenizer::Characters).unwrap());

    let file_size = fs::metadata(&rust_file).unwrap().len();
    let mut stream = Box::pin(process_file(&rust_file, file_size, chunker, None));

    let mut chunks = Vec::new();
    while let Some(result) = stream.next().await {
      chunks.push(result.unwrap());
    }

    assert!(!chunks.is_empty(), "Should get chunks from Rust file");

    // Filter out EOF chunks for semantic checks
    let semantic_chunks: Vec<_> = chunks
      .iter()
      .filter(|c| !matches!(c.chunk, Chunk::EndOfFile { .. }))
      .collect();

    assert!(!semantic_chunks.is_empty(), "Should have semantic chunks");
    assert!(semantic_chunks.iter().all(|c| c.is_semantic()));
    assert!(semantic_chunks.iter().all(|c| match &c.chunk {
      Chunk::Semantic(sc) => sc.metadata.language == "Rust",
      _ => false,
    }));

    // Should have exactly one EOF chunk at the end
    assert!(matches!(
      chunks.last().unwrap().chunk,
      Chunk::EndOfFile { .. }
    ));
  }

  #[tokio::test]
  async fn test_work_avoidance_hash_comparison() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.rs");
    let content = r#"
fn main() {
    println!("Hello, world!");
}
"#;

    fs::write(&test_file, content).unwrap();

    let chunker = Arc::new(InnerChunker::new(100, Tokenizer::Characters).unwrap());

    // First, process the file without any existing hashes
    let file_size = fs::metadata(&test_file).unwrap().len();
    let mut stream = Box::pin(process_file(&test_file, file_size, chunker.clone(), None));

    let mut chunks = Vec::new();
    while let Some(result) = stream.next().await {
      chunks.push(result.unwrap());
    }

    assert!(!chunks.is_empty(), "Should get chunks on first run");

    // Extract the hash from the EOF chunk
    let eof_chunk = chunks
      .iter()
      .find(|c| matches!(c.chunk, Chunk::EndOfFile { .. }))
      .unwrap();
    let content_hash = match &eof_chunk.chunk {
      Chunk::EndOfFile { content_hash, .. } => *content_hash,
      _ => panic!("Expected EOF chunk"),
    };

    // Now process the same file again with the hash
    let file_size = fs::metadata(&test_file).unwrap().len();
    let mut stream = Box::pin(process_file(
      &test_file,
      file_size,
      chunker.clone(),
      Some(content_hash),
    ));

    let mut chunks = Vec::new();
    while let Some(result) = stream.next().await {
      chunks.push(result.unwrap());
    }

    assert!(
      chunks.is_empty(),
      "Should get no chunks when file is unchanged"
    );

    // Now modify the file and process again
    let new_content = r#"
fn main() {
    println!("Hello, world!");
    println!("Modified!");
}
"#;
    fs::write(&test_file, new_content).unwrap();

    // Process with the old hash - should get chunks because file changed
    let file_size = fs::metadata(&test_file).unwrap().len();
    let mut stream = Box::pin(process_file(
      &test_file,
      file_size,
      chunker,
      Some(content_hash),
    ));

    let mut chunks = Vec::new();
    while let Some(result) = stream.next().await {
      chunks.push(result.unwrap());
    }

    assert!(
      !chunks.is_empty(),
      "Should get chunks when file is modified"
    );
  }

  #[tokio::test]
  async fn test_eof_chunk_expected_chunks() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("test.py");

    // Create a Python file that will generate multiple chunks
    let content = r#"
def function_one():
    """First function with a long docstring that should help ensure we get multiple chunks when using a small chunk size."""
    return 1

def function_two():
    """Second function."""
    return 2

def function_three():
    """Third function."""
    return 3
"#;

    fs::write(&test_file, content).unwrap();

    // Use a small chunk size to ensure multiple chunks
    let chunker = Arc::new(InnerChunker::new(50, Tokenizer::Characters).unwrap());
    let file_size = fs::metadata(&test_file).unwrap().len();

    let mut stream = Box::pin(process_file(&test_file, file_size, chunker, None));

    let mut chunks = Vec::new();
    while let Some(result) = stream.next().await {
      chunks.push(result.unwrap());
    }

    // Find the EOF chunk
    let eof_chunk = chunks
      .iter()
      .find(|c| matches!(c.chunk, Chunk::EndOfFile { .. }))
      .expect("Should have an EOF chunk");

    // Count non-EOF chunks
    let content_chunks = chunks
      .iter()
      .filter(|c| !matches!(c.chunk, Chunk::EndOfFile { .. }))
      .count();

    // Extract expected_chunks from EOF
    let expected_chunks = match &eof_chunk.chunk {
      Chunk::EndOfFile {
        expected_chunks, ..
      } => *expected_chunks,
      _ => panic!("Expected EOF chunk"),
    };

    assert_eq!(
      expected_chunks, content_chunks,
      "EOF chunk should have correct expected_chunks count"
    );

    assert!(
      content_chunks > 1,
      "Should have multiple chunks with small chunk size"
    );
  }
}
