use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tiktoken_rs::{CoreBPE, cl100k_base, o200k_base, p50k_base, r50k_base};
use tokenizers::Tokenizer;
use tracing::debug;

use super::{
  EmbeddingInput, EmbeddingProvider,
  batching::{BatchingStrategy, TokenAwareBatchingStrategy},
};
use crate::config::OpenAILikeConfig;
use crate::reqwestx::api_client::{ApiClient, ApiClientConfig};

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
  input: Vec<String>,
  model: String,
  // OpenAI-specific optional field
  #[serde(skip_serializing_if = "Option::is_none")]
  encoding_format: Option<String>,
  // OpenAI-specific optional field for TE3 family
  #[serde(skip_serializing_if = "Option::is_none")]
  dimensions: Option<usize>,
  // Optional provider-specific fields
  #[serde(skip_serializing_if = "Option::is_none")]
  output_dtype: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  output_dimension: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
  data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
  index: usize,
  embedding: Vec<f32>,
}

/// Tokenizer wrapper that supports both tiktoken and HuggingFace
enum TokenizerWrapper {
  Tiktoken(CoreBPE),
  HuggingFace(Arc<Tokenizer>),
}

impl TokenizerWrapper {
  /// Count tokens in text
  fn count_tokens(&self, text: &str) -> Result<usize, tokenizers::Error> {
    match self {
      Self::Tiktoken(encoder) => Ok(encoder.encode_ordinary(text).len()),
      Self::HuggingFace(tokenizer) => tokenizer.encode(text, false).map(|encoding| encoding.len()),
    }
  }
}

/// OpenAI-compatible embedding provider
pub struct OpenAILikeEmbeddingProvider {
  client: ApiClient,
  model: String,
  tokenizer: TokenizerWrapper,
  embedding_dim: usize,
  context_length: usize,
  max_batch_size: usize,
  max_tokens_per_request: usize,
  // Optional request parameters controlled via config
  encoding_format: Option<String>,
  output_dtype: Option<String>,
  output_dimension: Option<usize>,
}

impl OpenAILikeEmbeddingProvider {
  pub async fn new(config: &OpenAILikeConfig) -> super::EmbeddingResult<Self> {
    // Create API client configuration
    let api_config = ApiClientConfig {
      base_url: config.api_base.clone(),
      api_key: config.api_key.clone(),
      max_concurrent_requests: config.max_concurrent_requests,
      max_requests_per_minute: config.requests_per_minute as usize,
      max_tokens_per_minute: config.tokens_per_minute as usize,
      max_retries: 3,
      timeout: Duration::from_secs(30),
    };

    let client = ApiClient::new(api_config)?;

    // Load tokenizer based on config
    let tokenizer = match &config.tokenizer {
      breeze_chunkers::Tokenizer::Tiktoken(encoding) => {
        let encoder = match encoding.as_str() {
          "cl100k_base" => cl100k_base()?,
          "p50k_base" => p50k_base()?,
          "r50k_base" => r50k_base()?,
          "o200k_base" => o200k_base()?,
          _ => {
            return Err(super::EmbeddingError::InvalidConfig(format!(
              "Unknown tiktoken encoding: {}",
              encoding
            )));
          }
        };
        TokenizerWrapper::Tiktoken(encoder)
      }
      breeze_chunkers::Tokenizer::PreloadedTiktoken(tiktoken) => TokenizerWrapper::Tiktoken(
        std::sync::Arc::try_unwrap(tiktoken.clone()).unwrap_or_else(|arc| (*arc).clone()),
      ),
      breeze_chunkers::Tokenizer::HuggingFace(model_id) => {
        let tokenizer = Tokenizer::from_pretrained(model_id, None).map_err(|e| {
          super::EmbeddingError::ModelLoadFailed(format!(
            "Failed to load tokenizer {}: {}",
            model_id, e
          ))
        })?;
        TokenizerWrapper::HuggingFace(Arc::new(tokenizer))
      }
      breeze_chunkers::Tokenizer::PreloadedHuggingFace(tokenizer) => {
        TokenizerWrapper::HuggingFace(tokenizer.clone())
      }
      breeze_chunkers::Tokenizer::Characters => {
        return Err(super::EmbeddingError::InvalidConfig(
          "Character tokenizer not supported for embeddings API".to_string(),
        ));
      }
    };

    Ok(Self {
      client,
      model: config.model.clone(),
      tokenizer,
      embedding_dim: config.embedding_dim,
      context_length: config.context_length,
      max_batch_size: config.max_batch_size,
      max_tokens_per_request: config
        .max_tokens_per_request
        .unwrap_or(config.context_length),
      encoding_format: config.encoding_format.clone(),
      output_dtype: config.output_dtype.clone(),
      output_dimension: config.output_dimension,
    })
  }
}

#[async_trait]
impl EmbeddingProvider for OpenAILikeEmbeddingProvider {
  async fn embed(&self, inputs: &[EmbeddingInput<'_>]) -> super::EmbeddingResult<Vec<Vec<f32>>> {
    let request = EmbeddingRequest {
      input: inputs.iter().map(|input| input.text.to_string()).collect(),
      model: self.model.clone(),
      encoding_format: self.encoding_format.clone(),
      dimensions: None, // OpenAI TE3 optional override not configured here
      output_dtype: self.output_dtype.clone(),
      output_dimension: self.output_dimension,
    };

    // Calculate total tokens for rate limiting
    let estimated_tokens: u32 = inputs
      .iter()
      .map(|input| {
        if let Some(count) = input.token_count {
          count as u32
        } else {
          // Count tokens using our tokenizer
          self
            .tokenizer
            .count_tokens(input.text)
            .unwrap_or_else(|_| input.text.len() / 4) as u32
        }
      })
      .sum();

    debug!(
      "Embedding {} texts with ~{} tokens",
      inputs.len(),
      estimated_tokens
    );

    // Make request with token count
    let embedding_response: EmbeddingResponse = self
      .client
      .post_json("/embeddings", &request, estimated_tokens)
      .await
      .map_err(|e| super::EmbeddingError::ApiError(e.to_string()))?;

    // Extract embeddings in order
    let mut embeddings = vec![vec![]; embedding_response.data.len()];
    for data in embedding_response.data {
      if data.index < embeddings.len() {
        embeddings[data.index] = data.embedding;
      }
    }

    Ok(embeddings)
  }

  fn embedding_dim(&self) -> usize {
    self.embedding_dim
  }

  fn context_length(&self) -> usize {
    self.context_length
  }

  fn create_batching_strategy(&self) -> Box<dyn BatchingStrategy> {
    // Use max_tokens_per_request for batching, which defaults to context_length
    // This allows users to configure smaller batches even if the model supports more
    Box::new(TokenAwareBatchingStrategy::new(
      self.max_tokens_per_request,
      self.max_batch_size,
    ))
  }

  fn tokenizer(&self) -> Option<Arc<tokenizers::Tokenizer>> {
    // Return HuggingFace tokenizer if available
    match &self.tokenizer {
      TokenizerWrapper::HuggingFace(tokenizer) => Some(tokenizer.clone()),
      TokenizerWrapper::Tiktoken(_) => None, // Tiktoken doesn't implement the trait
    }
  }

  fn is_remote(&self) -> bool {
    true
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::config::OpenAILikeConfig;

  #[tokio::test]
  async fn test_max_tokens_per_request() {
    // Test with max_tokens_per_request specified
    let config = OpenAILikeConfig {
      api_base: "http://localhost:8080/v1".to_string(),
      api_key: Some("test-key".to_string()),
      model: "test-model".to_string(),
      embedding_dim: 768,
      context_length: 8192,
      max_batch_size: 32,
      tokenizer: "tiktoken:cl100k_base".parse().unwrap(),
      requests_per_minute: 100,
      tokens_per_minute: 10000,
      max_concurrent_requests: 5,
      max_tokens_per_request: Some(4096), // Half of context_length
      encoding_format: None,
      output_dtype: None,
      output_dimension: None,
    };

    let provider = OpenAILikeEmbeddingProvider::new(&config).await.unwrap();
    assert_eq!(provider.max_tokens_per_request, 4096);

    // Verify batching strategy uses max_tokens_per_request
    let _strategy = provider.create_batching_strategy();
    // The strategy should be configured with max_tokens_per_request
  }

  #[tokio::test]
  async fn test_max_tokens_per_request_defaults_to_context_length() {
    // Test without max_tokens_per_request (should default to context_length)
    let config = OpenAILikeConfig {
      api_base: "http://localhost:8080/v1".to_string(),
      api_key: Some("test-key".to_string()),
      model: "test-model".to_string(),
      embedding_dim: 768,
      context_length: 8192,
      max_batch_size: 32,
      tokenizer: "tiktoken:cl100k_base".parse().unwrap(),
      requests_per_minute: 100,
      tokens_per_minute: 10000,
      max_concurrent_requests: 5,
      max_tokens_per_request: None, // Not specified
      encoding_format: None,
      output_dtype: None,
      output_dimension: None,
    };

    let provider = OpenAILikeEmbeddingProvider::new(&config).await.unwrap();
    assert_eq!(provider.max_tokens_per_request, 8192); // Should equal context_length
  }
}
