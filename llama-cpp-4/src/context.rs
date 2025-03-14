//! Safe wrapper around `llama_context`.

use std::fmt::{Debug, Formatter};
use std::num::NonZeroI32;
use std::ptr::NonNull;
use std::slice;

use llama_cpp_sys_4::llama_pooling_type;
use params::LlamaPoolingType;
use perf::PerfContextData;

use crate::llama_batch::LlamaBatch;
use crate::model::{LlamaLoraAdapter, LlamaModel};
use crate::token::data::LlamaTokenData;
use crate::token::data_array::LlamaTokenDataArray;
use crate::token::LlamaToken;
use crate::{
    DecodeError, EmbeddingsError, EncodeError, LlamaLoraAdapterRemoveError,
    LlamaLoraAdapterSetError,
};

pub mod kv_cache;
pub mod params;
pub mod perf;
pub mod session;

/// A safe wrapper around the `llama_context` C++ context.
///
/// This struct provides a safe interface to interact with the `llama_context` used by the `LlamaModel`.
/// It encapsulates the raw C++ context pointer and provides additional fields for managing the model and
/// context-specific settings like embeddings and logits.
///
/// The `LlamaContext` struct ensures that the C++ context is always valid by using the `NonNull` type for
/// the context pointer, preventing it from being null. The struct also holds a reference to the model
/// (`LlamaModel`) that the context is tied to, along with some internal state like whether embeddings are enabled
/// and the initialized logits for the context.
///
/// # Fields
///
/// - `context`: A non-null pointer to the raw C++ `llama_context`. This is the main context used for interacting with the model.
/// - `model`: A reference to the `LlamaModel` associated with this context. This model provides the data and parameters
///   that the context interacts with.
/// - `initialized_logits`: A vector used to store the initialized logits. These are used in the model's processing and
///   are kept separate from the context data.
/// - `embeddings_enabled`: A boolean flag indicating whether embeddings are enabled in the context. This is useful for
///   controlling whether embedding data is generated during the interaction with the model.
#[allow(clippy::module_name_repetitions)]
pub struct LlamaContext<'a> {
    pub(crate) context: NonNull<llama_cpp_sys_4::llama_context>,
    /// a reference to the contexts model.
    pub model: &'a LlamaModel,
    initialized_logits: Vec<i32>,
    embeddings_enabled: bool,
}

impl Debug for LlamaContext<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaContext")
            .field("context", &self.context)
            .finish()
    }
}

impl<'model> LlamaContext<'model> {
    /// Creates a new instance of `LlamaContext` with the provided model, context, and embeddings flag.
    ///
    /// This function initializes a new `LlamaContext` object, which is used to interact with the
    /// `LlamaModel`. The context is created from a pointer to a C++ context and the embeddings flag
    /// determines whether embeddings are enabled in the context.
    ///
    /// # Parameters
    ///
    /// - `llama_model`: A reference to an existing `LlamaModel` that will be used with the new context.
    /// - `llama_context`: A non-null pointer to an existing `llama_cpp_sys_4::llama_context` representing
    ///   the context created in previous steps. This context is necessary for interacting with the model.
    /// - `embeddings_enabled`: A boolean flag indicating whether embeddings are enabled in this context.
    ///
    /// # Returns
    ///
    /// This function returns a new instance of `LlamaContext` initialized with the given parameters:
    /// - The model reference (`llama_model`) is stored in the context.
    /// - The raw context pointer (`llama_context`) is wrapped in a `NonNull` to ensure safety.
    /// - The `embeddings_enabled` flag is used to determine if embeddings are enabled for the context.
    ///
    /// # Example
    /// ```
    /// let llama_model = LlamaModel::load("path/to/model").unwrap();
    /// let context_ptr = NonNull::new(some_llama_context_ptr).unwrap();
    /// let context = LlamaContext::new(&llama_model, context_ptr, true);
    /// // Now you can use the context
    /// ```
    pub(crate) fn new(
        llama_model: &'model LlamaModel,
        llama_context: NonNull<llama_cpp_sys_4::llama_context>,
        embeddings_enabled: bool,
    ) -> Self {
        Self {
            context: llama_context,
            model: llama_model,
            initialized_logits: Vec::new(),
            embeddings_enabled,
        }
    }

    /// Gets the max number of logical tokens that can be submitted to decode. Must be greater than or equal to n_ubatch.
    #[must_use]
    pub fn n_batch(&self) -> u32 {
        unsafe { llama_cpp_sys_4::llama_n_batch(self.context.as_ptr()) }
    }

    /// Gets the max number of physical tokens (hardware level) to decode in batch. Must be less than or equal to n_batch.
    #[must_use]
    pub fn n_ubatch(&self) -> u32 {
        unsafe { llama_cpp_sys_4::llama_n_ubatch(self.context.as_ptr()) }
    }

    /// Gets the size of the context.
    #[must_use]
    pub fn n_ctx(&self) -> u32 {
        unsafe { llama_cpp_sys_4::llama_n_ctx(self.context.as_ptr()) }
    }

    /// Decodes the batch.
    ///
    /// # Errors
    ///
    /// - `DecodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn decode(&mut self, batch: &mut LlamaBatch) -> Result<(), DecodeError> {
        let result =
            unsafe { llama_cpp_sys_4::llama_decode(self.context.as_ptr(), batch.llama_batch) };

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(DecodeError::from(error)),
        }
    }

    /// Encodes the batch.
    ///
    /// # Errors
    ///
    /// - `EncodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn encode(&mut self, batch: &mut LlamaBatch) -> Result<(), EncodeError> {
        let result =
            unsafe { llama_cpp_sys_4::llama_encode(self.context.as_ptr(), batch.llama_batch) };

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(EncodeError::from(error)),
        }
    }

    /// Return Pooling type for Llama's Context
    pub fn pooling_type(&self) -> LlamaPoolingType {
        let pooling_type = unsafe { llama_pooling_type(self.context.as_ptr()) };

        LlamaPoolingType::from(pooling_type)
    }

    /// Get the embeddings for the `i`th sequence in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch.
    /// The size corresponds to the `n_embd` parameter of the context's model.
    ///
    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - If the current model had a pooling type of [`llama_cpp_sys_4::LLAMA_POOLING_TYPE_NONE`]
    /// - If the given sequence index exceeds the max sequence id.
    ///
    /// # Panics
    ///
    /// * `n_embd` does not fit into a usize
    pub fn embeddings_seq_ith(&self, i: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");

        unsafe {
            let embedding = llama_cpp_sys_4::llama_get_embeddings_seq(self.context.as_ptr(), i);

            // Technically also possible whenever `i >= max(batch.n_seq)`, but can't check that here.
            if embedding.is_null() {
                Err(EmbeddingsError::NonePoolType)
            } else {
                Ok(slice::from_raw_parts(embedding, n_embd))
            }
        }
    }

    /// Get the embeddings for the `i`th token in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch of the given token.
    /// The size corresponds to the `n_embd` parameter of the context's model.
    ///
    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - When the given token didn't have logits enabled when it was passed.
    /// - If the given token index exceeds the max token id.
    ///
    /// # Panics
    ///
    /// * `n_embd` does not fit into a usize
    pub fn embeddings_ith(&self, i: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");

        unsafe {
            let embedding = llama_cpp_sys_4::llama_get_embeddings_ith(self.context.as_ptr(), i);
            // Technically also possible whenever `i >= batch.n_tokens`, but no good way of checking `n_tokens` here.
            if embedding.is_null() {
                Err(EmbeddingsError::LogitsNotEnabled)
            } else {
                Ok(slice::from_raw_parts(embedding, n_embd))
            }
        }
    }

    /// Get the logits for the last token in the context.
    ///
    /// # Returns
    /// An iterator over unsorted `LlamaTokenData` containing the
    /// logits for the last token in the context.
    ///
    /// # Panics
    ///
    /// - underlying logits data is null
    pub fn candidates(&self) -> impl Iterator<Item = LlamaTokenData> + '_ {
        (0_i32..).zip(self.get_logits()).map(|(i, logit)| {
            let token = LlamaToken::new(i);
            LlamaTokenData::new(token, *logit, 0_f32)
        })
    }

    /// Get the token data array for the last token in the context.
    ///
    /// This is a convience method that implements:
    /// ```ignore
    /// LlamaTokenDataArray::from_iter(ctx.candidates(), false)
    /// ```
    ///
    /// # Panics
    ///
    /// - underlying logits data is null
    #[must_use]
    pub fn token_data_array(&self) -> LlamaTokenDataArray {
        LlamaTokenDataArray::from_iter(self.candidates(), false)
    }

    /// Token logits obtained from the last call to `decode()`.
    /// The logits for which `batch.logits[i] != 0` are stored contiguously
    /// in the order they have appeared in the batch.
    /// Rows: number of tokens for which `batch.logits[i] != 0`
    /// Cols: `n_vocab`
    ///
    /// # Returns
    ///
    /// A slice containing the logits for the last decoded token.
    /// The size corresponds to the `n_vocab` parameter of the context's model.
    ///
    /// # Panics
    ///
    /// - `n_vocab` does not fit into a usize
    /// - token data returned is null
    pub fn get_logits(&self) -> &[f32] {
        let data = unsafe { llama_cpp_sys_4::llama_get_logits(self.context.as_ptr()) };
        assert!(!data.is_null(), "logits data for last token is null");
        let len = usize::try_from(self.model.n_vocab()).expect("n_vocab does not fit into a usize");

        unsafe { slice::from_raw_parts(data, len) }
    }

    /// Get the logits for the ith token in the context.
    ///
    /// # Panics
    ///
    /// - logit `i` is not initialized.
    pub fn candidates_ith(&self, i: i32) -> impl Iterator<Item = LlamaTokenData> + '_ {
        (0_i32..).zip(self.get_logits_ith(i)).map(|(i, logit)| {
            let token = LlamaToken::new(i);
            LlamaTokenData::new(token, *logit, 0_f32)
        })
    }

    /// Get the logits for the ith token in the context.
    ///
    /// # Panics
    ///
    /// - `i` is greater than `n_ctx`
    /// - `n_vocab` does not fit into a usize
    /// - logit `i` is not initialized.
    #[must_use]
    pub fn get_logits_ith(&self, i: i32) -> &[f32] {
        assert!(
            self.initialized_logits.contains(&i),
            "logit {i} is not initialized. only {:?} is",
            self.initialized_logits
        );
        assert!(
            self.n_ctx() > u32::try_from(i).expect("i does not fit into a u32"),
            "n_ctx ({}) must be greater than i ({})",
            self.n_ctx(),
            i
        );

        let data = unsafe { llama_cpp_sys_4::llama_get_logits_ith(self.context.as_ptr(), i) };
        let len = usize::try_from(self.model.n_vocab()).expect("n_vocab does not fit into a usize");

        unsafe { slice::from_raw_parts(data, len) }
    }

    /// Reset the timings for the context.
    pub fn reset_timings(&mut self) {
        unsafe { llama_cpp_sys_4::ggml_time_init() }
    }

    /// Returns the timings for the context.
    pub fn timings(&mut self) -> PerfContextData {
        let perf_context_data =
            unsafe { llama_cpp_sys_4::llama_perf_context(self.context.as_ptr()) };
        PerfContextData {
            perf_context_data: perf_context_data,
        }
    }

    /// Sets a lora adapter.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterSetError`] for more information.
    pub fn lora_adapter_set(
        &self,
        adapter: &mut LlamaLoraAdapter,
        scale: f32,
    ) -> Result<(), LlamaLoraAdapterSetError> {
        let err_code = unsafe {
            // after renaming happened
            // https://github.com/ggerganov/llama.cpp/commit/afa8a9ec9b520137bbd1ca6838cda93ee39baf20#diff-201cbc8fd17750764ed4a0862232e81503550c201995e16dc2e2766754eaa57aR1016
            llama_cpp_sys_4::llama_set_adapter_lora(
                self.context.as_ptr(),
                adapter.lora_adapter.as_ptr(),
                scale,
            )
        };
        if err_code != 0 {
            return Err(LlamaLoraAdapterSetError::ErrorResult(err_code));
        }

        tracing::debug!("Set lora adapter");
        Ok(())
    }

    /// Remove a lora adapter.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterRemoveError`] for more information.
    pub fn lora_adapter_remove(
        &self,
        adapter: &mut LlamaLoraAdapter,
    ) -> Result<(), LlamaLoraAdapterRemoveError> {
        let err_code = unsafe {
            llama_cpp_sys_4::llama_rm_adapter_lora(
                self.context.as_ptr(),
                adapter.lora_adapter.as_ptr(),
            )
        };
        if err_code != 0 {
            return Err(LlamaLoraAdapterRemoveError::ErrorResult(err_code));
        }

        tracing::debug!("Remove lora adapter");
        Ok(())
    }
}

impl Drop for LlamaContext<'_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_4::llama_free(self.context.as_ptr()) }
    }
}
