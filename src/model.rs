use candle_core::{Result, Tensor, Device, DType, D};
use candle_nn::{Module, VarBuilder, layer_norm, LayerNorm, LayerNormConfig, embedding, Embedding, ops::softmax};

pub struct InputEmbedding {
    embedding: Embedding,
}

impl InputEmbedding {
    pub fn new(vocab_size: usize, d_model: usize, vb: VarBuilder) -> Result<Self> {
        let embedding = embedding(vocab_size, d_model, vb)?;
        Ok(Self { embedding })
    }
}

impl Module for InputEmbedding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.embedding.forward(x)
    }
}

fn positional_encoding(seq_len: usize, d_model: usize, dev: &Device) -> Result<Tensor> {
    let mut pe_data = vec![0.0f32; seq_len * d_model];
    
    for pos in 0..seq_len {
        for i in 0..d_model / 2 {
            let angle = pos as f32 / 10000.0f32.powf(2.0 * i as f32 / d_model as f32);
            pe_data[pos * d_model + 2 * i] = angle.sin();
            pe_data[pos * d_model + 2 * i + 1] = angle.cos();
        }
    }
    
    Tensor::from_slice(&pe_data, (seq_len, d_model), dev)
}

pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let d_k = query.dim(D::Minus1)? as f32;
    let scale = (d_k.sqrt()).recip();

    let key_t = key.transpose(D::Minus2, D::Minus1)?.contiguous()?;
    let scores = query.contiguous()?.matmul(&key_t)?;
    let scaled_scores = scores.affine(scale as f64, 0.0)?;

    let masked_scores = if let Some(mask) = mask {
        let mask_value = Tensor::new(-1e9f32, query.device())?;
        let scores_shape = scaled_scores.shape();
        let broadcast_mask = if mask.dims() != scores_shape.dims() {
            mask.broadcast_as(scores_shape)?
        } else {
            mask.clone()
        };
        scaled_scores.where_cond(&broadcast_mask, &mask_value)?
    } else {
        scaled_scores
    };

    let attention_weights = softmax(&masked_scores, D::Minus1)?;
    attention_weights.matmul(&value.contiguous()?)
}

#[derive(Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((in_dim, out_dim), "weight")?;
        let bias = vb.get(out_dim, "bias")?;
        Ok(Self { weight, bias })
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original_shape = x.dims().to_vec();
        
        let (x_2d, need_reshape) = if original_shape.len() == 3 {
            let batch_size = original_shape[0];
            let seq_len = original_shape[1];
            let features = original_shape[2];
            (x.reshape((batch_size * seq_len, features))?, true)
        } else {
            (x.clone(), false)
        };
        
        let output = x_2d.matmul(&self.weight)?;
        let output = output.broadcast_add(&self.bias)?;
        
        if need_reshape {
            let output_features = output.dim(1)?;
            output.reshape((original_shape[0], original_shape[1], output_features))
        } else {
            Ok(output)
        }
    }
}

pub struct MultiHeadAttention {
    h: usize,
    d_model: usize,
    d_k: usize,
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
}

impl MultiHeadAttention {
    pub fn new(h: usize, d_model: usize, vb: VarBuilder) -> Result<Self> {
        assert!(d_model % h == 0, "d_model must be divisible by h");
        let d_k = d_model / h;
        Ok(MultiHeadAttention {
            h,
            d_model,
            d_k,
            w_q: Linear::new(d_model, d_model, vb.pp("w_q"))?,
            w_k: Linear::new(d_model, d_model, vb.pp("w_k"))?,
            w_v: Linear::new(d_model, d_model, vb.pp("w_v"))?,
            w_o: Linear::new(d_model, d_model, vb.pp("w_o"))?,
        })
    }

    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let batch_size = query.dim(0)?;
        let seq_len_q = query.dim(1)?;
        let seq_len_k = key.dim(1)?;

        let q = self.w_q.forward(query)?;
        let k = self.w_k.forward(key)?;
        let v = self.w_v.forward(value)?;

        let q_heads = q.reshape((batch_size, seq_len_q, self.h, self.d_k))?
            .transpose(1, 2)?.contiguous()?;
        let k_heads = k.reshape((batch_size, seq_len_k, self.h, self.d_k))?
            .transpose(1, 2)?.contiguous()?;
        let v_heads = v.reshape((batch_size, seq_len_k, self.h, self.d_k))?
            .transpose(1, 2)?.contiguous()?;

        let attention_output = scaled_dot_product_attention(&q_heads, &k_heads, &v_heads, mask)?;

        let attention_output = attention_output.transpose(1, 2)?.contiguous()?;
        let concatenated = attention_output.reshape((batch_size, seq_len_q, self.d_model))?;

        self.w_o.forward(&concatenated)
    }
}

fn relu(x: &Tensor) -> Result<Tensor> {
    let zero = Tensor::zeros_like(x)?;
    x.maximum(&zero)
}

pub struct PositionwiseFeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl PositionwiseFeedForward {
    pub fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        Ok(PositionwiseFeedForward {
            linear1: Linear::new(d_model, d_ff, vb.pp("linear1"))?,
            linear2: Linear::new(d_ff, d_model, vb.pp("linear2"))?,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let after_linear1 = self.linear1.forward(input)?;
        let after_relu = relu(&after_linear1)?;
        self.linear2.forward(&after_relu)
    }
}

// We will now define our own LayerNorm struct which will use candle's
// functional `layer_norm` implementation.
#[derive(Debug)]
pub struct LayerNormalization {
    ln: LayerNorm,
}

impl LayerNormalization {
    pub fn new(d_model: usize, vb: VarBuilder) -> Result<Self> {
        let ln = layer_norm(d_model, LayerNormConfig::default(), vb)?;
        Ok(Self { ln })
    }
}

impl Module for LayerNormalization {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.ln.forward(x)
    }
}

pub struct EncoderBlock {
    attention: MultiHeadAttention,
    feed_forward: PositionwiseFeedForward,
    norm1: LayerNormalization,
    norm2: LayerNormalization,
}

impl EncoderBlock {
    pub fn new(d_model: usize, h: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        Ok(EncoderBlock {
            attention: MultiHeadAttention::new(h, d_model, vb.pp("attention"))?,
            feed_forward: PositionwiseFeedForward::new(d_model, d_ff, vb.pp("feed_forward"))?,
            norm1: LayerNormalization::new(d_model, vb.pp("norm1"))?,
            norm2: LayerNormalization::new(d_model, vb.pp("norm2"))?,
        })
    }

    pub fn forward(&self, input: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let attention_output = self.attention.forward(input, input, input, mask)?;
        let residual1 = input.broadcast_add(&attention_output)?;
        let norm1_output = self.norm1.forward(&residual1)?;

        let ff_output = self.feed_forward.forward(&norm1_output)?;
        let residual2 = norm1_output.broadcast_add(&ff_output)?;
        self.norm2.forward(&residual2)
    }
}

pub struct DecoderBlock {
    masked_attention: MultiHeadAttention,
    cross_attention: MultiHeadAttention,
    feed_forward: PositionwiseFeedForward,
    norm1: LayerNormalization,
    norm2: LayerNormalization,
    norm3: LayerNormalization,
}

impl DecoderBlock {
    pub fn new(d_model: usize, h: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        Ok(DecoderBlock {
            masked_attention: MultiHeadAttention::new(h, d_model, vb.pp("masked_attention"))?,
            cross_attention: MultiHeadAttention::new(h, d_model, vb.pp("cross_attention"))?,
            feed_forward: PositionwiseFeedForward::new(d_model, d_ff, vb.pp("feed_forward"))?,
            norm1: LayerNormalization::new(d_model, vb.pp("norm1"))?,
            norm2: LayerNormalization::new(d_model, vb.pp("norm2"))?,
            norm3: LayerNormalization::new(d_model, vb.pp("norm3"))?,
        })
    }

    pub fn forward(
        &self,
        input: &Tensor,
        encoder_output: &Tensor,
        look_ahead_mask: Option<&Tensor>,
        padding_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let masked_attn_output = self.masked_attention.forward(input, input, input, look_ahead_mask)?;
        let residual1 = input.broadcast_add(&masked_attn_output)?;
        let norm1_output = self.norm1.forward(&residual1)?;

        let cross_attn_output = self.cross_attention.forward(&norm1_output, encoder_output, encoder_output, padding_mask)?;
        let residual2 = norm1_output.broadcast_add(&cross_attn_output)?;
        let norm2_output = self.norm2.forward(&residual2)?;

        let ff_output = self.feed_forward.forward(&norm2_output)?;
        let residual3 = norm2_output.broadcast_add(&ff_output)?;
        self.norm3.forward(&residual3)
    }
}

pub struct Transformer {
    encoder_embedding: InputEmbedding,
    decoder_embedding: InputEmbedding,
    encoder_blocks: Vec<EncoderBlock>,
    decoder_blocks: Vec<DecoderBlock>,
    output_linear: Linear,
    max_seq_len: usize,
    d_model: usize,
}

impl Transformer {
    pub fn new(
        num_blocks: usize,
        d_model: usize,
        h: usize,
        d_ff: usize,
        input_vocab_size: usize,
        output_vocab_size: usize,
        max_seq_len: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut encoder_blocks = Vec::with_capacity(num_blocks);
        let mut decoder_blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            encoder_blocks.push(EncoderBlock::new(d_model, h, d_ff, vb.pp(&format!("encoder.{}", i)))?);
            decoder_blocks.push(DecoderBlock::new(d_model, h, d_ff, vb.pp(&format!("decoder.{}", i)))?);
        }

        Ok(Transformer {
            encoder_embedding: InputEmbedding::new(input_vocab_size, d_model, vb.pp("encoder_embedding"))?,
            decoder_embedding: InputEmbedding::new(output_vocab_size, d_model, vb.pp("decoder_embedding"))?,
            encoder_blocks,
            decoder_blocks,
            output_linear: Linear::new(d_model, output_vocab_size, vb.pp("output_linear"))?,
            max_seq_len,
            d_model,
        })
    }

    pub fn forward(
        &self,
        encoder_input: &Tensor,
        decoder_input: &Tensor,
        look_ahead_mask: Option<&Tensor>,
        padding_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let device = encoder_input.device();
        let encoder_seq_len = encoder_input.dim(1)?;
        let decoder_seq_len = decoder_input.dim(1)?;

        // Encoder pass
        let mut encoder_output = self.encoder_embedding.forward(encoder_input)?;
        
        // Add positional encoding to encoder
        let encoder_pos_encoding = positional_encoding(encoder_seq_len, self.d_model, device)?;
        encoder_output = encoder_output.broadcast_add(&encoder_pos_encoding)?;
        
        for block in &self.encoder_blocks {
            encoder_output = block.forward(&encoder_output, padding_mask)?;
        }

        // Decoder pass
        let mut decoder_output = self.decoder_embedding.forward(decoder_input)?;
        
        // Add positional encoding to decoder
        let decoder_pos_encoding = positional_encoding(decoder_seq_len, self.d_model, device)?;
        decoder_output = decoder_output.broadcast_add(&decoder_pos_encoding)?;
        
        for block in &self.decoder_blocks {
            decoder_output = block.forward(&decoder_output, &encoder_output, look_ahead_mask, padding_mask)?;
        }

        // Final linear layer
        self.output_linear.forward(&decoder_output)
    }
}

pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let log_sm = candle_nn::ops::log_softmax(logits, D::Minus1)?;
    let targets = targets.unsqueeze(D::Minus1)?;
    let loss = log_sm.gather(&targets, D::Minus1)?.squeeze(D::Minus1)?.neg()?;
    loss.mean_all()
}

pub fn create_look_ahead_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask = Tensor::tril2(size, DType::U8, device)?;
    let ones = Tensor::ones((size, size), DType::U8, device)?;
    ones.sub(&mask)?.to_dtype(DType::U8)
}

pub fn create_padding_mask(seq: &Tensor, pad_token: u32) -> Result<Tensor> {
    let pad_tensor = Tensor::full(pad_token, seq.dims(), seq.device())?;
    seq.eq(&pad_tensor)
}

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub warmup_steps: usize,
    pub max_seq_len: usize,
    pub vocab_size: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 8,
            num_epochs: 10,
            warmup_steps: 4000,
            max_seq_len: 128,
            vocab_size: 10000,
        }
    }
}

pub fn generate_dummy_data(
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut encoder_data = vec![0u32; batch_size * seq_len];
    let mut decoder_data = vec![0u32; batch_size * seq_len];
    
    for i in 0..batch_size * seq_len {
        encoder_data[i] = (i % vocab_size) as u32 + 1;
        decoder_data[i] = ((i + 1) % vocab_size) as u32 + 1;
    }
    
    let encoder_input = Tensor::from_slice(&encoder_data, (batch_size, seq_len), device)?;
    let decoder_input = Tensor::from_slice(&decoder_data, (batch_size, seq_len), device)?;
    
    let mut targets_data = vec![0u32; batch_size * seq_len];
    for i in 0..batch_size {
        for j in 0..seq_len {
            if j < seq_len - 1 {
                targets_data[i * seq_len + j] = ((i + j + 1) % vocab_size) as u32;
            } else {
                targets_data[i * seq_len + j] = 0;
            }
        }
    }
    let targets = Tensor::from_slice(&targets_data, (batch_size, seq_len), device)?;
    
    Ok((encoder_input, decoder_input, targets))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_embedding() -> Result<()> {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        
        let vocab_size = 1000;
        let d_model = 128;
        let seq_len = 10;
        let batch_size = 3;

        let embedding_layer = InputEmbedding::new(vocab_size, d_model, vb)?;

        let input = Tensor::zeros((batch_size, seq_len), DType::U32, &dev)?;
        
        let output = embedding_layer.forward(&input)?;

        assert_eq!(output.shape().dims(), &[batch_size, seq_len, d_model]);
        
        Ok(())
    }

    #[test]
    fn test_linear_layer() -> Result<()> {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        
        let in_dim = 10;
        let out_dim = 5;
        let linear = Linear::new(in_dim, out_dim, vb)?;

        let input = Tensor::zeros((3, in_dim), DType::F32, &dev)?;
        
        let output = linear.forward(&input)?;

        assert_eq!(output.shape().dims(), &[3, out_dim]);
        
        Ok(())
    }

    #[test]
    fn test_layer_norm() -> Result<()> {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let d_model = 10;

        let layer_norm = LayerNormalization::new(d_model, vb)?;
        let input = Tensor::ones((3, d_model), DType::F32, &dev)?;

        let output = layer_norm.forward(&input)?;

        let mean = output.mean_all()?.to_scalar::<f32>()?;
        assert!(mean.abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_transformer_creation() -> Result<()> {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        
        let config = TrainingConfig::default();
        let transformer = Transformer::new(
            2,  // num_blocks
            128, // d_model
            8,   // h
            512, // d_ff
            config.vocab_size,
            config.vocab_size,
            config.max_seq_len,
            vb,
        )?;
        
        let (encoder_input, decoder_input, _targets) = generate_dummy_data(
            2, 10, config.vocab_size, &dev
        )?;
        
        let output = transformer.forward(&encoder_input, &decoder_input, None, None)?;
        
        assert_eq!(output.shape().dims(), &[2, 10, config.vocab_size]);
        
        Ok(())
    }

    #[test]
    fn test_cross_entropy_loss() -> Result<()> {
        let dev = Device::Cpu;
        let batch_size = 2;
        let seq_len = 3;
        let vocab_size = 4;
        
        let logits = Tensor::randn(0f32, 1.0f32, (batch_size, seq_len, vocab_size), &dev)?;
        let mut targets_data = vec![0u32; batch_size * seq_len];
        for i in 0..batch_size * seq_len {
            targets_data[i] = (i % vocab_size) as u32;
        }
        let targets = Tensor::from_slice(&targets_data, (batch_size, seq_len), &dev)?;
        
        let loss = cross_entropy_loss(&logits, &targets)?;
        
        assert_eq!(loss.shape().dims(), &[] as &[usize]);
        
        Ok(())
    }
} 