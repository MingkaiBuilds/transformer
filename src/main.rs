use candle_core::{Device, Result, DType};
use candle_nn::{VarBuilder, AdamW, Optimizer};
pub mod model;

use model::{
    Transformer, TrainingConfig, generate_dummy_data, cross_entropy_loss,
    create_look_ahead_mask, create_padding_mask
};

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    let device = Device::new_metal(0)?;
    if !device.is_metal() {
        println!("Metal is not available, falling back to CPU.");
        Ok(Device::Cpu)
    } else {
        Ok(device)
    }
}

fn train_transformer(
    _transformer: &Transformer,
    config: &TrainingConfig,
    device: &Device,
) -> Result<()> {
    println!("Starting transformer training...");
    
    let varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    
    let transformer = Transformer::new(
        6, 512, 8, 2048,
        config.vocab_size,
        config.vocab_size,
        config.max_seq_len,
        vb,
    )?;
    
    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: config.learning_rate,
            ..Default::default()
        }
    )?;
    
    for epoch in 0..config.num_epochs {
        let mut total_loss = 0.0f32;
        let num_batches = 10;
        
        for batch in 0..num_batches {
            let (encoder_input, decoder_input, targets) = generate_dummy_data(
                config.batch_size,
                config.max_seq_len,
                config.vocab_size,
                device,
            )?;
            
            let look_ahead_mask = create_look_ahead_mask(config.max_seq_len, device)?;
            let padding_mask = create_padding_mask(&encoder_input, 0)?;
            
            let logits = transformer.forward(
                &encoder_input,
                &decoder_input,
                Some(&look_ahead_mask),
                Some(&padding_mask),
            )?;
            
            let loss = cross_entropy_loss(&logits, &targets)?;
            
            let grads = loss.backward()?;
            optimizer.step(&grads)?;
            
            let loss_val = loss.to_scalar::<f32>()?;
            total_loss += loss_val;
            
            if batch % 5 == 0 {
                println!("Epoch {}, Batch {}, Loss: {:.4}", epoch + 1, batch + 1, loss_val);
            }
        }
        
        let avg_loss = total_loss / num_batches as f32;
        println!("Epoch {} completed. Average Loss: {:.4}", epoch + 1, avg_loss);
    }
    
    println!("Training completed!");
    Ok(())
}

fn demo_inference(transformer: &Transformer, device: &Device) -> Result<()> {
    println!("\nRunning inference demo...");
    
    let config = TrainingConfig::default();
    
    let (encoder_input, decoder_input, _) = generate_dummy_data(
        1, 20, config.vocab_size, device,
    )?;
    
    let look_ahead_mask = create_look_ahead_mask(20, device)?;
    let padding_mask = create_padding_mask(&encoder_input, 0)?;
    
    let output = transformer.forward(
        &encoder_input,
        &decoder_input,
        Some(&look_ahead_mask),
        Some(&padding_mask),
    )?;
    
    println!("Input shape: {:?}", encoder_input.shape());
    println!("Output shape: {:?}", output.shape());
    println!("Output logits (first 5 values): {:?}", 
             output.narrow(2, 0, 5)?.to_vec3::<f32>()?[0][0]);
    
    Ok(())
}

fn main() -> Result<()> {
    let dev = device(true)?;
    println!("Device setup. Using: {:?}", dev);

    let config = TrainingConfig {
        learning_rate: 1e-4,
        batch_size: 4,
        num_epochs: 3,
        warmup_steps: 1000,
        max_seq_len: 64,
        vocab_size: 1000,
    };
    
    println!("Training configuration: {:?}", config);
    
    let varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    
    let transformer = Transformer::new(
        2, 512, 8, 2048,
        config.vocab_size,
        config.vocab_size,
        config.max_seq_len,
        vb,
    )?;
    
    demo_inference(&transformer, &dev)?;
    train_transformer(&transformer, &config, &dev)?;
    demo_inference(&transformer, &dev)?;

    Ok(())
}
