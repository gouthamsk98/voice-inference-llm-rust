use candle_core::{ Device, Result, Tensor, DType };
use candle_nn::{ Linear, Module };
use hf_hub::api::sync::Api;

fn main() -> Result<()> {
    // Use Device::new_cuda(0)?; to use the GPU.
    let device = Device::new_cuda(0)?;

    //creating a dummy model

    let api = Api::new().unwrap();
    let repo = api.model("bert-base-uncased".to_string());

    let weights_filename = repo.get("model.safetensors").unwrap();

    let weights = candle_core::safetensors::load(weights_filename, &device)?;
    let weight = weights.get("bert.encoder.layer.0.attention.self.query.weight").unwrap();
    let bias = weights.get("bert.encoder.layer.0.attention.self.query.bias").unwrap();
    let linear = Linear::new(weight.clone(), Some(bias.clone()));
    let input_ids = Tensor::zeros((3, 768), DType::F32, &device).unwrap();
    let output = linear.forward(&input_ids).unwrap();
    println!("Output {output:?} output");
    Ok(())
}
