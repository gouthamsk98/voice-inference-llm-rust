use tokenizers::Tokenizer;
use tch::{Device, Tensor, Kind}; // For PyTorch model handling

fn setup_tokenizer(model_path: &str) -> Tokenizer {
    let tokenizer = Tokenizer::from_pretrained("homebrewltd/llama3.1-s-instruct-v0.2", None)
        .expect("Tokenizer file could not be loaded");
    tokenizer
}

// Example function that uses tokenizer and runs model
fn main() {
    let model_path = "homebrewltd/llama3.1-s-instruct-v0.2";
    let tokenizer = setup_tokenizer(model_path);

    let encoding = tokenizer.encode("Hello, how are you?", true)
        .expect("Failed to encode input");
    
    // Display tokens or use encoding with model
    println!("Tokens: {:?}", encoding.get_tokens());
}
