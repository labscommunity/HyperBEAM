use std::{
    io::{self, BufRead, Write},
    path::Path,
    sync::Arc,
};

use anyhow::Result;
use server::runtime::{ChatbotRequest, WasmInstance};
use tokenizers::tokenizer::Tokenizer;
use wasmtime::{component::Component, Config, Engine};
use tokio::{io::{stdin, BufReader, AsyncBufReadExt}, sync::mpsc};

#[tokio::main]
async fn main() -> Result<()>
{
    tracing_subscriber::fmt::Subscriber::builder()
        .with_max_level(tracing::Level::DEBUG)
        .with_env_filter("error")
        .init();
    println!("🦙 Chatbot is getting prepared. Please wait!");
    let model_id = "llama3.2-1b-instruct-fp32";
    let tokenizer_path = format!("./models/onnx/{}/tokenizer.json", model_id);
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::Error::msg(e.to_string()))?;
    let mut config = Config::new();
    config.async_support(true);
    let engine = Arc::new(Engine::new(&config)?);
    let module = Arc::new(
        Component::from_file(&engine, Path::new("../inferencer/target/wasm32-wasip1/release/ncl_ml.wasm"))?,
    );

    let mut stdin = BufReader::new(stdin()).lines();
    let mut stdout = io::stdout();
    
    let (mut token_receiver, chatbot_sender) = WasmInstance::new(engine.clone(), module.clone(), model_id).await?;
    let mut current_session: Option<u64> = None;
    
    println!("🦙 Chatbot ready. Type a command: 'join', 'exit'");
    loop {
        tokio::select! {
            Ok(Some(input)) = stdin.next_line() => {
                let input = input.trim();
                match (input, current_session) {
                    ("exit", Some(session_id)) => {
                        // stop current inference
                        chatbot_sender.send(ChatbotRequest::EndSession(session_id));
                    }
                    ("exit", None) => {
                        tracing::error!("No active session to exit");
                    }
                    ("join", Some(session_id)) => {
                        tracing::error!("must exit current session {} before joining a new one!", session_id);
                    }
                    ("join", None) => {
                        chatbot_sender.send(ChatbotRequest::StartSession);
                        current_session = Some(0);
                        println!("joined a session, start chatting!");
                    }
                    (other, Some(session_id)) => {
                        let prompt = format!(
                            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nOnly answer one question at a time and make answers as short as possible.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\
                            <|start_header_id|>assistant<|end_header_id|>", 
                            other
                        );
                        let encoding = tokenizer.encode(prompt, false).map_err(|e| anyhow::Error::msg(e.to_string()))?;
                        let ids = encoding.get_ids().iter().map(|&id| id as i64).collect();
                        chatbot_sender.send(ChatbotRequest::Infer(session_id, ids));
                    }
                    (other, None) => {
                        tracing::error!("unexpected command: {}", other);
                    }
                };
            }
            Some((session_id, token)) = token_receiver.recv() => {
                if let Some(session_id) = current_session {
                    // print the token in the session
                    let token = tokenizer.decode(&[token], false).map_err(|e| anyhow::Error::msg(e.to_string()))?;
                    write!(stdout, "{}", token)?;
                    stdout.flush()?;
                }
            }
        }
    }
    Ok(())
}
