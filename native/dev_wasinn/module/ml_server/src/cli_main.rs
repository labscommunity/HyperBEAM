use std::io::{self, Write};

use anyhow::Result;
use ml_server::chatbot::{Chatbot, ChatbotRequest};
use tokio::io::{stdin, AsyncBufReadExt, BufReader};

#[tokio::main]
async fn main() -> Result<()>
{
    tracing_subscriber::fmt::Subscriber::builder()
        .with_max_level(tracing::Level::DEBUG)
        .with_env_filter("error")
        .init();
    println!("🦙 Chatbot is getting prepared. Please wait!");
    let model_id = "llama3.2-1b-instruct-fp32";

    let mut stdin = BufReader::new(stdin()).lines();
    let mut stdout = io::stdout();

    let (mut token_receiver, chatbot_sender) = Chatbot::new(model_id).await?;
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
                    (prompt, Some(session_id)) => {
                        chatbot_sender.send(ChatbotRequest::Infer(session_id, prompt.to_string()));
                    }
                    (other, None) => {
                        tracing::error!("unexpected command: {}", other);
                    }
                };
            }
            Some((_token_session_id, token)) = token_receiver.recv() => {
                if let Some(session_id) = current_session {
                    // print the token in the session                    
                    write!(stdout, "{}", token)?;
                    stdout.flush()?;
                }
            }
        }
    }
}
