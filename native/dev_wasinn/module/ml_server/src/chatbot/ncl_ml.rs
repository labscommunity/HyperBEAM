use std::{collections::HashMap, hash::Hash};

use tokenizers::tokenizer::Tokenizer;
use tokio::sync::mpsc::UnboundedSender;
use wasmtime::component::ResourceTable;

mod generated_
{
    wasmtime::component::bindgen!({
        world: "ml",
        path: "../inferencer/wit/ncl-ml.wit",
        async: true
    });
}

pub mod types
{
    pub use super::generated_::{exports::ncl::ml::chatbot::SessionConfig, ncl::ml::token_generator, Ml as NclML};
}

pub struct NclMlView<'a>
{
    ctx: &'a mut NclMlContenx,
    table: &'a mut ResourceTable,
}

impl<'a> NclMlView<'a>
{
    pub fn new(table: &'a mut ResourceTable, ctx: &'a mut NclMlContenx) -> Self
    {
        Self {
            ctx,
            table,
        }
    }
}

pub struct NclMlContenx
{
    pub sessions: HashMap<u64, UnboundedSender<(u64, String)>>,
    pub tokenizer: Tokenizer
}

impl NclMlContenx
{
    pub fn new(model_id: &str) -> Self {
        let tokenizer_path = format!("../models/onnx/{}/tokenizer.json", model_id);
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::Error::msg(e.to_string())).unwrap();
        Self { 
            sessions: HashMap::new(),
            tokenizer
        }
    }
    pub fn new_session(&mut self, session_id: u64, token_sender: UnboundedSender<(u64, String)>)
    {
        self.sessions.insert(session_id, token_sender);
    }

    pub fn end_session(&mut self, session_id: &u64)
    {
        self.sessions.remove(session_id);
    }
}

impl types::token_generator::Host for NclMlView<'_>
{
    async fn yield_(
        &mut self,
        session_id: types::token_generator::SessionId,
        token: types::token_generator::TokenId,
    ) -> u32
    {
        match self.ctx.sessions.get(&session_id) {
            None => 0,
            Some(token_sender) => {
                let token = self.ctx.tokenizer.decode(&[token], false).map_err(|e| anyhow::Error::msg(e.to_string())).unwrap();
                match token_sender.send((session_id, token)) {
                    Ok(()) => 1,
                    Err(e) => {
                        tracing::error!("failed to yield token due to SendError: {}", e);
                        0
                    },
                }
            }
        }
    }
}

pub fn add_to_linker<T: Send>(
    l: &mut wasmtime::component::Linker<T>,
    f: impl Fn(&mut T) -> NclMlView<'_> + Send + Sync + Copy + 'static,
) -> anyhow::Result<()>
{
    types::token_generator::add_to_linker_get_host(l, f);
    Ok(())
}
