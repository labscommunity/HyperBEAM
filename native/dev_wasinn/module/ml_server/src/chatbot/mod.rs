mod ncl_ml;

use std::{path::PathBuf, sync::Arc};

use ncl_ml::types::{NclML, SessionConfig};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use wasmtime::{
    component::{Component, Linker, ResourceTable},
    Engine, Store,
};
use wasmtime_wasi::{
    p2::{WasiCtx, WasiCtxBuilder},
    DirPerms, FilePerms,
};
use wasmtime_wasi_nn::{
    backend::onnx::OnnxBackend,
    wit::{WasiNnCtx, WasiNnView},
    Backend,
};

use super::registry::Registry;

pub enum ChatbotRequest
{
    StartSession,
    EndSession(u64),
    Infer(u64, Vec<i64>),
}

pub struct Chatbot
{
    ncl_ml_world: NclML,
    store: Store<ChatbotContext>,
    model_id: String,
}

impl Chatbot
{
    pub async fn new(
        engine: Arc<Engine>,
        component: Arc<Component>,
        model_id: &str,
    ) -> anyhow::Result<(UnboundedReceiver<(u64, u32)>, UnboundedSender<ChatbotRequest>)>
    {
        let (token_sender, token_receiver) = unbounded_channel::<(u64, u32)>();
        let context = ChatbotContext::new(Backend::from(OnnxBackend::default()), model_id, token_sender)?;
        let mut store = Store::new(&engine, context);

        let mut linker = Linker::new(&engine);
        wasmtime_wasi_nn::wit::add_to_linker(&mut linker, |c: &mut ChatbotContext| {
            WasiNnView::new(&mut c.table, &mut c.wasi_nn)
        })?;
        ncl_ml::add_to_linker(&mut linker, |c: &mut ChatbotContext| {
            ncl_ml::NclMlView::new(&mut c.table, &mut c.ncl_ml)
        })?;
        wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;
        let ncl_ml_world = NclML::instantiate_async(&mut store, &component, &linker).await?;
        let (prompt_sender, mut prompt_receiver) = unbounded_channel::<ChatbotRequest>();
        let model_id = model_id.to_owned();
        tokio::spawn(async move {
            let mut chatbot = Chatbot {
                ncl_ml_world,
                store,
                model_id,
            };
            async fn process_request(instance: &mut Chatbot, request: ChatbotRequest) -> anyhow::Result<()>
            {
                match request {
                    ChatbotRequest::EndSession(session_id) => {
                        let ctx = instance.store.data_mut();
                        ctx.ncl_ml.end_session(&session_id);
                        Ok(())
                    },
                    ChatbotRequest::StartSession => {
                        let guest = instance.ncl_ml_world.ncl_ml_chatbot();
                        let result = guest
                            .call_register(
                                &mut instance.store,
                                &SessionConfig {
                                    model_id: instance.model_id.clone(),
                                    max_token: Some(100),
                                    history: None,
                                },
                            )
                            .await?;
                        let ctx = instance.store.data_mut();
                        ctx.ncl_ml.new_session(result, ctx.token_sender.clone());
                        Ok(())
                    },
                    ChatbotRequest::Infer(session_id, ids) => {
                        let guest = instance.ncl_ml_world.ncl_ml_chatbot();
                        let result = guest.call_infer(&mut instance.store, session_id, &ids).await?;
                        Ok(())
                    },
                }
            }
            while let Some(request) = prompt_receiver.recv().await {
                if let Err(e) = process_request(&mut chatbot, request).await {
                    tracing::error!("Process request failed: {:?}", e);
                }
            }
        });
        Ok((token_receiver, prompt_sender))
    }
}

struct ChatbotContext
{
    wasi: WasiCtx,
    wasi_nn: WasiNnCtx,
    ncl_ml: ncl_ml::NclMlContenx,
    table: ResourceTable,
    token_sender: UnboundedSender<(u64, u32)>,
}

impl ChatbotContext
{
    fn new(mut backend: Backend, model_id: &str, token_sender: UnboundedSender<(u64, u32)>) -> anyhow::Result<Self>
    {
        let host_path: PathBuf = std::env::current_dir().unwrap().join("models").join("onnx").join(model_id);
        let mut builder = WasiCtxBuilder::new();
        builder.inherit_stdio().preopened_dir(&host_path, "", DirPerms::READ, FilePerms::READ)?;
        let wasi = builder.build();

        let mut registry = Registry::new();
        registry.load((backend).as_dir_loadable().unwrap(), &host_path, model_id)?;
        let wasi_nn = WasiNnCtx::new([backend], registry.into());
        Ok(Self {
            wasi,
            wasi_nn,
            table: ResourceTable::new(),
            ncl_ml: ncl_ml::NclMlContenx::default(),
            token_sender,
        })
    }
}
impl wasmtime_wasi::p2::IoView for ChatbotContext
{
    fn table(&mut self) -> &mut ResourceTable
    {
        &mut self.table
    }
}
impl wasmtime_wasi::p2::WasiView for ChatbotContext
{
    fn ctx(&mut self) -> &mut WasiCtx
    {
        &mut self.wasi
    }
}
