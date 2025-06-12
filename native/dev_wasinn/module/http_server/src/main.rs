
use std::{path::Path, sync::Arc};

use axum::{
	Router,
	extract::{FromRef, State},
	response::{
		Sse,
		sse::{Event, KeepAlive}
	},
	routing::post
};
use futures::Stream;
use ort::{
	session::{RunOptions, Session, builder::GraphOptimizationLevel},
	value::TensorRef,
	execution_providers::{CUDAExecutionProvider, CPUExecutionProvider, ExecutionProvider, ArenaExtendStrategy, cuda::CuDNNConvAlgorithmSearch}
};
use tokenizers::Tokenizer;
use tokio::{net::TcpListener, sync::Mutex, runtime::{Runtime, Builder}};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};



fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
		.with(tracing_subscriber::EnvFilter::try_from_default_env()
		.unwrap_or_else(|_| "info,ort=debug".into()))
		.with(tracing_subscriber::fmt::layer())
		.init();

    let runtime = create_runtime().expect("Tokio runtime not created!");
    let _ = runtime.block_on(start_server(3002, "../inferencer/target/wasm32-wasip1/release/ncl_ml.wasm", "llama3.2-1b-instruct-fp32", ExecutionTarget::GPU));
    Ok(())
}


fn create_runtime() -> anyhow::Result<Runtime>
{
    Builder::new_multi_thread()
        .enable_all()
        .worker_threads(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))
        .build()
        .map_err(Into::into)
}

enum ExecutionTarget {
	GPU,
	CPU
}
async fn start_server(port: u16, wasm_module_path: &str, model_id: &str, target: ExecutionTarget) -> anyhow::Result<()>
{
	let models_path = Path::new(env!("CARGO_MANIFEST_DIR"))
			.parent()
			.unwrap().join("models").join("onnx");
    let builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?;
    let builder = match target {
        ExecutionTarget::GPU => {
            let cuda = CUDAExecutionProvider::default()
                .with_device_id(0)
                .with_arena_extend_strategy(ArenaExtendStrategy::NextPowerOfTwo)
                .with_memory_limit(15 * 1024 * 1024 * 1024)
                .with_conv_algorithm_search(CuDNNConvAlgorithmSearch::Exhaustive)
                .build();
            let cpu = CPUExecutionProvider::default().build();
            builder.with_execution_providers([cuda])?
        }
        _ => builder
    };
	println!("{:?}", models_path.join(model_id).join("model.onnx"));
	println!("{:?}", models_path.join(model_id).join("tokenizer.json"));
    let session = builder.commit_from_file(&models_path.join(model_id).join("model.onnx"))?;
    
    // Load the tokenizer and encode the prompt into a sequence of tokens.
	let tokenizer = Tokenizer::from_file(
		&models_path.join(model_id).join("tokenizer.json")
	).unwrap();

    let app_state = AppState {
		session: Arc::new(Mutex::new(session)),
		tokenizer: Arc::new(tokenizer)
	};

    let app = Router::new().route("/chat/completions", post(generate)).with_state(app_state).into_make_service();
	let addr: std::net::SocketAddr = ([0, 0, 0, 0], port).into();
	let listener = TcpListener::bind(addr).await?;
	tracing::info!("Listening on {}", listener.local_addr()?);

	axum::serve(listener, app).await?;

	Ok(())
}

#[derive(Clone)]
struct AppState {
	session: Arc<Mutex<Session>>,
	tokenizer: Arc<Tokenizer>
}

fn generate_stream(
	tokenizer: Arc<Tokenizer>,
	session: Arc<Mutex<Session>>,
	prompt: String,
	max_tokens: Option<usize>
) -> impl Stream<Item = ort::Result<Event>> + Send {
	async_stream_lite::try_async_stream(|yielder| async move {
		let prompt = format!(
			"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nOnly answer one question at a time and make answers as short as possible.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\
			<|start_header_id|>assistant<|end_header_id|>",
			prompt
		);
		let encoding = tokenizer.encode(prompt, false).unwrap();
		// input_ids
		let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();		
		let mut length = input_ids.len();
		let mut tokens_dims: Vec<i64> = vec![1i64, length as i64];
		
		// position_ids
		let mut position_ids: Vec<i64> = Vec::with_capacity(length);
		for i in 0..length {
			position_ids.push(i as i64);
		}
		// attention_mask
		let mut attention_mask: Vec<i64> = Vec::with_capacity(length);
		for &id in &input_ids {
			attention_mask.push(get_attention_mask(&id));
		}

		let mut count = 0;

		loop {
			match max_tokens {
				Some(i) if count >= i => break,
				_ => (),
			};

			let inputs = ort::inputs!{
				"input_ids" => ort::value::Tensor::from_array((tokens_dims.clone(), input_ids.clone().into_boxed_slice()))?,
				"position_ids" => ort::value::Tensor::from_array((tokens_dims.clone(), position_ids.clone().into_boxed_slice()))?,
				"attention_mask" => ort::value::Tensor::from_array((tokens_dims.clone(), attention_mask.clone().into_boxed_slice()))?
			};
			let options = RunOptions::new()?;
			let mut next_logit: Vec<f32> = {
				let mut session = session.lock().await;
				let outputs = session.run_async(inputs, &options)?.await?;
				let (dim, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
				let [_batch_size, seq_len, vocab_size] = match dim[..3] {
					[a, b, c] => [a, b, c],
					_ => panic!("tensor dimensions must match [batch_len, seq_len, vocab_size]"),
				};
				// copy the next logit
				let next_logit_start: usize = ((seq_len - 1) * vocab_size) as usize;
				let next_logit_end: usize = next_logit_start + (vocab_size as usize);
				logits[next_logit_start..next_logit_end].to_vec()
			};
			softmax(&mut next_logit);
			let (next_token, _) = next_logit.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
			match next_token {
				128000..=128255 => break,
				_ => {
					let token = tokenizer.decode(&[next_token as _], true).unwrap();
					println!("next token: {}", token);
					yielder.r#yield(Event::default().data(token)).await;
					// extend input slices to include next_token
					input_ids.push(next_token as i64);
					position_ids.push(length as i64);
					attention_mask.push(get_attention_mask(&(next_token as i64)));
					length += 1;
					tokens_dims[1] = length as i64;
					count += 1;
				}
			}
		}
		Ok(())
	})
}

use axum::Json;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ChatRequest {
	model: Option<String>,
	message: String,
	stream: Option<bool>,
}

impl FromRef<AppState> for Arc<Mutex<Session>> {
	fn from_ref(input: &AppState) -> Self {
		Arc::clone(&input.session)
	}
}

impl FromRef<AppState> for Arc<Tokenizer> {
	fn from_ref(input: &AppState) -> Self {
		Arc::clone(&input.tokenizer)
	}
}

async fn generate(State(session): State<Arc<Mutex<Session>>>, State(tokenizer): State<Arc<Tokenizer>>, Json(payload): Json<ChatRequest>) -> Sse<impl Stream<Item = ort::Result<Event>>> {
	let prompt = payload.message;
	Sse::new(generate_stream(tokenizer, session, prompt, Some(50))).keep_alive(KeepAlive::new())
}

pub fn get_attention_mask(token_id: &i64) -> i64
{
    match token_id {
        128000..=128255 => 0,
        _ => 1,
    }
}

pub fn softmax(x: &mut [f32])
{
    let mut sum: f32 = 0.0;
    let mut max_val: f32 = x[0];

    for i in x.iter() {
        if *i > max_val {
            max_val = *i;
        }
    }

    for i in x.iter_mut() {
        *i = (*i - max_val).exp();
        sum += *i;
    }

    for i in x.iter_mut() {
        *i /= sum;
    }
}