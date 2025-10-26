import gradio as gr
from llm import generate_answer

def chat_fn(query, use_retrieval):
    answer = generate_answer(query, use_retrieval=use_retrieval)
    return answer

iface = gr.Interface(
    fn=chat_fn,
    inputs=[
        gr.Textbox(label="Your Question"),
        gr.Checkbox(label="Enable retrieval (RAG)", value=True)
    ],
    outputs="text",
    title="Distributed RAG Chatbot",
    description="Toggle retrieval (RAG) on or off to see difference in LLM hallucination mitigation."
)

iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
