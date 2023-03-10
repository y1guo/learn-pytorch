import torch
import gc
import IPython.display as ipd
import gradio as gr
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from TTS.api import TTS


# freeup the model and free up GPU RAM
def freeup_VRAM(*args):
    memory_used_before = torch.cuda.memory_reserved(0) / 1024**3
    for var in args:
        try:
            del globals()[var]
            print(f"'{var}' deleted from memory.")
        except:
            pass
    gc.collect()
    torch.cuda.empty_cache()
    memory_used_after = torch.cuda.memory_reserved(0) / 1024**3
    print(f"Freed up {memory_used_before - memory_used_after:.1f} GB of VRAM.")


def load_gpt(model_name, model_dtype):
    if model_name.endswith("6B"):
        model_dtype = "int8"
    # elif model_name.endswidth('2.7B') and model_dtype == 'fp32':
    #     model_dtype = 'fp16'

    freeup_VRAM("model", "tokenizer")

    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_dtype == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", load_in_8bit=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if model_dtype == "fp16" else "auto",
        )

    model.eval()

    # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id

    return f"{model_name}  in  {model_dtype}"


# loading the text-to-speech model
freeup_VRAM("tts")

tts = TTS(
    model_name="tts_models/en/vctk/vits", gpu=True
)  # speaker: p243, p259, p263, p270, p306

example_few_shots = [
    """Yi: Hi, Pimon. Long time no see! How are you? My name is Yi. Do you still remember me?
Pimon: It's you, Yi! It's so good to see you again! I haven't seen you for a while. What are you doing?
Yi: Yeah, you are right, Pimon. I've been really busy working on the TA stuff. You know, I'm a teaching assistent. It took me two whole days to grade the quiz and homeowrks. I really wish I could get some sleep tonight.
Pimon: Oh...Poor you. So sorry to hear that. I'm sure you'll get better. After all, the work are all done, aren't they?
Yi: Yeah, probably. Thank you. I'm feeling better now.
Pimon: No problem. Oh, by the way, I'm glad that you still remember my name. I mean, Pimon is not a common name.
Yi: Of course I do. Don't you remember the days we travelled together? We were best friends.
Pimon: Oh, you're so sweet. We'll always be best friends. 
""",
    """Yi: My name is Yi.
Pimon: My name is Pimon
Yi: Your name is Pimon.
Pimon: Your name is Yi.
Yi: What is my name?
Pimon: Your name is Yi.
Yi: Your name is Pimon.
Pimon: Yes, my name is Pimon.
Yi: Your name is Jane.
Pimon: No, my name is not Jane. My name is Pimon.
""",
]

chat_input_sample = """Do you think I'm annoying?"""


def chat(
    few_shot_training_prompt,
    chat_history_list,
    chat_input,
    do_sample,
    temperature,
    top_k,
    top_p,
    max_new_tokens,
    eos_token_id,
    user_name,
    bot_name,
):
    # prepare prompt including few-shot and chat history
    # regulate each term every iteration since there might be '<br>' added into them.
    chat_history_list = [
        (chat_regulator(user_text), chat_regulator(bot_text))
        for (user_text, bot_text) in chat_history_list
    ]
    chat_history_list += [[chat_regulator(chat_input), ""]]
    chat_history = "\n".join(
        [
            f"{user_name}: {user_text}\n{bot_name}: {bot_text}"
            for (user_text, bot_text) in chat_history_list
        ]
    )
    prompt = few_shot_training_prompt.strip() + "\n" + chat_history.strip()

    # extract AI response
    try:
        # 10 trials of searching for the stop token, i.e., user_name+":". If not found, continue generating text.
        recur_prompt = prompt
        for _ in range(10):
            encoded_input = tokenizer(recur_prompt, return_tensors="pt")
            output_sequences = model.generate(
                input_ids=encoded_input["input_ids"].cuda(),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=int(eos_token_id),
            )
            generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            eos_idx = generated_text.find(user_name + ":", len(prompt))
            if eos_idx < 0:
                recur_prompt = generated_text
            else:
                break
    except:
        generated_text = prompt + "[System Error]: Failed to do inference."
        eos_idx = -1
        print("[System Error]: Failed to do inference.")

    chat_response = generated_text[len(prompt) : eos_idx].replace(bot_name + ":", "")
    regulated_response = chat_regulator(chat_response)
    chat_history_list[-1][1] = regulated_response

    # output to .wav using tts
    try:
        wav = tts.tts(text=regulated_response, speaker="p306")
    except:
        wav = [0 for _ in range(20000)]
        print(f"[System Error]: Failed to generate audio for '{regulated_response}'.")
    wav_int16 = (np.array(wav) * 32767).astype(np.int16)

    return (
        chat_history_list,
        "",
        (22050, wav_int16),
        str([generated_text]) + "\n" + str(chat_history_list),
    )


def chat_regulator(text):
    if not text:
        return "Silence..."
    # return " ".join(re.split(" |\t|\n|\r", text)).strip()
    return text.strip().replace("<br>", "")


with gr.Blocks() as demo:
    gr.Markdown("""# <center>???  ???  ???  ???</center>""")
    with gr.Row():
        with gr.Column(scale=1):
            gpt_name_radio = gr.Radio(
                [
                    "EleutherAI/gpt-neo-125M",
                    "EleutherAI/gpt-neo-1.3B",
                    "EleutherAI/gpt-neo-2.7B",
                    "EleutherAI/gpt-j-6B",
                ],
                value="EleutherAI/gpt-neo-2.7B",
                label="GPT model",
            )
            gpt_dtype_radio = gr.Radio(
                ["int8", "fp16", "fp32"], value="fp16", label="GPT dtype"
            )
            gpt_in_use_box = gr.Textbox(label="GPT in use", show_label=False)
            gpt_load_button = gr.Button("Load")
            do_sample_checkbox = gr.Checkbox(True, label="Do sample")
            temperature_slider = gr.Slider(0, 1.2, 0.8, label="Temperature")
            top_k_slider = gr.Slider(0, 100, 50, label="Top k")
            top_p_slider = gr.Slider(0, 1, 0.95, label="Top p")
            max_new_tokens_box = gr.Number(50, label="Max new tokens")
            eos_token_id_box = gr.Number(50256, label="Eos token id")
            user_name_box = gr.Textbox("Yi", label="User name")
            bot_name_box = gr.Textbox("Pimon", label="AI name")
        with gr.Column(scale=3):
            with gr.Accordion("Few shot traning prompt"):
                few_shot_training_prompt_box = gr.Textbox(show_label=False)
            clear_history_button = gr.Button("Clear History")
            chat_history_box = gr.Chatbot(label="History")
            chat_input_box = gr.Textbox(
                show_label=False, placeholder="Do you think I'm annoying?"
            )
            with gr.Accordion("Debug", open=False):
                debug_info_box = gr.Textbox(show_label=False)
        with gr.Column(scale=1):
            chat_response_audio = gr.Audio()
            example_few_shots_box = gr.Examples(
                example_few_shots, few_shot_training_prompt_box
            )

    gpt_load_button.click(
        load_gpt, inputs=[gpt_name_radio, gpt_dtype_radio], outputs=gpt_in_use_box
    )
    clear_history_button.click(
        lambda: ([], []), None, [chat_history_box, chat_history_box]
    )
    chat_input_box.submit(
        chat,
        [
            few_shot_training_prompt_box,
            chat_history_box,
            chat_input_box,
            do_sample_checkbox,
            temperature_slider,
            top_k_slider,
            top_p_slider,
            max_new_tokens_box,
            eos_token_id_box,
            user_name_box,
            bot_name_box,
        ],
        [
            chat_history_box,
            chat_input_box,
            chat_response_audio,
            debug_info_box,
        ],
    )

demo.launch(share=False)
