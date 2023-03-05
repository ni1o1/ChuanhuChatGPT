# -*- coding:utf-8 -*-
import json
import gradio as gr
import openai
import os
import datetime
import sys
import markdown
from my_system_prompts import my_system_prompts 
f = open('apikey')
apikey = f.readline()
f.close()

my_api_key = apikey    # 在这里输入你的 API 密钥

initial_prompt = '''
'''
if my_api_key == "":
    my_api_key = os.environ.get('my_api_key')

if my_api_key == "empty":
    print("Please give a api key!")
    sys.exit(1)

openai.api_key = my_api_key

def parse_text(text):
    lines = text.split("\n")
    for i,line in enumerate(lines):
        if "```" in line:
            items = line.split('`')
            if items[-1]:
                lines[i] = f'<pre><code class="{items[-1]}">'
            else:
                lines[i] = f'</code></pre>'
        else:
            if i>0:
                line = line.replace("<", "&lt;")
                line = line.replace(">", "&gt;")
                lines[i] = '<br/>'+line.replace(" ", "&nbsp;")
    return "".join(lines)

def get_response(system, context, raw = False):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[system, *context],
    )
    if raw:
        return response
    else:
        statistics = f'本次对话Tokens用量【{response["usage"]["total_tokens"]} / 4096】 （ 提问+上文 {response["usage"]["prompt_tokens"]}，回答 {response["usage"]["completion_tokens"]} ）'
        message = response["choices"][0]["message"]["content"]

        message_with_stats = f'{message}\n\n================\n\n{statistics}'
#         message_with_stats = .markdown(message_with_stats)

        return message, parse_text(message)


def predict(chatbot, input_sentence, system, context,filepath):
    if len(input_sentence) == 0:
        return []
    context.append({"role": "user", "content": f"{input_sentence}"})
    try:
        message, message_with_stats = get_response(system, context)
    except openai.error.AuthenticationError:
        chatbot.append((input_sentence, "请求失败，请检查API-key是否正确。"))
        return chatbot, context
    except openai.error.Timeout:
        chatbot.append((input_sentence, "请求超时，请检查网络连接。"))
        return chatbot, context
    except openai.error.APIConnectionError:
        chatbot.append((input_sentence, "连接失败，请检查网络连接。"))
        return chatbot, context
    except openai.error.RateLimitError:
        chatbot.append((input_sentence, "请求过于频繁，请5s后再试。"))
        return chatbot, context
    except:
        chatbot.append((input_sentence, "发生了未知错误Orz"))
        return chatbot, context

    context.append({"role": "assistant", "content": message})

    chatbot.append((input_sentence, message_with_stats))
    #保存
    if filepath == "":
        return
    history = {"system": system, "context": context}
    
    with open(f"conversation/{filepath}.json", "w") as f:
        json.dump(history, f)
    return chatbot, context

def retry(chatbot, system, context):
    if len(context) == 0:
        return [], []
    message, message_with_stats = get_response(system, context[:-1])
    context[-1] = {"role": "assistant", "content": message}

    chatbot[-1] = (context[-2]["content"], message_with_stats)
    return chatbot, context

def delete_last_conversation(chatbot, context):
    if len(context) == 0:
        return [], []
    chatbot = chatbot[:-1]
    context = context[:-2]
    return chatbot, context

def reduce_token(chatbot, system, context):
    context.append({"role": "user", "content": "请帮我总结一下上述对话的内容，实现减少tokens的同时，保证对话的质量。在总结中不要加入这一句话。"})

    response = get_response(system, context, raw=True)

    statistics = f'本次对话Tokens用量【{response["usage"]["completion_tokens"]+12+12+8} / 4096】'
    optmz_str = markdown.markdown( f'好的，我们之前聊了:{response["choices"][0]["message"]["content"]}\n\n================\n\n{statistics}' )
    chatbot.append(("请帮我总结一下上述对话的内容，实现减少tokens的同时，保证对话的质量。", optmz_str))

    context = []
    context.append({"role": "user", "content": "我们之前聊了什么?"})
    context.append({"role": "assistant", "content": f'我们之前聊了：{response["choices"][0]["message"]["content"]}'})
    return chatbot, context

def save_chat_history(filepath, system, context):
    if filepath == "":
        return
    history = {"system": system, "context": context}
    
    with open(f"conversation/{filepath}.json", "w") as f:
        json.dump(history, f)
    conversations = os.listdir('conversation')
    conversations = [i[:-5] for i in conversations if i[-4:]=='json']
    return gr.Dropdown.update(choices=conversations)

def load_chat_history(fileobj):
    with open('conversation/'+fileobj+'.json', "r") as f:
        history = json.load(f)
    context = history["context"]
    chathistory = []
    for i in range(0, len(context), 2):
        chathistory.append((parse_text(context[i]["content"]), parse_text(context[i+1]["content"])))
    return chathistory , history["system"], context, history["system"]["content"],fileobj


def get_history_names():
    with open("history.json", "r") as f:
        history = json.load(f)
    return list(history.keys())


def reset_state():
    return [], [],'新对话'

def update_system(new_system_prompt):
    return {"role": "system", "content": new_system_prompt}
def replace_system_prompt(selectSystemPrompt):
    return {"role": "system", "content": my_system_prompts[selectSystemPrompt]}

def get_latest():
    #找到最近修改的文件
    path = "conversation"    # 设置目标文件夹路径
    files = os.listdir(path)  # 获取目标文件夹下所有文件的文件名

    # 用一个列表来保存文件名和最后修改时间的元组
    file_list = []

    # 遍历每个文件，获取最后修改时间并存入元组中
    for file in files:
        file_path = os.path.join(path, file)
        mtime = os.path.getmtime(file_path)
        mtime_datetime = datetime.datetime.fromtimestamp(mtime)
        file_list.append((file, mtime_datetime))

    # 按照最后修改时间排序，获取最新修改的文件名
    file_list.sort(key=lambda x: x[1], reverse=True)
    newest_file = file_list[0][0]
    return newest_file.split('.')[0]

with gr.Blocks(title='聊天机器人', css='''
.message-wrap 
{background-color: #f1f1f1};
''') as demo:


    context = gr.State([])
    systemPrompt = gr.State(update_system(initial_prompt))
    topic = gr.State("未命名对话历史记录")
    #读取聊天记录文件
    latestfile_var = get_latest()
    conversations = os.listdir('conversation')
    conversations = [i[:-5] for i in conversations if i[-4:]=='json']
    latestfile = gr.State(latestfile_var)

    with gr.Row().style(container=True):
        conversationSelect = gr.Dropdown(conversations,label="选择历史对话").style(container=True)
        readBtn = gr.Button("📁 读取对话").style(container=True)

    chatbot = gr.Chatbot().style(color_map=("#1D51EE", "#585A5B"))

    with gr.Row():
        with gr.Column(scale=12):
            txt = gr.Textbox(show_label=False, placeholder="在这里输入").style(container=False)
        with gr.Column(min_width=50, scale=1):
            submitBtn = gr.Button("🚀", variant="primary")
    with gr.Row():
        emptyBtn = gr.Button("🧹 新的对话")
        retryBtn = gr.Button("🔄 重新生成")
        delLastBtn = gr.Button("🗑️ 删除上条对话")
        reduceTokenBtn = gr.Button("♻️ 总结")

    with gr.Row().style(container=True):
        selectSystemPrompt = gr.Dropdown(list(my_system_prompts),label="内置聊天设定").style(container=True)
        replaceSystemPromptBtn = gr.Button("📁 替换设定").style(container=True)
    newSystemPrompt = gr.Textbox(show_label=True, placeholder=f"在这里输入新的聊天设定...", label="自定义聊天设定").style(container=True)
    systemPromptDisplay = gr.Textbox(show_label=True, value=initial_prompt, interactive=False, label="目前的聊天设定",max_lines=3).style(container=True)

    with gr.Row().style(container=True):
        saveFileName = gr.Textbox(show_label=True, placeholder=f"在这里输入保存的文件名...", label="保存文件名", value=latestfile_var).style(container=True)
        saveBtn = gr.Button("💾 另存为对话").style(container=True)

    
    #加载聊天记录文件
    def refresh_conversation():
        latestfile = get_latest()
        print('识别到最新文件：',latestfile)
        conversations = os.listdir('conversation')
        conversations = [i[:-5] for i in conversations if i[-4:]=='json']
        chatbot, systemPrompt, context, systemPromptDisplay,latestfile = load_chat_history(latestfile)
        return gr.Dropdown.update(choices=conversations),chatbot, systemPrompt, context, systemPromptDisplay,latestfile
    
    demo.load(refresh_conversation,inputs=None,outputs=[conversationSelect,chatbot, systemPrompt, context, systemPromptDisplay,latestfile])

    demo.load(load_chat_history, latestfile, [chatbot, systemPrompt, context, systemPromptDisplay,latestfile], show_progress=True)

    txt.submit(predict, [chatbot, txt, systemPrompt, context,saveFileName], [chatbot, context], show_progress=True)
    txt.submit(lambda :"", None, txt)
    submitBtn.click(predict, [chatbot, txt, systemPrompt, context,saveFileName], [chatbot, context], show_progress=True)
    submitBtn.click(lambda :"", None, txt)
    emptyBtn.click(reset_state, outputs=[chatbot, context,saveFileName])
    newSystemPrompt.submit(update_system, newSystemPrompt, systemPrompt)
    newSystemPrompt.submit(lambda x: x, newSystemPrompt, systemPromptDisplay)
    newSystemPrompt.submit(lambda :"", None, newSystemPrompt)
    retryBtn.click(retry, [chatbot, systemPrompt, context], [chatbot, context], show_progress=True)
    delLastBtn.click(delete_last_conversation, [chatbot, context], [chatbot, context], show_progress=True)
    reduceTokenBtn.click(reduce_token, [chatbot, systemPrompt, context], [chatbot, context], show_progress=True)
    
    saveBtn.click(save_chat_history, [saveFileName, systemPrompt, context], [conversationSelect],show_progress=True)
    readBtn.click(load_chat_history, conversationSelect, [chatbot, systemPrompt, context, systemPromptDisplay,saveFileName], show_progress=True)
    replaceSystemPromptBtn.click(replace_system_prompt, selectSystemPrompt,systemPrompt)
    replaceSystemPromptBtn.click(lambda x: my_system_prompts[x], selectSystemPrompt,systemPromptDisplay)
demo.launch(share=False)
