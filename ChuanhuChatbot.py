# -*- coding:utf-8 -*-
import json
import gradio as gr
import openai
import os
import datetime
import sys
import markdown
from my_system_prompts import my_system_prompts
from configparser import ConfigParser

config_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'config.ini')
config = ConfigParser()
config.read(config_path)
if not config.has_section('my_api_key'):
    config.add_section('my_api_key')
    
try:
    my_api_key = config['my_api_key']['api_key']
except:
    my_api_key = "" 
# 在这里输入你的 API 密钥
initial_prompt = "You are a helpful assistant."

if my_api_key == "":
    my_api_key = os.environ.get('my_api_key')

if my_api_key == "empty":
    print("Please give a api key!")
    sys.exit(1)

if my_api_key == "":
    initial_keytxt = None
elif len(str(my_api_key)) == 51:
    initial_keytxt = "api-key：" + str(my_api_key[:4] + "..." + my_api_key[-4:])
else:
    initial_keytxt = "默认api-key无效，请重新输入"



def set_apikey(new_api_key, myKey):
    try:
        get_response(update_system(initial_prompt), [{"role": "user", "content": "test"}], new_api_key)
    except openai.error.AuthenticationError:
        return "无效的api-key", myKey
    except openai.error.Timeout:
        return "请求超时，请检查网络设置", myKey
    except openai.error.APIConnectionError:
        return "网络错误", myKey
    except:
        return "发生了未知错误Orz", myKey
    
    encryption_str = "验证成功，api-key已做遮挡处理：" + new_api_key[:4] + "..." + new_api_key[-4:]
    config.set('my_api_key', 'api_key', new_api_key)
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    return encryption_str, new_api_key

def parse_text(text):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "```" in line:
            items = line.split('`')
            if items[-1]:
                lines[i] = f'<pre><code class="{items[-1]}">'
            else:
                lines[i] = f'</code></pre>'
        else:
            if i > 0:
                line = line.replace("<", "&lt;")
                line = line.replace(">", "&gt;")
                lines[i] = '<br/>'+line.replace(" ", "&nbsp;")
    return "".join(lines)


def get_response(system, context, myKey,raw=False):
    openai.api_key = myKey
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[system, *context],
    )
    if raw:
        return response
    else:
        message = response["choices"][0]["message"]["content"]

        return message, parse_text(message)


def predict(chatbot, input_sentence, system, context, filepath,myKey):
    if len(input_sentence) == 0:
        return []
    context.append({"role": "user", "content": f"{input_sentence}"})

    
    try:
        message, message_with_stats = get_response(system, context,myKey)
    except openai.error.AuthenticationError:
        chatbot.append((input_sentence, "请求失败，请检查API-key是否正确。"))
        context = context[:-1]
        return chatbot, context
    except openai.error.Timeout:
        chatbot.append((input_sentence, "请求超时，请检查网络连接。"))
        context = context[:-1]
        return chatbot, context
    except openai.error.APIConnectionError:
        chatbot.append((input_sentence, "连接失败，请检查网络连接。"))
        context = context[:-1]
        return chatbot, context
    except openai.error.RateLimitError:
        chatbot.append((input_sentence, "请求过于频繁，请5s后再试。"))
        context = context[:-1]
        return chatbot, context
    except Exception as e:
        chatbot = chatbot[:-1]
        chatbot.append((input_sentence, "错误信息："+str(e))+"已将上一条信息删除，避免再次出错")
        context = context[:-2]
        return chatbot, context

    context.append({"role": "assistant", "content": message})

    chatbot.append((input_sentence, message_with_stats))
    # 保存
    if filepath == "":
        return
    history = {"system": system, "context": context}

    with open(f"conversation/{filepath}.json", "w") as f:
        json.dump(history, f)
    return chatbot, context

def delete_last_conversation(chatbot, context):
    if len(context) == 0:
        return [], []
    chatbot = chatbot[:-1]
    context = context[:-2]
    return chatbot, context


def reduce_token(chatbot, system, context,myKey):
    text = "请把上面的聊天内容总结一下"
    context.append(
        {"role": "user", "content": text})

    response = get_response(system, context, myKey,raw=True)

    optmz_str = markdown.markdown(response["choices"][0]["message"]["content"])
    chatbot.append((text, optmz_str))

    context = []
    context.append({"role": "user", "content": text})
    context.append(
        {"role": "assistant", "content": response["choices"][0]["message"]["content"]})
    return chatbot, context

def translate_eng(chatbot, system, context,myKey):
    text = "请把你的回答翻译成英语"
    context.append(
        {"role": "user", "content": text})

    response = get_response(system, context, myKey,raw=True)

    optmz_str = markdown.markdown(response["choices"][0]["message"]["content"])
    chatbot.append((text, optmz_str))

    context = []
    context.append({"role": "user", "content": text})
    context.append(
        {"role": "assistant", "content": response["choices"][0]["message"]["content"]})
    return chatbot, context

def save_chat_history(filepath, system, context):
    if filepath == "":
        return
    history = {"system": system, "context": context}

    with open(f"conversation/{filepath}.json", "w") as f:
        json.dump(history, f)
    conversations = os.listdir('conversation')
    conversations = [i[:-5] for i in conversations if i[-4:] == 'json']
    return gr.Dropdown.update(choices=conversations)


def load_chat_history(fileobj):
    with open('conversation/'+fileobj+'.json', "r") as f:
        history = json.load(f)
    context = history["context"]
    chathistory = []
    for i in range(0, len(context), 2):
        chathistory.append(
            (parse_text(context[i]["content"]), parse_text(context[i+1]["content"])))
    return chathistory, history["system"], context, history["system"]["content"], fileobj


def get_history_names():
    with open("history.json", "r") as f:
        history = json.load(f)
    return list(history.keys())


def reset_state():
    return [], [], '新对话'

def clear_state():
    return [], []

def update_system(new_system_prompt):
    return {"role": "system", "content": new_system_prompt}


def replace_system_prompt(selectSystemPrompt):
    return {"role": "system", "content": my_system_prompts[selectSystemPrompt]}


def get_latest():
    # 找到最近修改的文件
    path = "conversation"    # 设置目标文件夹路径
    files = os.listdir(path)  # 获取目标文件夹下所有文件的文件名

    # 用一个列表来保存文件名和最后修改时间的元组
    file_list = []

    # 遍历每个文件，获取最后修改时间并存入元组中
    for file in files:
        file_path = os.path.join(path, file)
        mtime = os.path.getmtime(file_path)
        mtime_datetime = datetime.datetime.fromtimestamp(mtime)
        if file[-4:]=='json':
            file_list.append((file, mtime_datetime))

    # 按照最后修改时间排序，获取最新修改的文件名
    file_list.sort(key=lambda x: x[1], reverse=True)
    newest_file = file_list[0][0]
    return newest_file.split('.')[0]



with gr.Blocks(title='聊天机器人', css='''
.message-wrap 
{height: 60vh;}
''') as demo:

    context = gr.State([])
    systemPrompt = gr.State(update_system(initial_prompt))
    myKey = gr.State(my_api_key)
    topic = gr.State("未命名对话历史记录")
    # 读取聊天记录文件
    latestfile_var = get_latest()
    conversations = os.listdir('conversation')
    conversations = [i[:-5] for i in conversations if i[-4:] == 'json']
    latestfile = gr.State(latestfile_var)
    print(my_api_key)

    if len(str(my_api_key)) != 51:
        keyTxt = gr.Textbox(show_label=True,label='OpenAI API-key',placeholder=f"在这里输入你的OpenAI API-key...", value=initial_keytxt)

    with gr.Accordion(label="选择历史对话",open=True):
        with gr.Row():
            conversationSelect = gr.Dropdown(
                conversations,value=latestfile_var,show_label=False,  label="选择历史对话").style(container=True)
            readBtn = gr.Button("📁 读取对话").style(container=True)
            
    with gr.Box():
        with gr.Row():
            with gr.Column(scale=15):
                thisconvername = gr.Markdown('<center>'+latestfile_var+'</center>')
                gr.Markdown('')
        with gr.Row():
            with gr.Column(scale=1,min_width=68):
                emptyBtn = gr.Button("新建")
                clearBtn = gr.Button("清空")
                delLastBtn = gr.Button("撤回")
                reduceTokenBtn = gr.Button("总结")
                translateBtn = gr.Button("翻英")

            with gr.Column(scale=12):
                chatbot = gr.Chatbot().style(color_map=("#1D51EE", "#585A5B"))
                with gr.Row():
                    with gr.Column(scale=12):
                        txt = gr.Textbox(show_label=False, placeholder="在这里输入").style(
                            container=False)
                    with gr.Column(min_width=20, scale=1):
                        submitBtn = gr.Button("↑", variant="primary")
                        

    with gr.Box():
        gr.Markdown('聊天设定')
        with gr.Row(variant='panel').style(container=True):
            selectSystemPrompt = gr.Dropdown(
                list(my_system_prompts), label="选择内置聊天设定").style(container=True)
            replaceSystemPromptBtn = gr.Button("📁 替换设定").style(container=True)
        newSystemPrompt = gr.Textbox(
            show_label=True, placeholder=f"在这里输入新的聊天设定...", label="自定义聊天设定").style(container=True)
        systemPromptDisplay = gr.Textbox(show_label=True, value=initial_prompt,
                                         interactive=False, label="目前的聊天设定", max_lines=3).style(container=True)

    with gr.Box():
        gr.Markdown('对话另存为')
        with gr.Row():
            with gr.Column(scale=15):
                saveFileName = gr.Textbox(show_label=False, placeholder=f"在这里输入保存的文件名...",
                                         value=latestfile_var).style(
                            container=False)
            with gr.Column(min_width=20,scale=1):
                saveBtn = gr.Button("💾").style(container=True)

    if len(str(my_api_key)) == 51:
        keyTxt = gr.Textbox(show_label=True,label='OpenAI API-key',placeholder=f"在这里输入你的OpenAI API-key...", value=initial_keytxt)

    # 加载聊天记录文件

    def refresh_conversation():
        latestfile = get_latest()
        print('识别到最新文件：', latestfile)
        conversations = os.listdir('conversation')
        conversations = [i[:-5] for i in conversations if i[-4:] == 'json']
        chatbot, systemPrompt, context, systemPromptDisplay, latestfile = load_chat_history(
            latestfile)
        return gr.Dropdown.update(choices=conversations), chatbot, systemPrompt, context, systemPromptDisplay, latestfile
    
    demo.load(refresh_conversation, inputs=None, outputs=[
              conversationSelect, chatbot, systemPrompt, context, systemPromptDisplay, latestfile])
    demo.load(load_chat_history, latestfile, [
              chatbot, systemPrompt, context, systemPromptDisplay, latestfile], show_progress=True)
    txt.submit(predict, [chatbot, txt, systemPrompt, context, saveFileName,myKey], [
               chatbot, context], show_progress=True)
    txt.submit(lambda: "", None, txt)
    submitBtn.click(predict, [chatbot, txt, systemPrompt, context, saveFileName,myKey], [
                    chatbot, context], show_progress=True,scroll_to_output = True)
    submitBtn.click(lambda: "", None, txt)
    emptyBtn.click(reset_state, outputs=[chatbot, context, saveFileName])
    clearBtn.click(clear_state, outputs=[chatbot, context])
    newSystemPrompt.submit(update_system, newSystemPrompt, systemPrompt)
    newSystemPrompt.submit(lambda x: x, newSystemPrompt, systemPromptDisplay)
    newSystemPrompt.submit(lambda: "", None, newSystemPrompt)
    delLastBtn.click(delete_last_conversation, [chatbot, context], [
                     chatbot, context], show_progress=True)
    reduceTokenBtn.click(reduce_token, [chatbot, systemPrompt, context,myKey], [
                         chatbot, context], show_progress=True)
    translateBtn.click(translate_eng, [chatbot, systemPrompt, context,myKey], [
                         chatbot, context], show_progress=True)
    saveBtn.click(save_chat_history, [saveFileName, systemPrompt, context], [
                  conversationSelect], show_progress=True)
    readBtn.click(load_chat_history, conversationSelect, [
                  chatbot, systemPrompt, context, systemPromptDisplay, saveFileName], show_progress=True)
    replaceSystemPromptBtn.click(
        replace_system_prompt, selectSystemPrompt, systemPrompt)
    replaceSystemPromptBtn.click(
        lambda x: my_system_prompts[x], selectSystemPrompt, systemPromptDisplay)
    keyTxt.submit(set_apikey, [keyTxt, myKey], [keyTxt, myKey], show_progress=True)

    saveFileName.change(lambda x:'<center>'+x+'</center>',saveFileName,thisconvername)

demo.launch(share=False)
