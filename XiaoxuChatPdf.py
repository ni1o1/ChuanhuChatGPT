# -*- coding:utf-8 -*-
from configparser import ConfigParser
import json
import gradio as gr
import openai
import os
import datetime
import sys
from mycss import mycss
import csv
import tiktoken
from readpdf import *
my_system_prompts = {}
with open('my_system_prompts.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # 跳过标题行，获取第二行开始的数据
    for row in reader:
        my_system_prompts[row[0]] = row[1]


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


def postprocess(self, y):
    """
    Parameters:
        y: List of tuples representing the message and response pairs. Each message and response should be a string, which may be in Markdown format.
    Returns:
        List of tuples representing the message and response. Each message and response will be a string of HTML.
    """
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            # None if message is None else markdown.markdown(message),
            # None if response is None else markdown.markdown(response),
            None if message is None else parse_text(message),
            None if response is None else parse_text(response),
        )
    return y
# 开启latex
#gr.Chatbot.postprocess = postprocess


def set_apikey(new_api_key, myKey):
    try:
        get_response(update_system(initial_prompt), [
                     {"role": "user", "content": "test"}], new_api_key)
    except openai.error.AuthenticationError:
        return "无效的api-key", myKey
    except openai.error.Timeout:
        return "请求超时，请检查网络设置", myKey
    except openai.error.APIConnectionError:
        return "网络错误", myKey
    except:
        return "发生了未知错误Orz", myKey

    encryption_str = "验证成功，api-key已做遮挡处理：" + \
        new_api_key[:4] + "..." + new_api_key[-4:]
    config.set('my_api_key', 'api_key', new_api_key)
    with open(config_path, 'w') as configfile:
        config.write(configfile)
    return encryption_str, new_api_key


def parse_text(line):
    line = line.replace("\n", "<br/>")
    line = line.replace("`", "\`")
    line = line.replace("<", "&lt;")
    line = line.replace(">", "&gt;")
    line = line.replace(" ", "&nbsp;")
    line = line.replace("*", "&ast;")
    line = line.replace("_", "&lowbar;")
    line = line.replace("-", "&#45;")
    line = line.replace(".", "&#46;")
    line = line.replace("!", "&#33;")
    line = line.replace("(", "&#40;")
    line = line.replace(")", "&#41;")
    line = line.replace("$", "&#36;")
    return line



def get_response(system, context, myKey, temperature=1, presence_penalty=0, frequency_penalty=0, raw=False):
    openai.api_key = myKey
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[system, *context],
        temperature=temperature,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty
    )

    if raw:
        return response
    else:
        statistics = int(response["usage"]["total_tokens"])/4096
        message = response["choices"][0]["message"]["content"]

        return message, parse_text(message), {'对话Token用量': min(statistics, 1)}

# 缩短对话


def num_tokens_from_string(string, encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def count_all_token(systemPrompt, context):
    # 计算当前对话的token
    s = systemPrompt['content']
    for i in context:
        s += i['content']
    total_token = num_tokens_from_string(s, "cl100k_base")
    return total_token


def conclude_context(systemPrompt, context, myKey):
    # 计算当前对话的token
    total_token = count_all_token(systemPrompt, context)
    if total_token > 3000:
        print('缩短对话中')

        # 设定的token
        systemPrompt_token = num_tokens_from_string(
            systemPrompt['content'], "cl100k_base")
        t = systemPrompt_token

        # 要总结的context
        old_context = []
        new_context = []
        for i in range(int(len(context)/2)):
            user_word = context[i*2]['content']
            assistant_word = context[i*2+1]['content']
            num_token = num_tokens_from_string(
                user_word+assistant_word, "cl100k_base")
            t += num_token
            if t < (total_token-systemPrompt_token)*1/2+systemPrompt_token:
                old_context.append(context[i*2])
                old_context.append(context[i*2+1])
            else:
                new_context.append(context[i*2])
                new_context.append(context[i*2+1])

        # 进行总结
        text = '总结以上对话，100字内'
        old_context.append(
            {"role": "user", "content": text})
        message, _, statistics = get_response(
            systemPrompt, old_context, myKey, raw=False)

        # 新context
        new_context = [{"role": "user", "content": "前面对话是什么内容"},
                       {"role": "assistant", "content": message}]+new_context
        print('缩短成功，开始继续对话', total_token,
              count_all_token(systemPrompt, new_context))
        return new_context
    else:
        return context


def predict(chatbot, input_sentence, system, context, filepath, myKey, temperature, presence_penalty, frequency_penalty):
    # 如果太长，则缩短
    context = conclude_context(system, context, myKey)

    # 开始predict

    context.append({"role": "user", "content": f"{input_sentence}"})

    try:
        message, message_with_stats, statistics = get_response(
            system, context, myKey, temperature, presence_penalty, frequency_penalty)
    except openai.error.AuthenticationError:
        chatbot.append((input_sentence, "请求失败，请检查API-key是否正确。"))
        context = context[:-1]
        return chatbot, context, {'对话Token用量': 0}
    except openai.error.Timeout:
        chatbot.append((input_sentence, "请求超时，请检查网络连接。"))
        context = context[:-1]
        return chatbot, context, {'对话Token用量': 0}
    except openai.error.APIConnectionError:
        chatbot.append((input_sentence, "连接失败，请检查网络连接。"))
        context = context[:-1]
        return chatbot, context, {'对话Token用量': 0}
    except openai.error.RateLimitError:
        chatbot.append((input_sentence, "请求过于频繁，请5s后再试。"))
        context = context[:-1]
        return chatbot, context, {'对话Token用量': 0}
    except Exception as e:
        chatbot.append((input_sentence, "错误信息："+str(e)+"已将上一条信息删除，避免再次出错"))
        context = context[:-1]
        return chatbot, context, {'对话Token用量': 0}

    context.append({"role": "assistant", "content": message})
    chatbot.append((input_sentence, message_with_stats))
    # 保存
    save_chathistory(filepath, system, context, chatbot)
    return chatbot, context, statistics


def load_chat_history(fileobj):
    with open('conversation/'+fileobj+'.json', "r") as f:
        history = json.load(f)
    context = history["context"]
    try:
        chatbot = history['chatbot']
    except:
        chatbot = []
        for i in range(0, len(context), 2):
            chatbot.append(
                (parse_text(context[i]["content"]), parse_text(context[i+1]["content"])))

    return chatbot, history["system"], context, history["system"]["content"], fileobj


def get_history_names():
    with open("history.json", "r") as f:
        history = json.load(f)
    return list(history.keys())


def reset_state():
    return [], [], '新对话，点击这里改名', update_system(initial_prompt), initial_prompt


def clear_state(filepath, system):
    save_chathistory(filepath, system, [], [])
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
        if file[-4:] == 'json':
            file_list.append((file, mtime_datetime))

    # 按照最后修改时间排序，获取最新修改的文件名
    file_list.sort(key=lambda x: x[1], reverse=True)
    newest_file = file_list[0][0]
    return newest_file.split('.')[0]


def sendmessage(text, system, context, chatbot, myKey):
    context.append(
        {"role": "user", "content": text})
    message, _, statistics = get_response(
        system, context, myKey)

    chatbot.append((text, message))
    context.append({"role": "assistant", "content": message})
    return chatbot, context, statistics


def save_chathistory(filepath, system, context, chatbot):
    # 保存
    if filepath == "":
        return
    history = {"system": system, "context": context, "chatbot": chatbot}

    with open(f"conversation/{filepath}.json", "w") as f:
        json.dump(history, f)
    conversations = os.listdir('conversation')
    conversations = [i[:-5] for i in conversations if i[-4:] == 'json']
    return gr.Dropdown.update(choices=conversations)

# 自定义功能


def del_chat(filepath):
    os.remove(f"conversation/{filepath}.json")
    conversations = os.listdir('conversation')
    conversations = [i[:-5] for i in conversations if i[-4:] == 'json']
    return [], [], '新对话，点击这里改名', update_system(initial_prompt), initial_prompt, gr.Dropdown.update(choices=conversations)


def delete_last_conversation(chatbot, system, context, filepath):
    if len(context) <= 2:
        return chatbot, context
    else:
        chatbot = chatbot[:-1]
        context = context[:-2]
        save_chathistory(filepath, system, context, chatbot)
        return chatbot, context


def reduce_token(chatbot, system, context, myKey, filepath):
    text = "请把上面的聊天内容总结一下"
    chatbot, context, statistics = sendmessage(
        text, system, context, chatbot, myKey)
    save_chathistory(filepath, system, context, chatbot)
    return chatbot, context, statistics


def translate_eng(chatbot, system, context, myKey, filepath):
    text = "请把你的回答翻译成英语"
    chatbot, context, statistics = sendmessage(
        text, system, context, chatbot, myKey)
    save_chathistory(filepath, system, context, chatbot)
    return chatbot, context, statistics


def translate_ch(chatbot, system, context, myKey, filepath):
    text = "请把你的回答翻译成中文"
    chatbot, context, statistics = sendmessage(
        text, system, context, chatbot, myKey)
    save_chathistory(filepath, system, context, chatbot)
    return chatbot, context, statistics


def brainstorn(chatbot, system, context, myKey, filepath):
    text = "请你联想一下，还有吗？"
    chatbot, context, statistics = sendmessage(
        text, system, context, chatbot, myKey)
    save_chathistory(filepath, system, context, chatbot)
    return chatbot, context, statistics


def shorter(chatbot, system, context, myKey, filepath):
    text = "把你上面的回答精简一下"
    chatbot, context, statistics = sendmessage(
        text, system, context, chatbot, myKey)
    save_chathistory(filepath, system, context, chatbot)
    return chatbot, context, statistics


def longer(chatbot, system, context, myKey, filepath):
    text = "把你上面的回答扩展一下"
    chatbot, context, statistics = sendmessage(
        text, system, context, chatbot, myKey)
    save_chathistory(filepath, system, context, chatbot)
    return chatbot, context, statistics


def scholar(chatbot, system, context, myKey, filepath):
    text = "把你上面的回答换为更加正式、专业、学术的语气"
    chatbot, context, statistics = sendmessage(
        text, system, context, chatbot, myKey)
    save_chathistory(filepath, system, context, chatbot)
    return chatbot, context, statistics


def points(chatbot, system, context, myKey, filepath):
    text = "把你上面的回答分点阐述"
    chatbot, context, statistics = sendmessage(
        text, system, context, chatbot, myKey)
    save_chathistory(filepath, system, context, chatbot)
    return chatbot, context, statistics


def prase(chatbot, system, context, myKey, filepath):
    text = "请你结合上面的内容，夸一夸我，给我一些鼓励"
    chatbot, context, statistics = sendmessage(
        text, system, context, chatbot, myKey)
    save_chathistory(filepath, system, context, chatbot)
    return chatbot, context, statistics


def explain(chatbot, system, context, myKey, filepath):
    text = "你说得太复杂了，请用小朋友都能懂的方式详细解释一下"
    chatbot, context, statistics = sendmessage(
        text, system, context, chatbot, myKey)
    save_chathistory(filepath, system, context, chatbot)
    return chatbot, context, statistics


def resend(chatbot, system, context, myKey, filepath):
    text = context[-2]['content']
    context = context[:-2]
    chatbot = chatbot[:-1]
    chatbot, context, statistics = sendmessage(
        text, system, context, chatbot, myKey)
    save_chathistory(filepath, system, context, chatbot)
    return chatbot, context, statistics


def load_pdf(file_obj, myKey):
    '''
    PDF embedding
    '''
    df = read_pdf(file_obj.name)
    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df = shortened_text(df, max_tokens=800)
    df = embedding_df(df, myKey)
    import json
    df_embedding_json = json.loads(df.to_json())
    return df_embedding_json, '读取完成'


def predict_pdf(chatbot, txt, df_embedding_json, myKey):

    if df_embedding_json==[]:
        pass
    else:
        df = pd.DataFrame(df_embedding_json)
        message = answer_question(df,myKey, question=txt, max_len=3500,debug=False)
        chatbot.append((txt, message.replace('\n','<br/>').replace(" ", "&nbsp;")))
        return chatbot, '回答完成'
def mindgraph(chatbot,  df_embedding_json, myKey):
    
    if df_embedding_json==[]:
        pass
    else:
        txt = '用中文对这篇文章做一个详细的思维导图，必须包含多个层级，给出详细结论，不包含参考文献，必须用markdown，在新窗口生成代码，请不要用mermaid'
        df = pd.DataFrame(df_embedding_json)
        message = answer_question(df,myKey, question=txt, max_len=3500,debug=False)
        chatbot.append((txt, '打开<a href="https://markmap.js.org/repl" target="_blank" color="" style="text-decoration:underline;color:blue">这个页面</a>，粘贴下面内容<br/><pre><code>'+message.replace('\n','<br/>').replace(" ", "&nbsp;")+'</code></pre>'))
        return chatbot, '回答完成'

title = """<h3 align="center">ChatPDF By 小旭学长</h3>"""

with gr.Blocks(title='聊天机器人', css=mycss) as demo:
    context = gr.State([])
    df_embedding_json = gr.State([])
    systemPrompt = gr.State(update_system(initial_prompt))
    myKey = gr.State(my_api_key)
    topic = gr.State("未命名对话历史记录")
    # 读取聊天记录文件
    latestfile_var = get_latest()

    conversations = os.listdir('conversation')
    conversations = [i[:-5] for i in conversations if i[-4:] == 'json']
    latestfile = gr.State(latestfile_var)

    gr.HTML(title)
    if len(str(my_api_key)) != 51:
        keyTxt = gr.Textbox(show_label=True, label='OpenAI API-key',
                            placeholder=f"在这里输入你的OpenAI API-key...", value=initial_keytxt)


    file_obj = gr.File(label='选择pdf文件', show_label=False).style(container=True)
    file_read_label = gr.Label(value='选择pdf文件').style(container=True)

    with gr.Box():
        with gr.Column(scale=12):
            chatbot = gr.Chatbot(show_label=False, elem_id='chatbot').style(
                color_map=("#7beb67", "#FFF"))
            with gr.Row():
                with gr.Column(scale=12):
                    txt = gr.Textbox(show_label=False, placeholder="在这里输入").style(
                        container=False)
                with gr.Column(min_width=20, scale=1):
                    submitBtn = gr.Button("↑", variant="primary")

        gr.Markdown('# ')
        with gr.Row():
            with gr.Column(scale=1, min_width=68):
                mindgraphBtn = gr.Button("思维导图")

    if len(str(my_api_key)) == 51:
        keyTxt = gr.Textbox(show_label=True, label='OpenAI API-key',
                            placeholder=f"在这里输入你的OpenAI API-key...", value=initial_keytxt)


    txt.submit(predict_pdf, [chatbot, txt, df_embedding_json, myKey], [chatbot,file_read_label], show_progress=True)
    txt.submit(lambda: "", None, txt)

    submitBtn.click(predict_pdf, [chatbot, txt, df_embedding_json, myKey], [chatbot,file_read_label], show_progress=True)
    submitBtn.click(lambda: "", None, txt)

    keyTxt.submit(set_apikey, [keyTxt, myKey], [
                  keyTxt, myKey], show_progress=True)
    mindgraphBtn.click(mindgraph, [chatbot, df_embedding_json, myKey], [chatbot,file_read_label], show_progress=True)

    file_obj.change(load_pdf, [file_obj, myKey], [
                    df_embedding_json, file_read_label], show_progress=True)
demo.launch(share=True)
