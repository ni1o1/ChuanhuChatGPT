import re
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings
import pdfplumber
import pandas as pd
import tiktoken
#导入解析包
from bs4 import BeautifulSoup
import requests
from urllib.parse import quote
import docx

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chatcompletion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def embedding_with_backoff(**kwargs):
    return openai.Embedding.create(**kwargs)

def read_url(url):
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36"
                                        "(KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36"
                            }
    response = requests.get(url,'lxml', headers=headers)
    response.encoding = response.apparent_encoding
    response.raise_for_status()
    content = response.content
    #创建beautifulsoup解析对象
    bs_1=BeautifulSoup(content,'lxml')
    chars = pd.DataFrame([re.sub('\s+', ' ', i) for i in bs_1.text.split('\n') if (i != '')&(re.sub('\s+', ' ', i)!=' ')],columns=['text'])
    chars['n_tokens'] = chars.text.apply(lambda x: len(tokenizer.encode(x)))
    url_title = bs_1.find('title').text
    return chars,url_title

def read_pdf(filename):
    chars = []
    t = 0
    with pdfplumber.open(filename) as pdf:
        for page in pdf.pages:
            t += 1
            char = pd.DataFrame(page.chars)
            char['page'] = t
            chars.append(char)
    chars = pd.concat(chars)
    
    chars['group'] = (chars['size'].shift()!=chars['size']).cumsum()
    chars = chars.groupby(['group'])['text'].sum().reset_index()[['text']]
    chars['n_tokens'] = chars.text.apply(lambda x: len(tokenizer.encode(x)))
    return chars


def read_docx(filename):
    # 打开Word文档
    document = docx.Document(filename)
    paras = []
    # 输出文档段落（段落中的内容）
    for para in document.paragraphs:
        paras.append(para.text)
    chars = pd.DataFrame([re.sub('\s+', ' ', i) for i in paras if (i != '')&(re.sub('\s+', ' ', i)!=' ')],columns=['text'])
    chars['n_tokens'] = chars.text.apply(lambda x: len(tokenizer.encode(x)))
    return chars

def read_txt(filename):
    f = open(filename)
    paras = f.readlines()
    f.close()
    chars = pd.DataFrame([re.sub('\s+', ' ', i) for i in paras if (i != '')&(re.sub('\s+', ' ', i)!=' ')],columns=['text'])
    chars['n_tokens'] = chars.text.apply(lambda x: len(tokenizer.encode(x)))
    return chars

def read_question(question):
    headers = {
"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36",
    }
    url = 'http://cn.bing.com/search?q='+quote(question)
    response = requests.get(url, 'lxml', headers=headers)
    response.encoding = response.apparent_encoding
    response.raise_for_status()
    content = response.content
    # 创建beautifulsoup解析对象
    bs_1 = BeautifulSoup(content, 'lxml')
    chars = pd.DataFrame([re.sub('\s+', ' ', i) for i in bs_1.text.split('\n')
                         if (i != '') & (re.sub('\s+', ' ', i) != ' ')], columns=['text'])
    chars['n_tokens'] = chars.text.apply(lambda x: len(tokenizer.encode(x)))
    url_title = bs_1.find('title').text
    return chars, url_title


def split_into_many(text, max_tokens=500):
    # Function to split the text into chunks of a maximum number of tokens
    # Split the text into sentences
    sentences = re.split("[.。 ]", text)

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence))
                for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(" ".join(chunk))
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        # if token > max_tokens:
        #    continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks



def shortened_text(df,max_tokens=500):
    #切分长句子
    texts = []
    for i in range(len(df)):
        r = df.iloc[i]
        n_tokens = r['n_tokens']
        text = r['text']
        if n_tokens>max_tokens:
            longtext = pd.DataFrame(list(text))
            longtext['group'] = ((longtext[0]=='.')|(longtext[0]=='。')|(longtext[0]==' ')).shift().cumsum().fillna(0)
            longtext = longtext.groupby('group')[0].sum()
            texts+= list(longtext)
        else:
            texts.append(text)
    texts = pd.DataFrame(texts,columns = ['text'])
    texts['n_tokens'] = texts.text.apply(lambda x: len(tokenizer.encode(x)))

    #拼起来
    token_count = 0
    splited_texts = []
    thistext = []
    for i in range(len(texts)):
        r = texts.iloc[i]
        n_tokens = r['n_tokens']
        text = r['text']
        if token_count+n_tokens<max_tokens:
            thistext.append(text)
            token_count += n_tokens
        else:
            splited_texts.append(''.join(thistext).replace('  ',' '))
            thistext = [text]
            token_count = n_tokens
    splited_texts.append(''.join(thistext).replace('  ',' '))
    splited_texts = pd.DataFrame(splited_texts,columns=['text'])

    splited_texts['n_tokens'] = splited_texts.text.apply(lambda x: len(tokenizer.encode(x)))
    return splited_texts


def embedding_df(df, api_key):
    openai.api_key = api_key

    df['embeddings'] = df.text.apply(lambda x: embedding_with_backoff(
        input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    df['embeddings'] = df['embeddings'].apply(np.array)
    return df


def create_context(
    question, df, api_key, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """
    openai.api_key = api_key

    # Get the embeddings for the question
    q_embeddings = embedding_with_backoff(
        input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(
        q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n".join(returns)


def answer_question(
    df,
    api_key,
    model="gpt-3.5-turbo",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    openai.api_key = api_key
    context = create_context(
        question,
        df,
        api_key,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = chatcompletion_with_backoff(
            messages=[{'role': "system", "content": '根据以下文章内容，用中文回答问题，文章如下：'+context},
                      {"role": "user", "content": question}],
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model=model,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(e)
        return ""

def get_mind_graph(message):
    '''
    整理思维导图的信息
    '''
    mind_graph = pd.DataFrame(message.split('\n'))
    mind_graph = mind_graph[mind_graph[0]!='']
    texts = []
    for i in range(len(mind_graph)):
        text = mind_graph[0].iloc[i]

        if re.split('[. ]',text)[0]!='':
            if re.split('[. ]',text)[0][0] in '0123456789':
                text = '- '+'.'.join(text.split('.')[1:])
            else:
                if re.split('[. ]',text)[0][0] not in '#-':
                    text = '- '+text
        texts.append(text.replace('-  ','- ').replace('<code>','').replace('</code>',''))
    mind_graph = '\n'.join(texts)
    return mind_graph