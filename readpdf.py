import re
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings
import pdfplumber
import pandas as pd
import tiktoken
# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")


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
    return pd.DataFrame([chars['text'].sum()], columns=['text'])

# Function to split the text into chunks of a maximum number of tokens


def split_into_many(text, max_tokens=500):

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


def shortened_text(df, max_tokens=500):
    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'], max_tokens=max_tokens)

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append(row[1]['text'])

    df = pd.DataFrame(shortened, columns=['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    return df


def embedding_df(df, api_key):
    openai.api_key = api_key

    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(
        input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    df['embeddings'] = df['embeddings'].apply(np.array)
    return df


def create_context(
    question, df,     api_key, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """
    openai.api_key = api_key

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(
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
        response = openai.ChatCompletion.create(
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
