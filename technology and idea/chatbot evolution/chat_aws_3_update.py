all_questions = []
all_user = []
all_chatbot = []


# adjust the speaker identity
def edit_identity(sentence):
    forbidden = ["I am not dog", "I am not a dog", "I am not doge", "I am not a doge", "I am a human", "I am a person",
                 "I am a dog person", "I am a cat person", "I'm sorry to hear that", "I have a dog", "I have a cat",
                 "it is hard for me to believe", "I'm sorry about that", "I am so sorry to hear that"
                                                                         "I have a dog and a cat",
                 "I have a cat and a dog"]
    response_tmp = ''
    for i in sentence.split(".")[:-1]:
        skip = False
        for l in forbidden:
            if l in i:
                skip = True
                break
        if skip == False:
            response_tmp += i.strip() + ". "
    return response_tmp[:-1].replace("either", "").replace("my dog", "myself")


# check if the question is asked before if not store the question, if asked, skip the question
def question_asked(sentence):
    for l in all_questions:
        if sentence.split("?")[0] in l:
            return "True"
    all_questions.append(sentence.split("?")[0])
    return "False"


"""
Open Model
"""
from flask import render_template, request, Flask
from flask_sqlalchemy import SQLAlchemy
from parlai.core.agents import create_agent_from_model_file
from datetime import datetime

blender_msc_agent = create_agent_from_model_file("zoo:msc/blender3B_1024/model")


def add_persona(context_temp):
    persona = []
    for i in context_temp:
        persona.append('your persona: ' + i)
    return persona


def add_partner_persona(context_temp):
    persona = []
    for i in context_temp:
        persona.append("partner's persona: " + i)
    return persona


def process_input(context_temp):
    return "\n".join(context_temp)


def find_response_open(context_temp, persona, partner_persona, agent):
    agent.reset()
    info = add_persona(persona) + add_partner_persona(partner_persona)
    for i in context_temp:
        info.append(i)
    # if len(info[-1]) <= 8:
    #     agent.observe({'text': process_input(info), 'episode_done': False})
    #     response = agent.act()
    #     print("Short extra", response['text'])
    #     for i in response['text'].split("."):
    #         if "?" not in i:
    #             info[-1] += " " + i + ". "
    #     agent.reset()
    agent.observe({'text': process_input(info), 'episode_done': False})
    response = agent.act()
    print("Origin blender", response['text'])
    response_tmp = ''
    response_list_tmp = []
    if response['text'].split(".")[-1] != '':
        response_list_tmp = response['text'].split(".")
    else:
        response_list_tmp = response['text'].split(".")[:-1]
    for i in response_list_tmp:
        if "?" in i:
            asked = question_asked(i)
            if asked == "False":
                response_tmp += i.strip() + ". "
        else:
            response_tmp += i.strip() + ". "
    response_tmp = response_tmp[:-1]
    response_tmp = edit_identity(response_tmp)
    if len(response_tmp) <= 5:
        response_tmp = "yeah"
    return response_tmp.replace("person", "dog").replace("take my dog", "go")


"""
Acquire Model
"""
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('stsb-roberta-large')
import numpy as np
import faiss
import random

raw = {}


def get_one_combination(info):
    tmp_key = ""
    context = ""
    count = 0
    for i in info:
        count += 1
        if count == 2:
            count = 0
            context = ""
        context += i[3:-1]
        if tmp_key != "" and i[0] == "D":
            try:
                if i[3:-1] not in raw[tmp_key]:
                    raw[tmp_key].append(i[3:-1])
            except:
                raw[tmp_key] = [i[3:-1]]
        elif i[0] == "P":
            tmp_key = context


def get_all_combination(info):
    for i in range(len(info)):
        get_one_combination(info[i:])


with open('deploy_doge_1/acquire/dogs', 'r', encoding='utf-8') as file:
    store_info = []
    for i in file:
        if i == "\n":
            get_all_combination(store_info)
            store_info = []
        else:
            store_info.append(i)

for index, sentence in enumerate(raw.keys()):
    if index == 0:
        sentence_embeddings = model.encode([sentence])
    else:
        sentence_embeddings = np.append(sentence_embeddings, model.encode([sentence]), axis=0)

d = 1024
index = faiss.IndexFlatIP(d)
faiss.normalize_L2(sentence_embeddings)
index.add(sentence_embeddings)


def find_response_acquire(context):
    k = 4
    context_tmp = ''
    if len(context[-1]) <= 10:
        for i in context:
            context_tmp += i
        xq = model.encode([context])
    else:
        xq = model.encode([context[-1]])
    D, I = index.search(xq, k)
    return [random.choice(raw[list(raw.keys())[i]]).lower() for i in I[0]]


# def find_response_acquire(context):
#     k = 4
#     context_tmp = ''
#     if len(context[-1]) <= 10:
#         for i in context:
#             context_tmp += i
#         xq = model.encode([context])
#     else:
#         xq = model.encode([context[-1]])
#     D, I = index.search(xq, k)
#     if D[0][0] >= 21:
#         print(raw[list(raw.keys())[I[0][0]]])
#         return raw[list(raw.keys())[I[0][0]]].lower()
#     else:
#         xq = model.encode([context[-1]])
#         D, I = index.search(xq, k)
#         return [random.choice(raw[list(raw.keys())[i]]).lower() for i in I[0]]

from deploy_doge_1.acquire.acquire_deploy import NeuralNetwork
from deploy_doge_1.acquire.config import args as args_acquire
import random

model_acquire = NeuralNetwork(args=args_acquire)
model_acquire.load_model(args_acquire.save_path)
data_acquire = open("deploy_doge_1/acquire/dogs_result", 'r', encoding="utf-8").readlines()


## acquire model generate response
def find_response_acquired(context_temp, data):
    result_acquire = model_acquire.predict('', context_temp, data)
    final = random.choice(result_acquire)
    final = data[int(final)][:-1]
    return final


def find_multiple_response_acquired(context_temp, data):
    result_acquire = model_acquire.predict('', context_temp, data)
    multiple_final = []
    for i in result_acquire:
        multiple_final.append(data[int(i)][:-1])
    return multiple_final


"""
Summary Model
"""
from parlai.core.agents import create_agent_from_model_file

summary_agent = create_agent_from_model_file("zoo:msc/dialog_summarizer/model")


def get_summary(context_tmp):
    print(context_tmp)
    info = ""
    for persona in context_tmp:
        info += persona
    summary_agent.reset()
    # Model actually witnesses the human's text
    summary_agent.observe({'text': info, 'episode_done': False})
    response = summary_agent.act()
    return response['text'].split(".")


"""
RAG Model
"""
from parlai.core.agents import create_agent_from_model_file

rag_agent = create_agent_from_model_file(model_file="zoo:hallucination/bart_rag_token/model",
                                         opt_overrides={'rag_model_type': 'token', 'rag_retriever_type': 'dpr',
                                                        'n_docs': 10,
                                                        'generation_model': 'bart',
                                                        'path_to_index': 'deploy_doge_1/rag/embeddings/embeddings',
                                                        'indexer_type': 'compressed',
                                                        'compressed_indexer_factory': 'HNSW32',
                                                        'path_to_dpr_passages': 'deploy_doge_1/rag/dog_data/dog_index.csv',
                                                        'compressed_indexer_gpu_train': False})

import string


def get_sentence_list(sentence):
    sentence_list = []
    for i in sentence.split(","):
        for il in i.split("!"):
            for ill in il.split("?"):
                for illl in ill.split("."):
                    sentence_list.append(illl)
    return sentence_list


def delete_first(sentence, context):
    same_number = 0
    sentence_list = get_sentence_list(sentence)
    skip = False
    if len(context) <= 5:
        return ""
    if len(sentence_list[0]) <= 5:
        sentence_list = sentence_list[1:]
    for i in sentence_list[0].split():
        if i in context:
            same_number += 1
    if same_number >= (len(sentence_list[0].split()) // 2):
        return sentence_list[0]
    else:
        return ""


def find_rag_open(context):
    rag_agent.reset()
    # Model actually witnesses the human's text
    rag_agent.observe({'text': process_input(context[-1:]), 'episode_done': False})
    response = rag_agent.act()
    print("Origin rag", response['text'])
    response_tmp = ""
    response_list_tmp = []
    if response['text'].split(".")[-1] != '':
        response_list_tmp = response['text'].split(".")
    else:
        response_list_tmp = response['text'].split(".")[:-1]
    for i in response_list_tmp:
        if "?" in i:
            asked = question_asked(i)
            if asked == "False":
                response_tmp += i.strip() + ". "
        else:
            response_tmp += i.strip() + ". "
    sentence_list = delete_first(response['text'], context[-1])
    index = response_tmp.index(sentence_list)
    if sentence_list != "":
        if index != 0:
            response_tmp = response_tmp[:index] + response_tmp[index + len(sentence_list) + 1:]
        else:
            response_tmp = response_tmp[index + len(sentence_list) + 2:]
            # for i in response['text'].split("."):
    #     if i not in context[-2]:
    #         response_tmp += i + '. '
    response_tmp = edit_identity(response_tmp)
    if len(response_tmp) <= 5:
        response_tmp = "yeah"
    return response_tmp.replace("you're", "i'm").replace(" /", ",")


"""
Rule Based Model
"""
import json
from transformers import TextClassificationPipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

with open('deploy_doge_1/ruleBase/intents.json', 'r') as json_data:
    intents_result = json.load(json_data)

tokenizer_rule = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model_rule = AutoModelForSequenceClassification.from_pretrained("deploy_doge_1/ruleBase/results/checkpoint-4500",
                                                                num_labels=21)
pipe_rule = TextClassificationPipeline(model=model_rule, tokenizer=tokenizer_rule, return_all_scores=True)
tags_rule = {'Name': 0, 'Language': 1, 'Gender': 2, 'Identity': 3, 'Age': 4, 'Hobbies': 5, 'Friends': 6,
             'Favorites_Color': 7, 'Favorites_Food': 8, 'Worst_Food': 9,
             'Hometown': 10, 'Superpower': 11, 'Dislikes and Enemies': 12, 'Limits': 13, 'Work': 14, 'Life': 15,
             'Family': 16, 'Jokes': 17, 'Music': 18, 'Thanks': 19, 'Unknown': 20}


## rule-base model generate response
def find_response_rule(context_temp, tags, intents):
    result = pipe_rule(context_temp)
    all_score = []
    for r in result[0]:
        all_score.append(r['score'])
    final_result = max(all_score)
    final_tag_index = all_score.index(final_result)
    final_tag = str(list(tags.keys())[final_tag_index])
    if final_result > .90 and final_tag != "Unknown":
        for intent in intents['intents']:
            if final_tag == intent["tag"]:
                if final_tag == "Music":
                    return 'Music'
                if final_tag == "Jokes":
                    return "Jokes"
                respond = random.choice(intent['responses'])
                return respond
    return "Unknown"


"""
Fact, Life or Rag Rank
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer_fact_life_rank = AutoTokenizer.from_pretrained("distilbert-base-uncased")
from transformers import TextClassificationPipeline

model_fact_life_rank = AutoModelForSequenceClassification.from_pretrained(
    "deploy_doge_1/rank/results/checkpoint-2500", num_labels=2)

pipe = TextClassificationPipeline(model=model_fact_life_rank, tokenizer=tokenizer_fact_life_rank,
                                  return_all_scores=True)
tags = {'Fact': 0, 'Life': 1, 'Rag': 2}


## fact-life rank model generate response
def fact_life_rank_response(tags, query):
    result = pipe(query)
    final = []
    for r in result[0]:
        final.append(r['score'])
    final_result = final.index(max(final))
    return list(tags.keys())[final_result]


"""
Respond Rank
"""
from deploy_doge_1.acquire.acquire_deploy import NeuralNetwork
from deploy_doge_1.acquire.config import args as args_acquire
import random

#
model_acquire = NeuralNetwork(args=args_acquire)
model_acquire.load_model(args_acquire.save_path)
data_acquire = open("deploy_doge_1/acquire/dogs_result", 'r', encoding="utf-8").readlines()


## rank model generate response
def rank_response(context_temp, data, total):
    result_acquire = model_acquire.predict_rank('', context_temp, data, total)
    final = result_acquire[0]
    final = data[int(final)]
    return final


"""
Deploy Doge Chatbot
"""
import string

bot_name = "Doge"

app = Flask(__name__)

context = []
all_jokes = {"What breed of dog goes after anything that is red?": "A Bulldog.",
             "When you cross an aggressive dog with a computer, what do you get?": "A lot of bites.",
             "Why are dogs' barks so loud?": "They have built-in sub-woofers.",
             "Why does a noisy yappy dog resembles a tree?": "It's because they both have a lot of bark.",
             "What do you get when you cross a dog and a lion?": "You're not going to get any mail, that's for sure.",
             "Whenever I take my dog to the park, the ducks always try to bite him.": "I guess it makes sense, since he's pure bread."}

all_music = ["I'm a dog, you're a dog, everybody do the dog",
             "Who let the dogs out? Who, who, who, who, who? Who let the dogs out? Who, who, who, who, who?",
             "Well, the party was nice, the party was pumpin' Yippie yi yo And everybody havin' a ball Yippie yi yo",
             "Scratchin' in the sunshine, diggin' in the dirt Beggin' for a doggie treat, 'Get off the furniture' [Bark!]"]

all_dog_start = []
with open('deploy_doge_1/acquire/dogs_start', 'r') as file:
    for line in file:
        all_dog_start.append(line)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'
# Initialize the database
db = SQLAlchemy(app)


# Create db model
class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sentence = db.Column(db.String(2000), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow())
    history = db.relationship('Chat')


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/history")
def all_user_page():
    users = User.query.all()
    return render_template('all_user.html', users=users)


@app.route("/history/<string:user_id>")
def history(user_id):
    user = User.query.get_or_404(user_id)
    return render_template('history.html', chat=user.history)


@app.route("/get")
def get_bot_response(context=context, all_user=all_user):
    global result_final
    if User.query.filter_by(name=str(request.remote_addr)).first() is not None:
        cur_user = User.query.filter_by(name=str(request.remote_addr)).first()
    else:
        cur_user = User(name=str(request.remote_addr))
        db.session.add(cur_user)
        db.session.commit()
    sentence = request.args.get("msg")
    if cur_user is not None and sentence is not None:
        tmp_history = Chat(sentence="human:   " + sentence, user_id=cur_user.id)
        db.session.add(tmp_history)
        db.session.commit()
    if context == [] or sentence is None:
        dog_start = random.choice(all_dog_start).replace("\n", "")
        for i in range(len(context)):
            context.remove(context[0])
        context.append("HEY!")
        context.append(dog_start)
        tmp_history = Chat(sentence="doge:   " + dog_start, user_id=cur_user.id)
        db.session.add(tmp_history)
        db.session.commit()
        return dog_start
    else:
        context.append(sentence)
        all_user.append(sentence)
        if len(context) >= 30:
            context.remove(context[0])
            context.remove(context[0])
        result_acquired = find_response_acquire(context[-2:])
        # rule model
        result_rule = find_response_rule("".join([char for char in sentence if char not in string.punctuation])
                                         , tags_rule, intents_result)
        if context[-2] in all_jokes:
            pre_joke = context[-2]
            context.append(all_jokes[pre_joke])
            tmp_history = Chat(sentence="doge:   " + all_jokes[pre_joke], user_id=cur_user.id)
            db.session.add(tmp_history)
            db.session.commit()
            return {'result_final': [all_jokes[pre_joke]], 'number': 1}
        if result_rule != "Unknown":
            if result_rule == "Music":
                result_final = random.choice(all_music)
                if context[-3] != "Here you go!":
                    context.append("Here you go!")
                    context.append(result_final)
                    tmp_history = Chat(sentence="doge:   " + result_final, user_id=cur_user.id)
                    db.session.add(tmp_history)
                    db.session.commit()
                    return {'result_final': ["Here you go!", '[Music] ' + result_final], 'number': 2}
            if result_rule == "Jokes":
                result_final = random.choice(list(all_jokes.keys()))
                context.append("Tell you something funny:")
                context.append(result_final)
                tmp_history = Chat(sentence="doge:   " + result_final, user_id=cur_user.id)
                db.session.add(tmp_history)
                db.session.commit()
                return {'result_final': ["Tell you something funny:", result_final], 'number': 2}
            result_final = result_rule
            if result_final in context:
                result_final = find_rag_open(context[-1:])
        else:
            skip = False
            for word in ["it", "its", "that", "those", "he", "his", "him", "she", "her", "they", "those", "them", "did",
                         "was", "were", "me", "mine", "us"]:
                if word in sentence:
                    if type(result_acquired) != list:
                        result_acquired_tmp = [result_acquired, "I am doge", "I am a dog"]
                    else:
                        result_acquired_tmp = result_acquired + ["I am doge", "I am a dog"]
                    result_open = find_response_open(context, result_acquired_tmp, get_summary(all_user),
                                                     blender_msc_agent)
                    result_final = result_open
                    skip = True
                    print("fact")
                if skip:
                    break
            if len(sentence) <= 10 and not skip:
                result_open = find_response_open(context, result_acquired, get_summary(all_user),
                                                 blender_msc_agent)
                result_final = result_open
                print("fact")
            if not skip:
                if type(result_acquired) != list and result_acquired != context[-2]:
                    result_final = result_acquired
                else:
                    fact_life_rank = fact_life_rank_response(tags, sentence)
                    print(fact_life_rank)
                    if fact_life_rank == "Life":
                        result_acquire = find_multiple_response_acquired(context, data_acquire)
                        result_open = find_response_open(context, result_acquired, get_summary(all_user),
                                                         blender_msc_agent)
                        result_rag = find_rag_open(context[-1:])
                        result_acquire.append(result_open)
                        result_acquire.append(result_rag)
                        result_final = rank_response(context[-3:], result_acquired, 6)
                        print(result_acquire)
                    elif fact_life_rank == "Rag":
                        result_rag = find_rag_open(context[-1:])
                        result_final = result_rag
                    else:
                        result_open = find_response_open(context, result_acquired, get_summary(all_user),
                                                         blender_msc_agent)
                        result_final = result_open
        if result_final in context:
            result_final = find_response_open(context, result_final, get_summary(all_user), blender_msc_agent)
        context.append(result_final)
        tmp_history = Chat(sentence="doge:   " + result_final, user_id=cur_user.id)
        db.session.add(tmp_history)
        db.session.commit()
        print("HERE IS THE HISTORY", context)
        all_result = result_final.split(".")
        if all_result[-1] == "":
            return {'result_final': all_result[:-1], 'number': len(all_result[:-1])}
        else:
            return {'result_final': all_result, 'number': len(all_result)}


if __name__ == "__main__":
    app.run('0.0.0.0', port='8080')
