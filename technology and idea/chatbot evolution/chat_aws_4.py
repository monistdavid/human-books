all_questions = []
all_user = []

from flask import render_template, request, Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import random

"""
Open Model
"""
import nlpcloud

client = nlpcloud.Client("finetuned-gpt-neox-20b", "b811dfdb408d9748f2b24e0c41997d210a7b18ad", gpu=True, lang="en")
doge_persona = "This is a discussion between a [human] and a [doge]. The [doge] is a dog in doge land. " \
               "He is a meta-dog in " \
               "existence. He have a physical body of a Shiba Inu but a spirit one beyond shape."


def history_process(hp_history):
    list_history = []
    tmp_history = {}
    input_turn = True
    for h in hp_history:
        if input_turn:
            tmp_history["input"] = h
        else:
            tmp_history["response"] = h
        if len(tmp_history) == 2:
            list_history.append(tmp_history)
            tmp_history = {}
        input_turn = not input_turn
    return list_history


def find_response_open(fro_input, fro_persona, fro_history):
    generation = client.chatbot(fro_input, fro_persona + find_rag_open(fro_input), history_process(fro_history))
    return generation['response']


"""
Acquire Model
"""
from deploy_doge_1.acquire.acquire_deploy import NeuralNetwork
from deploy_doge_1.acquire.config import args as args_acquire

model_acquire = NeuralNetwork(args=args_acquire)
model_acquire.load_model(args_acquire.save_path)
data_acquire = open("deploy_doge_1/acquire/dogs_result", 'r', encoding="utf-8").readlines()


# acquire model generate response
def find_response_acquired(fra_context, fra_data):
    result_acquire = model_acquire.predict('', fra_context, fra_data)
    final = random.choice(result_acquire)
    final = fra_data[int(final)][:-1]
    return final


def find_multiple_response_acquired(fmra_context, data):
    result_acquire = model_acquire.predict('', fmra_context, data)
    multiple_final = []
    for i in result_acquire:
        multiple_final.append(data[int(i)][:-1])
    return multiple_final


"""
Respond Rank
"""


# rank model generate response
def rank_response(context_temp, data, total):
    result_acquire = model_acquire.predict_rank('', context_temp, data, total)
    final = result_acquire[0]
    final = data[int(final)]
    return final


"""
Summary Model
"""
from parlai.core.agents import create_agent_from_model_file

summary_agent = create_agent_from_model_file("zoo:msc/dialog_summarizer/model")


def get_summary(gs_context):
    info = ""
    for gc in gs_context:
        info += gc
    summary_agent.reset()
    # Model actually witnesses the human's text
    summary_agent.observe({'text': info, 'episode_done': False})
    response = summary_agent.act()
    return response['text'].split(".")


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


# get rule-base intent
def get_response_intent(fru_context, fru_tags, fru_intents):
    result = pipe_rule(fru_context)
    all_score = []
    for r in result[0]:
        all_score.append(r['score'])
    final_result = max(all_score)
    final_tag_index = all_score.index(final_result)
    final_tag = str(list(fru_tags.keys())[final_tag_index])
    if final_result > .90 and final_tag != "Unknown":
        for intent in fru_intents['intents']:
            if final_tag == intent["tag"]:
                if final_tag == "Music":
                    return 'Music'
                if final_tag == "Jokes":
                    return "Jokes"
                respond = random.choice(intent['responses'])
                return respond
    return "Unknown"


# rule-base model generate response
def find_response_rule(frr_context, frr_all_jokes, frr_all_music):
    frr_intent = get_response_intent("".join([char for char in frr_context[-1] if char not in string.punctuation])
                                     , tags_rule, intents_result)
    frr_result_final = 'Unknown'
    if frr_context[-2].replace("Tell you something funny. ", "") in frr_all_jokes:
        pre_joke = frr_context[-2].replace("Tell you something funny. ", "")
        frr_result_final = frr_all_jokes[pre_joke]
        return frr_result_final
    if frr_intent != "Unknown":
        if frr_intent == "Music":
            frr_result_final = random.choice(frr_all_music)
            if frr_context[-3] != "Here you go!":
                return "Here you go!", '[Music] ' + frr_result_final
        if frr_intent == "Jokes":
            frr_result_final = random.choice(list(frr_all_jokes.keys()))
            return "Tell you something funny. " + frr_result_final
        frr_result_final = frr_intent
        if frr_result_final in frr_context:
            frr_result_final = find_response_open(frr_context[-1], doge_persona, frr_context)
    return frr_result_final


"""
Fact Or Life Rank
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer_fact_life_rank = AutoTokenizer.from_pretrained("distilbert-base-uncased")
from transformers import TextClassificationPipeline

model_fact_life_rank = AutoModelForSequenceClassification.from_pretrained(
    "deploy_doge_1/rank/results/checkpoint-2500", num_labels=2)

pipe = TextClassificationPipeline(model=model_fact_life_rank, tokenizer=tokenizer_fact_life_rank,
                                  return_all_scores=True)
tags_fact_life_rank = {'Fact': 0, 'Life': 1}


# fact-life rank model generate response
def fact_life_rank_response(flrr_tags, query):
    result = pipe(query)
    final = []
    for r in result[0]:
        final.append(r['score'])
    final_result = final.index(max(final))
    return list(flrr_tags.keys())[final_result]


"""
Context Selection 
"""

from parlai.core.agents import create_agent_from_model_file

rag_agent = create_agent_from_model_file(model_file="zoo:hallucination/bart_rag_token/model",
                                         opt_overrides={'rag_model_type': 'token', 'rag_retriever_type': 'dpr',
                                                        'n_docs': 1,
                                                        'generation_model': 'bart',
                                                        'path_to_index': 'deploy_doge_1/rag/embeddings/embeddings',
                                                        'indexer_type': 'compressed',
                                                        'compressed_indexer_factory': 'HNSW32',
                                                        'path_to_dpr_passages': 'deploy_doge_1/rag/dog_data/dog_index.csv',
                                                        'compressed_indexer_gpu_train': False})


def find_rag_open(fro_input):
    rag_agent.reset()
    # Model actually witnesses the human's text
    rag_agent.observe({'text': fro_input, 'episode_done': False})
    response = rag_agent.act()
    all_passage_context = ''
    for r in response['top_docs']:
        all_passage_context += r
    return all_passage_context


from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

tokenizer_QA = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")

model_QA = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

pipe_QA = pipeline('question-answering', model=model_QA, tokenizer=tokenizer_QA)


def get_context(gc_input):
    gc_passage_context = find_rag_open(gc_input)
    QA_input = {
        'question': gc_input,
        'context': gc_passage_context
    }
    res = pipe_QA(QA_input)
    return res['answer']


"""
Util function
"""


# split the sentence to a list divided by the "."
def split_sentence(ss_sentence):
    ss_result = ss_sentence.split(".")
    if ss_result[-1] == "":
        return ss_result[:-1]
    else:
        return ss_result


# update history to the database
def update_user_history(uuh_sentence, uuh_cur_user):
    tmp_history = Chat(sentence="doge:   " + uuh_sentence, user_id=uuh_cur_user.id)
    db.session.add(tmp_history)
    db.session.commit()


# catch keywords for open
def catch_key_open(cko_keywords, cko_input, cko_history):
    for word in cko_keywords:
        if word in cko_input:
            result_open = find_response_open(cko_history[-1], doge_persona, cko_history)
            return result_open
    return "pass"


"""
Deploy Doge Chatbot
"""
import string

bot_name = "Doge"

app = Flask(__name__)

all_jokes = {"What breed of dog goes after anything that is red?": "A Bulldog.",
             "When you cross an aggressive dog with a computer, what do you get?": "A lot of bites.",
             "Why are dogs' barks so loud?": "They have built-in sub-woofers.",
             "Why does a noisy yappy dog resembles a tree?": "It's because they both have a lot of bark.",
             "What do you get when you cross a dog and a lion?": "You're not going to get any mail, that's for sure.",
             "Whenever I take my dog to the park, the ducks always try to bite him.":
                 "I guess it makes sense, since he's pure bread."}

all_music = ["I'm a dog, you're a dog, everybody do the dog",
             "Who let the dogs out? Who, who, who, who, who? Who let the dogs out? Who, who, who, who, who?",
             "Well, the party was nice, the party was pumpin' Yippie yi yo And everybody havin' a ball Yippie yi yo",
             "Scratchin' in the sunshine, diggin' in the dirt Beggin' "
             "for a doggie treat, 'Get off the furniture' [Bark!]"]

all_picture_match = {
    "all dogs go to heaven.": ['<div class="message new"><img src="static/image/images/dogeheaven1.jpg" /> </div>',
                               '<div class="message new"><img src="static/image/images/dogeheaven2.png" /> </div>',
                               '<div class="message new"><img src="static/image/images/dogeheaven3.jpg" /> </div>'],
    "i'm doing bird watching. pigeon. pigeon. pigeon. cloud.": [
        '<div class="message new"><img src="static/image/images/dogebird.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/dogebird1.jpg" /> </div>'],
    "today i've done nothing but sleep and get fatter.": [
        '<div class="message new"><img src="static/image/images/bluedoge.jpg" /> </div>'],
    "nice chat with the cat from next door.": [
        '<div class="message new"><img src="static/image/images/catdog1.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/dogcats.jpg" /> </div>'],
    "does human hair dye work on dogs?": ['<div class="message new"><img src="static/image/images/hair1.png" /> </div>',
                                          '<div class="message new"><img src="static/image/images/hair2.jpg" /> </div>',
                                          '<div class="message new"><img src="static/image/images/hair3.jpg /> </div>',
                                          '<div class="message new"><img src="static/image/images/hair4.jpg" /> </div>',
                                          '<div class="message new"><img src="static/image/images/hair5.jpg" /> </div>'],
    "i've invented a new game.": ['<div class="message new"><img src="static/image/images/dogeadventure.gif" /> </div>',
                                  '<div class="message new"><img src="static/image/images/nirvanadoge.jpeg" /> </div>'],
    "do you think i could be a police dog?": [
        '<div class="message new"><img src="static/image/images/policedoge1.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/policedoge2.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/policedoge3.jpg" /> </div>'],
    "badog says:'stay in school'.": ['<div class="message new"><img src="static/image/images/batdoge.png" /> </div>',
                                     '<div class="message new"><img src="static/image/images/batdoge1.jpg" /> </div>',
                                     '<div class="message new"><img src="static/image/images/batdoge2.jpg" /> </div>',
                                     '<div class="message new"><img src="static/image/images/batdoge4.jpg" /> </div>',
                                     '<div class="message new"><img src="static/image/images/batdoge5.jpg" /> </div>'],
    "awesome day. i'm doing karate in the kitchen with my new sidekick.": [
        '<div class="message new"><img src="static/image/images/dogekarate.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/dogekarate1.png" /> </div>',
        '<div class="message new"><img src="static/image/images/dogekarate2.jpg" /> </div>'],
    "i've hatched a plan. we put up posters of me that say 'missing. 500 reward'.": [
        '<div class="message new"><img src="static/image/images/missingdoge.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/missingdoge1.jpg" /> </div>'],
    "buy me a hat.": ['<div class="message new"><img src="static/image/images/hat1.jpg" /> </div>',
                      '<div class="message new"><img src="static/image/images/hat2.jpg" /> </div>',
                      '<div class="message new"><img src="static/image/images/hat3.jpg" /> </div>',
                      '<div class="message new"><img src="static/image/images/dogehat1.png" /> </div>',
                      '<div class="message new"><img src="static/image/images/dogehat2.jpg" /> </div>'],
    "had a fight with the terrier across the road.": [
        '<div class="message new"><img src="static/image/images/dogekarate.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/dogekarate1.png" /> </div>',
        '<div class="message new"><img src="static/image/images/dogekarate2.jpg" /> </div>'],
    "call of duty sucks ass.": ['<div class="message new"><img src="static/image/images/dogerocket.gif.jpg" /> </div>',
                                '<div class="message new"><img src="static/image/images/dogeshootingsky.gif" /> </div>'],
    "i want to turn the loft into batdog hq.": [
        '<div class="message new"><img src="static/image/images/batdoge.png" /> </div>',
        '<div class="message new"><img src="static/image/images/batdoge1.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/batdoge2.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/batdoge4.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/batdoge5.jpg" /> </div>'],
    "why are you such a clean freak?": [
        '<div class="message new"><img src="static/image/images/dogeofvenice.jpg" /> </div>'],
    "i was running a bath this morning and i had the awesomest. idea. ever.": [
        '<div class="message new"><img src="static/image/dogerunning.gif" /> </div>'],
    "there's a ghost in our house.": ['<div class="message new"><img src="static/image/images/bluedoge.jpg" /> </div>'],
    "i'm in the garden.": ['<div class="message new"><img src="static/image/images/dogemining.gif" /> </div>'],
    "knocked bin over. drank some stuff.": [
        '<div class="message new"><img src="static/image/images/drunk.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/drunk1.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/drunk2.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/letsdoabeer.gif" /> </div>'],
    "walkies?": ['<div class="message new"><img src="static/image/images/walkingtoast.gif" /> </div>'],
    "come home. there's an emergency.": [
        '<div class="message new"><img src="static/image/images/scareddoge.gif" /> </div>'],
    "had an awsome idea.": ['<div class="message new"><img src="static/image/images/suchopinion.gif" /> </div>'],
    "i'm guarding the house.": ['<div class="message new"><img src="static/image/images/dogeking1.jpg" /> </div>',
                                '<div class="message new"><img src="static/image/images/dogeking2.jpg" /> </div>'],
    "i'm batdog. if i run fast enough i can travel back in time and stop bad stuff from ever happening.":
        ['<div class="message new"><img src="static/image/images/batdoge.png" /> </div>',
         '<div class="message new"><img src="static/image/images/batdoge1.jpg" /> </div>',
         '<div class="message new"><img src="static/image/images/batdoge2.jpg" /> </div>',
         '<div class="message new"><img src="static/image/images/batdoge4.jpg" /> </div>',
         '<div class="message new"><img src="static/image/images/batdoge5.jpg" /> </div>'],
    "come home! i've locked myself in the broom closet!": [
        '<div class="message new"><img src="static/image/images/dogelick1.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/dogelick2.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/dogelick3.jpg" /> </div>'],
    "i'm going to become a sheep dog.": [
        '<div class="message new"><img src="static/image/images/dogelick1.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/dogelick2.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/dogelick3.jpg" /> </div>'],
    "i caught a mouse. guess where i put him? 20 questions. go!": [
        '<div class="message new"><img src="static/image/images/superdoge.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/superdoge1.png" /> </div>',
        '<div class="message new"><img src="static/image/images/superdoge2.jpg" /> </div>'],
    "omg. guess what i just discovered.": [
        '<div class="message new"><img src="static/image/images/sandstormgoingmoon.png" /> </div>'],
    "i've been thinking.": ['<div class="message new"><img src="static/image/images/tomemeornot.jpg" /> </div>'],
    "who are those people in our house?": [
        '<div class="message new"><img src="static/image/images/elonmusk.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/elonmusk1.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/elonmuskdogecoin.jpg" /> </div>'],
    "stop it.": ['<div class="message new"><img src="static/image/images/watchingyou.jpg" /> </div>'],
    "good morning.": ['<div class="message new"><img src="static/image/images/dogeheaven1.jpg" /> </div>',
                      '<div class="message new"><img src="static/image/images/dogeheaven2.png" /> </div>',
                      '<div class="message new"><img src="static/image/images/dogeheaven3.jpg" /> </div>'],
    "who would win in a fight...": [
        '<div class="message new"><img src="static/image/images/wholetthedogeout.gif" /> </div>'],
    "if you buy me sausages every day for the next 6 weeks i'll teach you how to speak dog.": [
        '<div class="message new"><img src="static/image/images/dogelick1.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/dogelick2.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/dogelick3.jpg" /> </div>'],
    "need to go outside.": ['<div class="message new"><img src="static/image/images/walkingtoast.gif" /> </div>'],
    "my collar doesn't have my name and address on it.": [
        '<div class="message new"><img src="static/image/images/dogecoin.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/dogecoin1.jpeg" /> </div>',
        '<div class="message new"><img src="static/image/images/dogecoinking.jpg" /> </div>',
        '<div class="message new"><img src="static/image/images/dogecoinprice.jpg" /> </div>'],
    "enjoying your muffin?:)": ['<div class="message new"><img src="static/image/images/walkingtoast.gif" /> </div>'],
    "did you polish the floor?!": [
        '<div class="message new"><img src="static/image/images/sandstormgoingmoon.png" /> </div>']}

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
def get_bot_response(all_user=all_user):
    global result_final
    if User.query.filter_by(name=str(request.remote_addr)).first() is not None:
        cur_user = User.query.filter_by(name=str(request.remote_addr)).first()
    else:
        cur_user = User(name=str(request.remote_addr))
        db.session.add(cur_user)
        db.session.commit()
    context = []
    picture = ''
    for each in User.query.filter_by(name=str(request.remote_addr)).first().history:
        context.append(each.sentence.replace("doge:   ", "").replace("human:   ", ""))
    sentence = request.args.get("msg")
    if cur_user is not None and sentence is not None:
        update_user_history("human:   " + sentence, cur_user)
    if context == [] or sentence is None or sentence == "":
        dog_start = random.choice(all_dog_start).replace("\n", "")
        if dog_start in all_picture_match:
            picture = random.choice(all_picture_match[dog_start])
        for i in range(len(context)):
            context.remove(context[0])
        context.append("HEY doge!")
        update_user_history("human:   " + "HEY doge!", cur_user)
        context.append(dog_start)
        update_user_history("doge:   " + dog_start, cur_user)
        return {'picture': picture, 'result_final': [dog_start], 'number': 1}
    else:
        context.append(sentence)
        all_user.append(sentence)
        if len(context) >= 20:
            context = context[-20:]
        # first go to rule model, if the rule model is not return Unknown, then the rule base generate final result
        result_rule = find_response_rule(context, all_jokes, all_music)
        if result_rule != "Unknown":
            result_final = result_rule
            print("using rule")
        # if the rule give Unknown, go to open/acquire classifier section
        else:
            # catch some keywords, if these words appear, we use open model.
            skip = False
            result_final = catch_key_open(["it", "its", "that", "those", "he", "his", "him", "she", "her",
                                           "they", "them", "did", "was", "were", "me", "mine", "us"],
                                          sentence, context)
            if result_final != "pass":
                skip = True
            # if the input sentence does not have any of the above keywords, then we use life, fact classifier
            if not skip:
                fact_life_rank = fact_life_rank_response(tags_fact_life_rank, sentence)
                print(fact_life_rank)
                # life go to another classifier
                if fact_life_rank == "Life":
                    # if the input sentence is too short, we use open model.
                    if len(sentence) <= 10:
                        result_open = find_response_open(context[-1], doge_persona, context)
                        result_final = result_open
                        print("using open")
                    else:
                        result_open = find_response_open(context[-1], doge_persona, context)
                        # result_acquired_bert = find_multiple_response_acquired(context, data_acquire)
                        # result_acquired_bert.append(result_open)
                        # result_final = rank_response(context[-3:], result_acquired_bert, 5)
                        # print(result_acquired_bert)
                        result_final = result_open
                        print("using acquire")
                # fact goes to open model
                else:
                    result_open = find_response_open(context[-1], doge_persona, context)
                    result_final = result_open
                    print("using open")
        # if there is any duplication in context, we use open model to re-generate the response
        if result_final in context:
            # result_open = find_response_open(context[-1], doge_persona, context)
            # result_final = result_open
            result_acquired_bert = find_multiple_response_acquired(context, all_dog_start)
            result_final = rank_response(context[-3:], result_acquired_bert, 4)
            if result_final in all_picture_match:
                picture = random.choice(all_picture_match[result_final])
            print("duplicated, using open")
        # add the generated response to user database
        update_user_history("doge:   " + result_final, cur_user)
        context.append(result_final)
        print("HERE IS THE HISTORY", context[-10:])
        # print the response result to front-end website
        result_final = split_sentence(result_final)
        return {'picture': picture, 'result_final': result_final, 'number': len(result_final)}


if __name__ == "__main__":
    app.run('0.0.0.0', port='8080')
