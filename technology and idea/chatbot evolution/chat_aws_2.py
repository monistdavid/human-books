"""
open:  zoo:blenderbot2/blenderbot2_3B/model with memory_only config
acquire: BERT-FP, 4 results from more than 1000 response candidates | rerank with open and rag model to obtain the best result
rag: custom compressed index, zoo:hallucination/bart_rag_sequence/model
rule: 22 intents
fact_life rank: classifier trained on 600 life questions with 6000 fact questions
question_statement rank: classifier shahrukhx01/question-vs-statement-classifier to see when to use rag or acquire
"""


# """
# Open Model
# """
# from flask import render_template, request, Flask
# from flask_sqlalchemy import SQLAlchemy
# from parlai.core.agents import create_agent_from_model_file
# from datetime import datetime
#
# # blender_agent = create_agent_from_model_file("zoo:msc/blender3B_1024/model")
# import os
# import multiprocess
# import shlex
# import subprocess
# import threading
# import time
#
# # Using port 8080 doesn't work on Colab
# HOST = "0.0.0.0:2000"
# # Change the following as neede
# PATH_TO_SEARCH_SERVER = "search_server.py"
#
# assert os.path.exists(PATH_TO_SEARCH_SERVER), (
#     f"Incorrect path {PATH_TO_SEARCH_SERVER}"
# )
#
# command = ["python3.7", "-u", shlex.quote(PATH_TO_SEARCH_SERVER),
#            "serve", "--host", HOST]
# command_str = " ".join(command)
# p = subprocess.Popen(
#     command,
#     stderr=subprocess.STDOUT,
#     stdout=subprocess.PIPE,
# )
#
# # Wait a bit before the next cell to let a lot of the potential errors happen.
# time.sleep(1)
#
# # Test If the server crashed.
# # You can rerun this cell to check if the server crashed.
# if p.poll() is not None:
#     print(p.communicate()[0].decode())
#     # If this says that the adress is already used, then it is very likely that
#     # you already started a server with a different Process object than p.
# print("______________________________________________________________________________")
# # blender_agent = create_agent_from_model_file(model_file="zoo:blenderbot2/blenderbot2_3B/model",
# #                                              opt_overrides={"search_server": HOST,
# #                                                             "knowledge_access_method": "classify",
# #                                                             "loglevel": "debug",
# #                                                             "query_generator_model_file": "zoo:sea/bart_sq_gen/model",
# #                                                             "doc_chunk_split_mode": False})
#
#
# blender_memory_agent = create_agent_from_model_file(model_file="zoo:blenderbot2/blenderbot2_3B/model",
#                                                     opt_overrides={"search_server": HOST,
#                                                                    "knowledge_access_method": "memory_only",
#                                                                    "loglevel": "debug",
#                                                                    "query_generator_model_file": "zoo:sea/bart_sq_gen/model",
#                                                                    "doc_chunk_split_mode": False})
#
# #
# # blender_memory_agent = create_agent_from_model_file(model_file="zoo:blenderbot2/blenderbot2_3B/model",
# #                                                     opt_overrides={'path_to_index': 'DPR/embeddings/HNSW32',
# #                                                                    'indexer_type': 'compressed',
# #                                                                    'compressed_indexer_factory': 'HNSW32',
# #                                                                    'path_to_dpr_passages': 'DPR/dog_data/dog_index.csv',
# #                                                                    "search_server": HOST,
# #                                                                    "knowledge_access_method": "classify",
# #                                                                    "loglevel": "debug",
# #                                                                    "query_generator_model_file": "zoo:sea/bart_sq_gen/model",
# #                                                                    'doc_chunk_split_mode': False})
#
#
# # zoo:msc/blender3B_1024/model
# # zoo:blenderbot2/blenderbot2_400M/model
# # zoo:blender/blender_3B/model
# # zoo:msc/blender3B_1024/model
# # zoo:blender/blender_90M/model
# def find_response_open(context_temp, persona, agent):
#     agent.reset()
#     info = add_persona(persona)
#     for i in context_temp:
#         info.append(i)
#     # Model actually witnesses the human's text
#     agent.observe({'text': process_input(info), 'episode_done': False})
#     # print(f"You said: {first_turn}")
#     # model produces a response
#     response = agent.act()
#     return response['text'].replace("_POTENTIALLY_UNSAFE__", "").replace("_PAUSE__.", "")
#
#
# def add_persona(context_temp):
#     persona = []
#     for i in context_temp:
#         persona.append('your persona: ' + i)
#     return persona
#
#
# def process_input(context_temp):
#     return "\n".join(context_temp)
#
#
# """
# Acquire Model
# """
# from deploy_doge_1.acquire.faiss_top_20.simpleredial.dataloader import *
# from deploy_doge_1.acquire.faiss_top_20.simpleredial.model.RepresentationModels import *
# from deploy_doge_1.acquire.faiss_top_20.simpleredial.config import *
# from deploy_doge_1.acquire.faiss_top_20.simpleredial.deploy import *
# from deploy_doge_1.acquire.faiss_top_20.simpleredial.es import *
#
# args_acquire_fast = vars(parser_args())
# data_args, inf_args, searcher, agent = prepare_inference(**args_acquire_fast)
# vocab_acquire_fast = BertTokenizerFast.from_pretrained('bert-base-uncased')
#
#
# from deploy_doge_1.acquire.acquire_deploy import NeuralNetwork
# from deploy_doge_1.acquire.config import args as args_acquire
# import random
#
# model_acquire = NeuralNetwork(args=args_acquire)
# model_acquire.load_model(args_acquire.save_path)
# data_acquire = open("deploy_doge_1/acquire/dogs_result", 'r', encoding="utf-8").readlines()
#
#
# ## acquire model generate response
# def find_response_acquired(context_temp, data):
#     result_acquire = model_acquire.predict('', context_temp, data)
#     final = random.choice(result_acquire)
#     final = data[int(final)][:-1]
#     return final
#
#
# def find_multiple_response_acquired(context_temp, data):
#     context_fast = context_temp.copy()
#     context_fast.append("hallo")
#     response_acquire_fast = main_recall(context_fast, data_args, inf_args, searcher, agent, vocab_acquire_fast,
#                                         **args_acquire_fast)
#     result_acquire = model_acquire.predict('', context_temp, response_acquire_fast)
#     multiple_final = []
#     for i in result_acquire:
#         multiple_final.append(data[int(i)][:-1])
#     return multiple_final
#
#
# """
# RAG Model
# """
# from parlai.core.agents import create_agent_from_model_file
#
# rag_agent = create_agent_from_model_file(model_file="zoo:hallucination/bart_rag_sequence/model",
#                                          opt_overrides={'rag_model_type': 'token', 'rag_retriever_type': 'dpr',
#                                                         'generation_model': 'bart',
#                                                         'path_to_index': 'deploy_doge_1/rag/embeddings/HNSW32',
#                                                         'indexer_type': 'compressed',
#                                                         'compressed_indexer_factory': 'HNSW32',
#                                                         'path_to_dpr_passages': 'deploy_doge_1/rag/dog_data/dog_index.csv',
#                                                         'compressed_indexer_gpu_train': False})
#
#
# def find_rag_open(context):
#     rag_agent.reset()
#     # Model actually witnesses the human's text
#     rag_agent.observe({'text': process_input(context), 'episode_done': False})
#     # print(f"You said: {first_turn}")
#     # model produces a response
#     response = rag_agent.act()
#     return response['text']
#
#
# """
# Rule Based Model
# """
# import json
# import random
# from transformers import TextClassificationPipeline
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# with open('deploy_doge_1/ruleBase/intents.json', 'r') as json_data:
#     intents_result = json.load(json_data)
#
# tokenizer_rule = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# model_rule = AutoModelForSequenceClassification.from_pretrained("deploy_doge_1/ruleBase/results/checkpoint-4500",
#                                                                 num_labels=22)
# pipe_rule = TextClassificationPipeline(model=model_rule, tokenizer=tokenizer_rule, return_all_scores=True)
# tags_rule = {'Name': 0, 'Language': 1, 'Gender': 2, 'Breed': 3, 'Age': 4, 'Identity': 5, 'Hobbies': 6, 'Friends': 7,
#              'Favorites_Color': 8, 'Favorites_Food': 9, 'Worst_Food': 10,
#              'Hometown': 11, 'Superpower': 12, 'Dislikes and Enemies': 13, 'Limits': 14, 'Work': 15, 'Life': 16,
#              'Family': 17, 'Jokes': 18, 'Music': 19, 'Thanks': 20, 'Unknown': 21}
#
#
# ## rule-base model generate response
# def find_response_rule(context_temp, tags, intents):
#     result = pipe_rule(context_temp)
#     all_score = []
#     for r in result[0]:
#         all_score.append(r['score'])
#     final_result = max(all_score)
#     final_tag_index = all_score.index(final_result)
#     final_tag = str(list(tags.keys())[final_tag_index])
#     if final_result > .90 and final_tag != "Unknown":
#         for intent in intents['intents']:
#             if final_tag == intent["tag"]:
#                 if final_tag == "Music":
#                     return 'Music'
#                 if final_tag == "Jokes":
#                     return "Jokes"
#                 respond = random.choice(intent['responses'])
#                 return respond
#     return "Unknown"
#
#
# """
# Fact Or Life Rank
# """
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# tokenizer_fact_life_rank = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# from transformers import TextClassificationPipeline
#
# model_fact_life_rank = AutoModelForSequenceClassification.from_pretrained(
#     "deploy_doge_1/rank/results/checkpoint-2000-people-fact-update-3-600-10-10", num_labels=2)
#
# pipe = TextClassificationPipeline(model=model_fact_life_rank, tokenizer=tokenizer_fact_life_rank,
#                                   return_all_scores=True)
# tags = {'Fact': 0, 'Life': 1}
#
#
# ## fact-life rank model generate response
# def fact_life_rank_response(tags, query):
#     result = pipe(query)
#     final = []
#     for r in result[0]:
#         final.append(r['score'])
#     final_result = final.index(max(final))
#     return list(tags.keys())[final_result]
#
#
# """
# Question or statement Rank
# """
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
#
# question_statement_tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/question-vs-statement-classifier")
#
# question_statement_model = AutoModelForSequenceClassification.from_pretrained(
#     "shahrukhx01/question-vs-statement-classifier")
#
# question_statement_classifier = TextClassificationPipeline(model=question_statement_model,
#                                                            tokenizer=question_statement_tokenizer,
#                                                            return_all_scores=True)
#
# question_statement_labels = ['natural', 'question']
#
#
# ## question statement rank model generate response
# def question_statement_response(tags, query):
#     result = question_statement_classifier(query)
#     final = []
#     for r in result[0]:
#         final.append(r['score'])
#     final_result = final.index(max(final))
#     return tags[final_result]
#
#
# """
# Respond Rank
# """
#
#
# ## rank model generate response
# def rank_response(context_temp, data):
#     result_acquire = model_acquire.predict_rank('', context_temp, data, 5)
#     final = result_acquire[0]
#     final = data[int(final)]
#     return final
#
#
# ## rank model generate response
# def rank_response_second(context_temp, data, number):
#     result_acquire = model_acquire.predict_rank('', context_temp, data, number)
#     final = result_acquire[0]
#     final = data[int(final)]
#     return final
#
#
# """
# Topic Change Detection
# """
# from torch.nn.functional import softmax
# from transformers import BertForNextSentencePrediction, BertTokenizer
#
#
# topic_change_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# topic_change_model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
#
# topic_change_classifier = TextClassificationPipeline(model=topic_change_model,
#                                                      tokenizer=topic_change_tokenizer,
#                                                      return_all_scores=True)
#
# topic_change_labels = ['same', 'different']
#
#
# ## topic change  model generate response
# def topic_change_response(tags, seq_A, seq_B):
#     encoded = topic_change_tokenizer.encode_plus(seq_A, text_pair=seq_B, return_tensors='pt')
#     seq_relationship_logits = topic_change_model(**encoded)[0]
#     probs = softmax(seq_relationship_logits, dim=1)
#     final = []
#     for r in probs[0]:
#         final.append(r)
#     final_result = final.index(max(final))
#     return tags[final_result]
#
#
# """
# Deploy Doge Chatbot
# """
# import string
#
# bot_name = "Doge"
#
# app = Flask(__name__)
#
# context = []
# all_jokes = {"What breed of dog goes after anything that is red?": "A Bulldog.",
#              "When you cross an aggressive dog with a computer, what do you get?": "A lot of bites.",
#              "Why are dogs' barks so loud?": "They have built-in sub-woofers.",
#              "Why does a noisy yappy dog resembles a tree?": "It's because they both have a lot of bark.",
#              "What do you get when you cross a dog and a lion?": "You're not going to get any mail, that's for sure.",
#              "Whenever I take my dog to the park, the ducks always try to bite him.": "I guess it makes sense, since he's pure bread."}
#
# all_music = ["I'm a dog, you're a dog, everybody do the dog",
#              "Who let the dogs out? Who, who, who, who, who? Who let the dogs out? Who, who, who, who, who?",
#              "Well, the party was nice, the party was pumpin' Yippie yi yo And everybody havin' a ball Yippie yi yo",
#              "Scratchin' in the sunshine, diggin' in the dirt Beggin' for a doggie treat, 'Get off the furniture' [Bark!]"]
#
# all_dog_start = []
# with open('deploy_doge_1/acquire/dogs_start', 'r') as file:
#     for line in file:
#         all_dog_start.append(line)
#
#
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'
# # Initialize the database
# db = SQLAlchemy(app)
#
#
# # Create db model
# class Chat(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     sentence = db.Column(db.String(2000), nullable=False)
#     date_created = db.Column(db.DateTime, default=datetime.utcnow())
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#
#
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(200), nullable=False)
#     date_created = db.Column(db.DateTime, default=datetime.utcnow())
#     history = db.relationship('Chat')
#
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
#
# @app.route("/history")
# def all_user():
#     users = User.query.all()
#     return render_template('all_user.html', users=users)
#
#
# @app.route("/history/<string:user_id>")
# def history(user_id):
#     user = User.query.get_or_404(user_id)
#     return render_template('history.html', chat=user.history)
#
#
# @app.route("/get")
# def get_bot_response(context=context):
#     print(str(request.remote_addr))
#     if User.query.filter_by(name=str(request.remote_addr)).first() is not None:
#         cur_user = User.query.filter_by(name=str(request.remote_addr)).first()
#     else:
#         cur_user = User(name=str(request.remote_addr))
#         db.session.add(cur_user)
#         db.session.commit()
#     sentence = request.args.get("msg")
#     if cur_user is not None and sentence is not None:
#         tmp_history = Chat(sentence="human:   " + sentence, user_id=cur_user.id)
#         db.session.add(tmp_history)
#         db.session.commit()
#     if context == [] or sentence is None:
#         dog_start = random.choice(all_dog_start)
#         for i in range(len(context)):
#             context.remove(context[0])
#         context.append("HEY!")
#         context.append(dog_start)
#         tmp_history = Chat(sentence="doge:   " + dog_start, user_id=cur_user.id)
#         db.session.add(tmp_history)
#         db.session.commit()
#         return dog_start
#     else:
#         context.append(sentence)
#         if len(context) >= 17:
#             context.remove(context[0])
#             context.remove(context[0])
#         # rule model
#         result_rule = find_response_rule("".join([char for char in sentence if char not in string.punctuation])
#                                          , tags_rule, intents_result)
#         if context[-2] in all_jokes:
#             pre_joke = context[-2]
#             context.append(all_jokes[pre_joke])
#             tmp_history = Chat(sentence="doge:   " + all_jokes[pre_joke], user_id=cur_user.id)
#             db.session.add(tmp_history)
#             db.session.commit()
#             return {'result_final': [all_jokes[pre_joke]], 'number': 1}
#         if result_rule != "Unknown":
#             if result_rule == "Music":
#                 result_final = random.choice(all_music)
#                 if context[-3] != "Here you go!":
#                     context.append("Here you go!")
#                     context.append(result_final)
#                     tmp_history = Chat(sentence="doge:   " + result_final, user_id=cur_user.id)
#                     db.session.add(tmp_history)
#                     db.session.commit()
#                     return {'result_final': ["Here you go!", '[Music] ' + result_final], 'number': 2}
#             if result_rule == "Jokes":
#                 result_final = random.choice(list(all_jokes.keys()))
#                 context.append("Tell you something funny:")
#                 context.append(result_final)
#                 tmp_history = Chat(sentence="doge:   " + result_final, user_id=cur_user.id)
#                 db.session.add(tmp_history)
#                 db.session.commit()
#                 return {'result_final': ["Tell you something funny:", result_final], 'number': 2}
#             result_final = result_rule
#             if result_final in context:
#                 result_final = find_rag_open(context[-1:])
#         else:
#             fact_life_rank = fact_life_rank_response(tags, sentence)
#             tmp = " ".join(i for i in context[:-1])
#             topic_change = topic_change_response(topic_change_labels, context[:-1], sentence)
#             if topic_change == "different":
#                 # for i in range(len(context)):
#                 #     context.remove(context[0])
#                 # context.append(sentence)
#                 fact_life_rank = 'Fact'
#             print(fact_life_rank)
#             if fact_life_rank == "Life" and "why" not in sentence.lower():
#                 question_non_question_rank = question_statement_response(question_statement_labels, sentence)
#                 print(question_non_question_rank)
#                 if question_non_question_rank == "question":
#                     result_final = find_rag_open(context[-1:])
#                     if len(sentence) <= 15:
#                         result_final = find_response_open(["HI", result_final], [], blender_memory_agent)
#                 else:
#                     result_acquire = find_multiple_response_acquired(context[-3:], data_acquire)
#                     result_open = find_response_open(context, result_acquire, blender_memory_agent)
#                     result_acquire.append(result_open)
#                     result_final = rank_response(context, result_acquire)
#                     print(result_acquire)
#                 # if len(context) >= 7:
#                 #     while result_final in context[-7:]:
#                 #         result_acquire.remove(result_final)
#                 #         print(result_acquire)
#                 #         result_final = rank_response_second(context, result_acquire, len(result_acquire))
#                 # else:
#                 #     while result_final in context:
#                 #         result_acquire.remove(result_final)
#                 #         print(result_acquire)
#                 #         result_final = rank_response_second(context, result_acquire, len(result_acquire))
#             else:
#                 result_open = find_response_open(context, [], blender_memory_agent)
#                 result_final = result_open
#         if result_final in context:
#             result_final = find_response_open(context, result_final, blender_memory_agent)
#         result_final = result_final.replace("doge wikipedia /", "")
#         context.append(result_final)
#         tmp_history = Chat(sentence="doge:   " + result_final, user_id=cur_user.id)
#         db.session.add(tmp_history)
#         db.session.commit()
#         print("HERE IS THE HISTORY", context)
#         return {'result_final': [result_final], 'number': 1}
#
# if __name__ == "__main__":
#     app.run('0.0.0.0', port='8080')
