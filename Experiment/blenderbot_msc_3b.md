#blender_msc_agent = create_agent_from_model_file("zoo:msc/blender3B_1024/model")

blenderbot msc 3b has a small issue about context. If the user input length is too low or it does not has a
clear meaning, chatbot might not be able to realize what user says and use the previous its own saying as user's
input.

context = ["HEY  ", "GOT INTO NEIGHBOUR'S GARDEN LOL  ", "what did you find?  ", 
"I found my neighbors cat stuck in a flap of my neighbor's cat litter box.  ", "lol lol lol"]

bot_answer: OMG!  That is so gross!  I am glad you were able to get it out!

context = ["HEY  ", "GOT INTO NEIGHBOUR'S GARDEN LOL  ", "what did you find?  ", 
"I found my neighbors cat stuck in a flap of my neighbor's cat litter box.  ", "lol OMG! That is so gross!  "]

Yeah, I was so grossed out. I had to call animal control to get it out of there.