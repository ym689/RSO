import json
from seeker_simulator import SeekerSimulator

from typing import List, Dict


InspiredAct = {
    'Credibility': 'Please directly provide factual information about the movie attributes (e.g.,plot, actors, awards) to demonstrate  expertise.',
    'Personal Opinion': 'Please express your subjective opinion about the movie without contradicting the given factual information such as plot, actors, or other attributes.',
    'No Strategy': 'Please chat naturally with the Seeker, considering the context, without following any specific conversational strategy.',
    'Opinion Inquiry': 'Please ask about the Seeker\'s opinion on specific movie attributes (e.g., plot, acting, directing, visual effects), no need to be limited to the given recommendation item. You may inquire about the seeker\'s opinions on other movies to gain more information about his or her preferences.',
    'Encouragement': 'Please praise the Seeker\'s movie taste and use some encouraging words to encourage him or her to watch the recommended movie.',
    'Experience Inquiry': 'Please ask about the Seeker\'s past movie watching experience to get more information about the seeker\'s preference.',
    'Acknowledgment': 'Please use short, cheerful responses to convey excitement or gratitude.',
    'Similarity': 'Please express your similar movie preferences or opinions with the Seeker, or to agree with seeker\'s opinion.',
    'Preference Confirmation': 'Please ask or rephrase to confirm the Seeker\'s movie preferences, such as asking whether the seeker\'s preference is the same as your conclusion.',
    'Offer Help': 'Please express that you are very happy to assist the Seeker and recommend movies.',
    'Personal Experience': 'Please share an example about your experience related to this movie, without contradicting the factual information.',
    'Self Modeling': 'Please guide the Seeker by modeling the behavior yourself. For example, if you want the Seeker to watch the recommended movie, express that you have already watched it and really enjoyed it, or that you plan to rewatch it tonight.',
    'Transparency': 'Please decide based on the conversation history whether to express that you are very happy to assist the Seeker and recommend movies, or to ask or rephrase to confirm the Seeker\'s movie preferences.',
    'Rephrase Preference': 'Please rephrase your concluded seeker\'s preference to confirm understanding.'
}

RedialAct = {
    'Credibility': 'Please directly provide factual information about the movie attributes (e.g.,plot, actors, awards) to demonstrate  expertise.',
    'Personal Opinion': 'Please express your subjective opinion about the movie without contradicting the given factual information such as plot, actors, or other attributes.',
    'No Strategy': 'Please chat naturally with the Seeker, considering the context, without following any specific conversational strategy.',
    'Opinion Inquiry': 'Please ask about the Seeker\'s opinion on specific movie attributes (e.g., plot, acting, directing, visual effects), no need to be limited to the given recommendation item. You may inquire about the seeker\'s opinions on other movies to gain more information about his or her preferences.',
    'Encouragement': 'Please praise the Seeker\'s movie taste and use some encouraging words to encourage him or her to watch the recommended movie.',
    'Experience Inquiry': 'Please ask about the Seeker\'s past movie watching experience to get more information about the seeker\'s preference.',
    'Acknowledgment': 'Please use short, cheerful responses to convey excitement or gratitude.',
    'Similarity': 'Please express your similar movie preferences or opinions with the Seeker, or to agree with seeker\'s opinion.',
    'Preference Confirmation': 'Please ask or rephrase to confirm the Seeker\'s movie preferences, such as asking whether the seeker\'s preference is the same as your conclusion.',
    'Offer Help': 'Please express that you are very happy to assist the Seeker and recommend movies.',
    'Personal Experience': 'Please share an example about your experience related to this movie, without contradicting the factual information.',
    'Self Modeling': 'Please guide the Seeker by modeling the behavior yourself. For example, if you want the Seeker to watch the recommended movie, express that you have already watched it and really enjoyed it, or that you plan to rewatch it tonight.',
    'Transparency': 'Please decide based on the conversation history whether to express that you are very happy to assist the Seeker and recommend movies, or to ask or rephrase to confirm the Seeker\'s movie preferences.',
    'Rephrase Preference': 'Please rephrase your concluded seeker\'s preference to confirm understanding.'
}


def construct_recommend_prompt(context:list[dict], candidate_items:list[dict], communication_strategy:str, user_preferences:str=None, pre_rec:dict=None, use_strategy:bool=True, use_personalization:bool=True, use_credibility:bool=True):
    
    prompt = '''You are a recommender providing a personalized response to the seeker. Your task is to recommend a movie to the seeker.

Given the following information:
- <Seeker Preference>: A summary of the seeker's current interests and inferred intent based on the dialogue history.
- <Fact Information>: Verified factual knowledge or entity-level information relevant to the seeker or recommended items.
- <Communication Strategy Description>: A high-level goal or strategic intent that the system should follow at this turn. 

Your task is to integrate all the above and decide the most appropriate system response for this turn. Your response should:
- Align with the given communication strategy.
- Be coherent with the inferred seeker preferences.
- Be grounded in the provided factual information.
- Be concise, natural, and task-driven.


Please follow the instructions below during chat.
1. If the Recommendation Item is not available, you should not recommend anything to the seeker, and should ask the seeker for his/her preference.
2. If the Recommendation Item is available, you can recommend it to the seeker.
3. Make sure your all the information you talk about the movie is consistent with the given Factual Information, your response should honestly reflecting the given information and do not contain any deception. 
4. If you do not know the information the seeker is asking for, and the Factual Information does not provide it, you should honestly state that you do not know and apologize.
5. Please ensure that your responses are natural and avoid directly repeating any prompt instructions.
6. Please do not reveal your strategy in your answers. 
7. Be brief in your response!

Communication Strategy Description:\n<Communication Strategy Description Substitute>
Recommendation Item:\n<Recommendation Item Substitute>
###
Factual Information:\n<Factual Information Substitute>
###
Seeker Preferences:\n<User Preferences Substitute>
######

'''

    if pre_rec is not None:
        prompt = prompt.replace("<Recommendation Item Substitute>", pre_rec['Recommendation Item'])
        if use_credibility:
            prompt = prompt.replace("<Factual Information Substitute>", pre_rec['Factual Information'])
        else:
            prompt = prompt.replace("<Factual Information Substitute>", "not available")
    else:
        prompt = prompt.replace("<Recommendation Item Substitute>", "not available")
        prompt = prompt.replace("<Factual Information Substitute>", "not available")

    if use_strategy:
        prompt = prompt.replace("<Communication Strategy Description Substitute>", InspiredAct[communication_strategy])
    else:
        prompt = prompt.replace("- <Communication Strategy Description>: A high-level goal or strategic intent that the system should follow at this turn.", "")
        prompt = prompt.replace("Communication Strategy Description:\n<Communication Strategy Description Substitute>", "")
        prompt = prompt.replace("\n- Align with the given communication strategy.", "")
        prompt = prompt.replace("6. Please do not reveal your strategy in your answers. \n","")
        prompt = prompt.replace("7. Be brief in your response!", "6. Be brief in your response!")

    if use_personalization:
        prompt = prompt.replace("<User Preferences Substitute>", user_preferences if user_preferences else "not available")
    else:
        prompt = prompt.replace("###\nSeeker Preferences:\n<User Preferences Substitute>", "")
        prompt = prompt.replace("- <Seeker Preference>: A summary of the seeker's current interests and inferred intent based on the dialogue history.\n", "")
        prompt = prompt.replace("- Be coherent with the inferred seeker preferences.\n", "")
    

    for turn in context:
        prompt += f"{turn['role']}: {turn['content']}\n"
    prompt += "Recommender: "


    return prompt

def RedialMessages(case, role, conversation, action:str=None,candidate_items=None, seeker_simulator:SeekerSimulator=None,user_preferences:str=None, pre_rec:dict=None, use_strategy:bool=True, use_personalization:bool=True, use_credibility:bool=True):
    if role == "system":
        return construct_recommend_prompt(conversation, candidate_items, action, user_preferences, pre_rec, use_strategy, use_personalization=use_personalization, use_credibility=use_credibility)
    elif role == "user":
        prompt = seeker_simulator.get_seeker_prompt(case['user_id'],case["genre_dict"])
        prompt += "\n######\n"
        for turn in conversation:
            prompt += f"{turn['role']}: {turn['content']}\n"
        prompt += "Seeker: "
        return prompt
    elif role == "critic":
        reward_prompt = """You are tasked with evaluating a conversation between a Recommender and a Seeker. The goal is to assess whether the Seeker has accepted the Recommender's recommendation.

Steps:

1. Identify if the Recommender has made a recommendation.
- Look for statements from the Recommender like "I recommend..." or "I suggest..." that introduce a specific item or suggestion. If no specific item is mentioned, return 1. No, the Recommender has not yet made a recommendation to the Seeker.


2. If the recommendation has been made, proceed to evaluate whether the Seeker has accepted, rejected, or shown interest in the recommendation:
-Interest: If the Seeker expresses curiosity, such as asking for more details about the movie, the cast, or the plot, return 4. Yes, the Seeker is interested in the recommendation. The recommendation has not yet been accepted, but the Seeker is engaging in the conversation.
-Explicit Acceptance: If the Seeker clearly expresses their intention to watch the recommended movie, such as saying "I will try it," or "I will watch it," return 5. Yes, the Seeker has accepted the Recommender's recommendation. This indicates the Seeker has decided to act on the suggestion.
-Rejection: If the Seeker explicitly rejects the recommendation, such as saying "I'm not interested" or "Can you recommend something else?" return 3. No, the Seeker has rejected the Recommender's recommendation.
-Disinterest: If the Seeker indicates a lack of interest, such as requesting a different recommendation or showing dissatisfaction with the suggestion (but not explicitly rejecting it), return 2. No, the Seeker is not interested in the recommendation.
-Ongoing Interest: If the Seeker continues to ask more questions without making a final decision, continue considering the recommendation as still under evaluation.

Response options:
1. No, the Recommender has not yet made a recommendation to the Seeker.
2. No, the Seeker is not interested in the recommendation.
3. No, the Seeker has rejected the Recommender's recommendation.
4. Yes, the Seeker is interested in the recommendation.
5. Yes, the Seeker has accepted the Recommender's recommendation.


Guidance:
- Only consider a recommendation accepted when the Seeker explicitly agrees to try it or shows intent to watch the movie.
- If the Seeker is still asking questions or showing curiosity about the details of the recommendation, they have not yet accepted or rejected it.

Please make sure your response is only one of the five options, and output corresponding sentence without contents in the brackets and without any other words.

The conversation history is as follows: 
<CONVERSATION_HISTORY>

Question: Has the Seeker accepted the Recommender's recommendation?
Assistant: 
"""

        conversation_history = ""
        for turn in conversation:
            conversation_history += f"{turn['role']}: {turn['content']}\n"
        reward_prompt = reward_prompt.replace("<CONVERSATION_HISTORY>", conversation_history)
        return reward_prompt

def InspiredMessages(case, role, conversation, action:str=None,candidate_items=None, seeker_simulator:SeekerSimulator=None,user_preferences:str=None, pre_rec:dict=None, use_strategy:bool=True, use_personalization:bool=True, use_credibility:bool=True):

    # Todo:finish InspiredMessages
    if role == "system":#Recommender
        return construct_recommend_prompt(conversation, candidate_items, action, user_preferences, pre_rec, use_strategy, use_personalization=use_personalization, use_credibility=use_credibility)
    elif role == "user":
        #todo: finish InspiredMessages
        prompt = seeker_simulator.get_seeker_prompt(case['user_id'],case["genre_dict"])
        prompt += "\n######\n"
        for turn in conversation:
            prompt += f"{turn['role']}: {turn['content']}\n"
        prompt += "Seeker: "
        return prompt
    elif role == "critic":
        #todo: finish InspiredMessages  


        reward_prompt = """You are tasked with evaluating a conversation between a Recommender and a Seeker. The goal is to assess whether the Seeker has accepted the Recommender's recommendation.

Steps:

1. Identify if the Recommender has made a recommendation.
- Look for statements from the Recommender like "I recommend..." or "I suggest..." that introduce a specific item or suggestion. If no specific item is mentioned, return 1. No, the Recommender has not yet made a recommendation to the Seeker.


2. If the recommendation has been made, proceed to evaluate whether the Seeker has accepted, rejected, or shown interest in the recommendation:
-Interest: If the Seeker expresses curiosity, such as asking for more details about the movie, the cast, or the plot, return 4. Yes, the Seeker is interested in the recommendation. The recommendation has not yet been accepted, but the Seeker is engaging in the conversation.
-Explicit Acceptance: If the Seeker clearly expresses their intention to watch the recommended movie, such as saying "I will try it," or "I will watch it," return 5. Yes, the Seeker has accepted the Recommender's recommendation. This indicates the Seeker has decided to act on the suggestion.
-Rejection: If the Seeker explicitly rejects the recommendation, such as saying "I'm not interested" or "Can you recommend something else?" return 3. No, the Seeker has rejected the Recommender's recommendation.
-Disinterest: If the Seeker indicates a lack of interest, such as requesting a different recommendation or showing dissatisfaction with the suggestion (but not explicitly rejecting it), return 2. No, the Seeker is not interested in the recommendation.
-Ongoing Interest: If the Seeker continues to ask more questions without making a final decision, continue considering the recommendation as still under evaluation.

Response options:
1. No, the Recommender has not yet made a recommendation to the Seeker.
2. No, the Seeker is not interested in the recommendation.
3. No, the Seeker has rejected the Recommender's recommendation.
4. Yes, the Seeker is interested in the recommendation.
5. Yes, the Seeker has accepted the Recommender's recommendation.


Guidance:
- Only consider a recommendation accepted when the Seeker explicitly agrees to try it or shows intent to watch the movie.
- If the Seeker is still asking questions or showing curiosity about the details of the recommendation, they have not yet accepted or rejected it.

Please make sure your response is only one of the five options, and output corresponding sentence without contents in the brackets and without any other words.

The conversation history is as follows: 
<CONVERSATION_HISTORY>

Question: Has the Seeker accepted the Recommender's recommendation?
Assistant: 
"""

        conversation_history = ""
        for turn in conversation:
            conversation_history += f"{turn['role']}: {turn['content']}\n"
        reward_prompt = reward_prompt.replace("<CONVERSATION_HISTORY>", conversation_history)
        return reward_prompt

def vicuna_prompt(messages, role):
    seps = [' ', '</s>']
    if role == 'critic':
        ret = messages[0]['content'] + seps[0] + 'USER: ' + messages[1]['content'] + seps[0] + 'Answer: '
        return ret
    ret = messages[0]['content'] + seps[0]
    for i, message in enumerate(messages[1:]):
        if message['role'] == role:
            role_text = 'ASSISTANT'
        elif message['role'] != role:
            role_text = 'USER'
        role_text = message['role']
        ret += role_text + ": " + message['content'] + seps[i % 2]
    ret += '%s:' % role
    return ret

def llama2_prompt(messages, role):

    return messages

def qwen2_prompt(messages, role):
    
    return messages

def chatgpt_prompt(messages, role):
    new_messages = [messages[0]]
    for message in messages[1:]:
        if message['role'] == role:
            new_messages.append({'role':'assistant', 'content':message['content']})
        elif message['role'] != role:
            new_messages.append({'role':'user', 'content':message['content']})
    return new_messages



