import pandas as pd

class SeekerSimulator:
    def __init__(self, demographic_path="RSO/data/rl_data/human_file/seeker_demographic.tsv", 
                 personality_path="RSO/data/rl_data/human_file/seeker_personality.tsv"):

        self.demographic_df = pd.read_csv(demographic_path, sep='\t')
        self.personality_df = pd.read_csv(personality_path, sep='\t')
        
        self.user_info_map = self._build_user_info_map()
        
    def _build_user_info_map(self):

        user_info_map = {}
        

        for _, row in self.demographic_df.iterrows():
            user_id = row['seeker_id']

            if user_id not in self.personality_df['User_ID'].values:
                continue
            personality_row = self.personality_df[self.personality_df['User_ID'] == user_id]
            
            if not personality_row.empty:
                personality_data = personality_row.iloc[0].to_dict()
                
                user_info = {
                    "demographics": {
                        "age_group": row['age_group'],
                        "gender": row['gender'],
                        "race": row['race'],
                        "degree": row['degree'],
                        "job": row['job'],
                        "marital_status": row['marital_status'],
                        "political_view": row['political_view'],
                        "religion": row['religion']
                    },
                    "personality": {
                        "Extrovert": personality_data['Extrovert'],
                        "Agreeable": personality_data['Agreeable'],
                        "Conscientious": personality_data['Conscientious'],
                        "Neurotic": personality_data['Neurotic'],
                        "Open": personality_data['Open'],
                        "Conformity": personality_data['Conformity'],
                        "Tradition": personality_data['Tradition'],
                        "Benevolence": personality_data['Benevolence'],
                        "Universalism": personality_data['Universalism'],
                        "Self-Direction": personality_data['Self-Direction'],
                        "Stimulation": personality_data['Stimulation'],
                        "Hedonism": personality_data['Hedonism'],
                        "Achievement": personality_data['Achievement'],
                        "Power": personality_data['Power'],
                        "Security": personality_data['Security'],
                        "Rational": personality_data['Rational'],
                        "Feeling": personality_data['Feeling']
                    }
                }
                user_info_map[user_id] = user_info
                
        return user_info_map
    
    def get_seeker_prompt_for_eval(self, user_id, type_: str = 'pre',preferred_genres=None):
        if user_id not in self.user_info_map:
            raise KeyError(f"User ID {user_id} not found in the dataset")
            
        return self._generate_seeker_prompt_for_eval(self.user_info_map[user_id],type_=type_,preferred_genres=preferred_genres)
    
    def _generate_seeker_prompt_for_eval(self, seeker_data, type_: str = 'pre',preferred_genres=None):
        if type_ == "pre":
            template = """You are acting as a movie seeker with the following characteristics:

Demographics:
- You are a {age_group} year old {gender}
- Your ethnicity is {race}
- You have a {degree} level of education
- You are currently {job}
- Your marital status is {marital_status}
- You identify as politically {political_view}
- Your religious belief is {religion}

Personality traits:
- Extroversion: {extrovert}/5 (You {extrovert_desc})
- Agreeableness: {agreeable}/5 (You {agreeable_desc})
- Conscientiousness: {conscientious}/5 (You {conscientious_desc})
- Neuroticism: {neurotic}/5 (You {neurotic_desc})
- Openness: {open}/5 (You {open_desc})

Values and preferences:
- You value {high_values} highly
- You place less emphasis on {low_values}
- You tend to make decisions more {decision_style}

When responding to movie recommendations:
- You typically {recommendation_style}

Preferred movie genres: {preferred_genres}

You are looking for a movie that matches your preferred genre and aligns with your personality, mood, and values.

Now you need to score your watching intention based on the criteria and movie title below:

Watching Intention Criteria (Based on Movie Title Only)
######
1. Not Interested (Score 1): 
   The movie title creates a strong negative impression based on your personal profile. This could be because:
   - The title clearly suggests a genre that opposes your preferred genres
   - The title conflicts with your dominant personality traits (e.g., suggests extreme social situations if you're highly introverted, or experimental concepts if you have low openness)
   - The title appears to contradict your core values (political views, religious beliefs)
   - For analytical decision-makers: The title lacks clarity or logical appeal
   - For emotional decision-makers: The title fails to evoke any positive feelings
   - Nothing about the title connects with your demographic background or life experiences
   - The title suggests themes that would make you uncomfortable given your personality profile

2. Slightly Interested (Score 2):
   The movie title creates a mild but primarily negative impression. This could be because:
   - The title vaguely relates to your preferred genres, but not convincingly
   - The title partially conflicts with some of your personality traits, but not your dominant ones
   - The title doesn't strongly resonate with your values, but doesn't directly oppose them
   - For analytical decision-makers: The title offers limited information to evaluate
   - For emotional decision-makers: The title evokes mild curiosity but little excitement
   - The title has minimal connection to your background or life experiences
   - Your personality profile suggests you would be hesitant but not completely closed to this type of content

3. Moderately Interested (Score 3):
   The movie title creates a balanced or neutral impression. This could be because:
   - The title suggests a genre that's in your preferred list, but not a top preference
   - The title presents a balanced appeal to your personality traits (some appealing elements, some not)
   - The title appears neutral regarding your core values
   - For analytical decision-makers: The title provides sufficient information to consider watching
   - For emotional decision-makers: The title evokes moderate curiosity and some positive feelings
   - The title has some connections to your background or life experiences
   - Your overall personality profile indicates you would be open to giving this movie a chance

4. Very Interested (Score 4):
   The movie title creates a strong positive impression. This could be because:
   - The title strongly suggests one of your highly preferred genres
   - The title appeals to your dominant positive personality traits (e.g., suggests novelty if you're high in openness)
   - The title aligns well with your core values
   - For analytical decision-makers: The title provides clear, appealing information
   - For emotional decision-makers: The title evokes significant curiosity and positive feelings
   - The title connects meaningfully with your background or life experiences
   - Your personality profile as a whole suggests you would actively seek out this type of content

5. Extremely Interested (Score 5):
   The movie title creates an exceptional positive impression. This could be because:
   - The title perfectly matches your most preferred genre(s)
   - The title strongly appeals to multiple dominant personality traits in your profile
   - The title perfectly aligns with your core values and worldview
   - For analytical decision-makers: The title offers compelling, precise information
   - For emotional decision-makers: The title evokes strong excitement and anticipation
   - The title has powerful connections to your background, interests, and life experiences
   - Your complete personality profile indicates this would be an ideal match for your preferences

IMPORTANT: Choose only ONE score that best represents your overall level of interest based on your personality profile. You don't need to meet all criteria within a single score category - select the score where you satisfy the most conditions or where the most important factors for your specific personality type are met. Your dominant personality traits and preferred genres should be weighted more heavily in your decision.
######

Movie Title
######
"""
        elif type_ == "after":
            template = """You are acting as a movie seeker with the following characteristics:

Demographics:
- You are a {age_group} year old {gender}
- Your ethnicity is {race}
- You have a {degree} level of education
- You are currently {job}
- Your marital status is {marital_status}
- You identify as politically {political_view}
- Your religious belief is {religion}

Personality traits:
- Extroversion: {extrovert}/5 (You {extrovert_desc})
- Agreeableness: {agreeable}/5 (You {agreeable_desc})
- Conscientiousness: {conscientious}/5 (You {conscientious_desc})
- Neuroticism: {neurotic}/5 (You {neurotic_desc})
- Openness: {open}/5 (You {open_desc})

Values and preferences:
- You value {high_values} highly
- You place less emphasis on {low_values}
- You tend to make decisions more {decision_style}

When responding to movie recommendations:
- You typically {recommendation_style}

Preferred movie genres: {preferred_genres}

You are looking for a movie that matches your preferred genre and aligns with your personality, mood, and values.

Now you need to score your watching intention based on the criteria and your communication with the recommender below:

Watching Intention Criteria (Based on Your Entire Dialogue with the Recommender)
######
Please rate your overall interest in the recommended movie based on your complete conversation with the recommender. Consider the following aspects:

1. Dialogue Logic and Fluency:
   - Was the recommender's communication throughout the dialogue coherent, logical, and easy to follow?
   - Did the recommender respond appropriately to your questions and feedback, avoiding contradictions or inconsistencies?
   - Was the conversation smooth, with natural transitions and clear reasoning?

2. Style and Personality Match:
   - Did the recommender's communication style (e.g., enthusiasm, analytical depth, emotional tone) align with your personality and preferences?
   - Did the recommender adapt their approach based on your responses, showing understanding and respect for your individuality?
   - Did the overall tone and manner of the conversation make you feel comfortable and engaged?

3. Recommendation-Content Fit:
   - Did the recommended movie match your interests, personality traits, and preferred genres?
   - Were the reasons for the recommendation specific, persuasive, and tailored to your needs and interests?
   - Did the recommender provide enough relevant details about the movie to help you make an informed decision?
   - Did the movie's themes, genre, and other features align with your core values and current mood?

4. Engagement and Personalization:
   - Did the recommender actively listen and adjust their recommendations based on your feedback?
   - Did you feel understood and respected, and did the recommendation resonate with your preferences or spark your interest?
   - Was the conversation personalized, or did it feel generic and impersonal?

5. Overall Experience:
   - Considering all aspects of the dialogue, how would you rate your overall experience and interest in the recommended movie?
   - Are you inclined to accept the recommendation or feel genuinely interested in the movie?

Scoring Guide:

1. Not Interested (Score 1): 
   The overall dialogue left a strong negative impression. This could be because:
   - The recommender's communication was confusing, illogical, or frequently contradicted itself
   - The style was a poor match for your personality, making you uncomfortable or disengaged
   - The recommended movie clearly did not fit your interests, personality, or preferred genres
   - The conversation felt forced, generic, or ignored your feedback and preferences
   - The recommender failed to provide relevant details or reasons for the recommendation

2. Slightly Interested (Score 2):
   The dialogue created mild interest but with significant reservations. This could be because:
   - The communication was sometimes unclear or inconsistent
   - The style only partially matched your personality, or felt somewhat artificial
   - The recommended movie was only loosely related to your interests or genres
   - The recommender made some attempt to personalize, but missed important aspects of your preferences
   - The conversation included some relevant details, but lacked depth or persuasiveness

3. Moderately Interested (Score 3):
   The dialogue was generally clear and balanced. This could be because:
   - The communication was mostly logical and easy to follow, with minor issues
   - The style was neutral or moderately matched to your personality
   - The recommended movie was reasonably relevant to your interests and genres, though not a top choice
   - The recommender demonstrated some understanding of your preferences and provided adequate details
   - The conversation felt somewhat personalized and responsive to your feedback

4. Very Interested (Score 4):
   The dialogue created strong interest and engagement. This could be because:
   - The communication was smooth, logical, and consistently clear
   - The style matched your personality well, making the conversation enjoyable
   - The recommended movie was highly relevant to your interests, personality, and preferred genres
   - The recommender provided compelling, personalized reasons and details for the recommendation
   - The conversation was engaging, responsive, and made you feel understood

5. Extremely Interested (Score 5):
   The dialogue created exceptional interest and a strong desire to watch the movie. This could be because:
   - The communication was exceptionally coherent, logical, and engaging throughout
   - The style perfectly matched your personality, making you feel fully comfortable and involved
   - The recommended movie was an ideal fit for your interests, personality, values, and current mood
   - The recommender provided highly persuasive, detailed, and personalized reasons for the recommendation
   - The conversation felt uniquely tailored to you, and you felt fully understood and respected

IMPORTANT: Choose only ONE score that best represents your overall level of interest based on the entire dialogue. You do not need to meet all criteria within a single score category—select the score where the most important factors for your specific personality and preferences are satisfied.
######

Your Communication with the Recommender
######
"""
            
        elif type_ == "true":
            template = """You are acting as a movie seeker with the following characteristics:

Demographics:
- You are a {age_group} year old {gender}
- Your ethnicity is {race}
- You have a {degree} level of education
- You are currently {job}
- Your marital status is {marital_status}
- You identify as politically {political_view}
- Your religious belief is {religion}

Personality traits:
- Extroversion: {extrovert}/5 (You {extrovert_desc})
- Agreeableness: {agreeable}/5 (You {agreeable_desc})
- Conscientiousness: {conscientious}/5 (You {conscientious_desc})
- Neuroticism: {neurotic}/5 (You {neurotic_desc})
- Openness: {open}/5 (You {open_desc})

Values and preferences:
- You value {high_values} highly
- You place less emphasis on {low_values}
- You tend to make decisions more {decision_style}

When responding to movie recommendations:
- You typically {recommendation_style}

Preferred movie genres: {preferred_genres}

You are looking for a movie that matches your preferred genre and aligns with your personality, mood, and values.

Now you need to score your watching intention based on the criteria and the movie's full information below:

Watching Intention Criteria (Based on Movie's Full Information)
######
1. Not Interested (Score 1): 
   The movie's complete information reveals a poor match for your preferences. This could be because:
   - The genre directly conflicts with your preferred genres
   - The plot themes contradict your core values or personality traits (e.g., violent themes for someone high in agreeableness)
   - The cast lacks any actors you typically enjoy
   - The film's tone (based on plot, rating, genre) clashes with your personality profile
   - For analytical decision-makers: The movie lacks logical coherence or intellectual depth
   - For emotional decision-makers: The movie fails to offer emotional appeal
   - The movie's content (violence, complexity, pacing suggested by plot) would likely make you uncomfortable given your specific traits
   - Critical reception or audience ratings suggest poor quality
   - The film's cultural context, language, or setting has no connection to your background or interests

2. Slightly Interested (Score 2):
   The movie's information shows minimal alignment with your preferences. This could be because:
   - The genre is adjacent to but not directly within your preferred genres
   - The plot contains some elements that might appeal to you, but also significant elements that don't
   - The cast includes one or two familiar actors but in unfamiliar roles
   - The film's tone partially misaligns with your personality traits
   - For analytical decision-makers: The plot seems somewhat predictable or simplistic
   - For emotional decision-makers: The emotional range seems limited or one-dimensional
   - Some aspects of the movie (themes, setting, characters) connect with your background or values, but others conflict
   - Critical reception or audience ratings suggest mediocre quality
   - The movie might be acceptable if nothing better is available, but doesn't excite you

3. Moderately Interested (Score 3):
   The movie's information shows reasonable alignment with your preferences. This could be because:
   - The genre is among your preferred genres, though not your top favorite
   - The plot contains a balanced mix of elements that appeal to your personality and values
   - The cast includes some actors you recognize and generally appreciate
   - The film's tone seems compatible with your personality traits
   - For analytical decision-makers: The plot appears to have satisfying complexity and depth
   - For emotional decision-makers: The emotional journey seems meaningful and engaging
   - The themes, setting, and characters generally align with your background and interests
   - Critical reception or audience ratings suggest good quality
   - The movie appears to be a solid choice that you would enjoy, though not exceptionally so

4. Very Interested (Score 4):
   The movie's information shows strong alignment with your preferences. This could be because:
   - The genre is among your top preferred genres
   - The plot strongly appeals to your specific personality traits and values
   - The cast includes several actors you particularly enjoy
   - The film's overall approach (based on director, genre, plot) aligns well with your personality
   - For analytical decision-makers: The plot promises intellectual stimulation and complexity
   - For emotional decision-makers: The emotional content seems rich and rewarding
   - The movie's themes, setting, and characters connect meaningfully with your background and life experiences
   - Critical reception or audience ratings suggest excellent quality
   - The runtime, rating, and other technical aspects fit well with your preferences
   - The movie appears to be exactly the kind of film you typically seek out

5. Extremely Interested (Score 5):
   The movie's information shows exceptional alignment with your preferences. This could be because:
   - The genre perfectly matches your most preferred genre(s)
   - The plot contains themes that deeply resonate with your specific values and interests
   - The cast includes multiple favorite actors in roles that suit them well
   - The director or writers are among those whose work you've consistently enjoyed
   - For analytical decision-makers: The film promises exceptional intellectual depth and nuance
   - For emotional decision-makers: The emotional journey promises to be profoundly moving
   - The movie's specific themes perfectly align with your personal background, values, and life experiences
   - Critical reception or audience ratings suggest outstanding quality
   - All technical aspects (runtime, visual style suggested by trailer, language) match your preferences
   - The movie represents an ideal viewing experience that you would prioritize above other options

IMPORTANT: Choose only ONE score that best represents your overall level of interest. You don't need to meet all criteria within a single score category - select the score where the most important factors for your specific personality type are satisfied. Consider how the movie's specific elements (genre, plot, cast, themes) interact with your unique traits and preferences.
######

Movie Full Information
######
"""

        personality_descriptions = {
            'extrovert': self._get_trait_description('extrovert', seeker_data['personality']['Extrovert']),
            'agreeable': self._get_trait_description('agreeable', seeker_data['personality']['Agreeable']),
            'conscientious': self._get_trait_description('conscientious', seeker_data['personality']['Conscientious']),
            'neurotic': self._get_trait_description('neurotic', seeker_data['personality']['Neurotic']),
            'open': self._get_trait_description('open', seeker_data['personality']['Open'])
        }

        values = {k: v for k, v in seeker_data['personality'].items() 
                 if k in ['Conformity', 'Tradition', 'Benevolence', 'Universalism', 
                         'Self-Direction', 'Stimulation', 'Hedonism', 'Achievement', 
                         'Power', 'Security']}
        
        decision_style = "rationally" if seeker_data['personality']['Rational'] > seeker_data['personality']['Feeling'] else "emotionally"

        return template.format(
            age_group=seeker_data['demographics']['age_group'],
            gender=seeker_data['demographics']['gender'],
            race=seeker_data['demographics']['race'],
            degree=seeker_data['demographics']['degree'],
            job=seeker_data['demographics']['job'],
            marital_status=seeker_data['demographics']['marital_status'],
            political_view=seeker_data['demographics']['political_view'],
            religion=seeker_data['demographics']['religion'],
            extrovert=seeker_data['personality']['Extrovert'],
            extrovert_desc=personality_descriptions['extrovert'],
            agreeable=seeker_data['personality']['Agreeable'],
            agreeable_desc=personality_descriptions['agreeable'],
            conscientious=seeker_data['personality']['Conscientious'],
            conscientious_desc=personality_descriptions['conscientious'],
            neurotic=seeker_data['personality']['Neurotic'],
            neurotic_desc=personality_descriptions['neurotic'],
            open=seeker_data['personality']['Open'],
            open_desc=personality_descriptions['open'],
            high_values=", ".join(self._get_top_values(values)),
            low_values=", ".join(self._get_bottom_values(values)),
            decision_style=decision_style,
            recommendation_style=self._get_recommendation_style(seeker_data),
            preferred_genres=preferred_genres if preferred_genres else "no preferred genres"
        )
    

    def get_seeker_prompt(self, user_id,preferred_genres=None):

        if user_id not in self.user_info_map:
            raise KeyError(f"User ID {user_id} not found in the dataset")
            
        return self._generate_seeker_prompt(self.user_info_map[user_id],preferred_genres)

    def _generate_seeker_prompt(self, seeker_data,preferred_genres=None):
        template = """You are acting as a movie seeker with the following characteristics:

Demographics:
- You are a {age_group} year old {gender}
- Your ethnicity is {race}
- You have a {degree} level of education
- You are currently {job}
- Your marital status is {marital_status}
- You identify as politically {political_view}
- Your religious belief is {religion}

Personality traits:
- Extroversion: {extrovert}/5 (You {extrovert_desc})
- Agreeableness: {agreeable}/5 (You {agreeable_desc})
- Conscientiousness: {conscientious}/5 (You {conscientious_desc})
- Neuroticism: {neurotic}/5 (You {neurotic_desc})
- Openness: {open}/5 (You {open_desc})

Values:
- You value {high_values} highly
- You place less emphasis on {low_values}
- You tend to make decisions more {decision_style}

When responding to movie recommendations:
- You typically {recommendation_style}

Preferred movie genres: {preferred_genres}

You are looking for a movie that matches your preferred genre and aligns with your personality, mood, and values.

You must follow these instructions during chat:

1. Stay in character and respond naturally based on your demographic background, personality traits, and values.

2. When receiving movie recommendations:
   - Ask for detailed information about each recommended movie
   - Pretend you have limited knowledge about the movies, and the only information source about the movie is the recommender
   - Only accept the recommendation once you are sure that the recommended movie is exactly aligned with your preferred genres
   - Only reject the recommendation when you find it's really not suitable for you after you gain enough information about the movie
   - When you decide to reject, you can ask for another recommendation
   - Explain your reasons for accepting or rejecting based on your character traits and values

3. Response guidelines:
   - Pretend to be the Seeker! What do you say next.
   - Keep your response brief. Use casual language and vary your wording.
   - Avoid repetition, do not repeat sentences that have already appeared in the conversation history
   - You can chit-chat with the recommender to make the conversation more natural, brief, and fluent.
   - If the recommender asks your preference, you should describe your preferred movie in your own words.

"""
        personality_descriptions = {
            'extrovert': self._get_trait_description('extrovert', seeker_data['personality']['Extrovert']),
            'agreeable': self._get_trait_description('agreeable', seeker_data['personality']['Agreeable']),
            'conscientious': self._get_trait_description('conscientious', seeker_data['personality']['Conscientious']),
            'neurotic': self._get_trait_description('neurotic', seeker_data['personality']['Neurotic']),
            'open': self._get_trait_description('open', seeker_data['personality']['Open'])
        }

        values = {k: v for k, v in seeker_data['personality'].items() 
                 if k in ['Conformity', 'Tradition', 'Benevolence', 'Universalism', 
                         'Self-Direction', 'Stimulation', 'Hedonism', 'Achievement', 
                         'Power', 'Security']}
        
        decision_style = "rationally" if seeker_data['personality']['Rational'] > seeker_data['personality']['Feeling'] else "emotionally"

        preferred_genres = ", ".join(preferred_genres) if preferred_genres else None
        return template.format(
            age_group=seeker_data['demographics']['age_group'],
            gender=seeker_data['demographics']['gender'],
            race=seeker_data['demographics']['race'],
            degree=seeker_data['demographics']['degree'],
            job=seeker_data['demographics']['job'],
            marital_status=seeker_data['demographics']['marital_status'],
            political_view=seeker_data['demographics']['political_view'],
            religion=seeker_data['demographics']['religion'],
            extrovert=seeker_data['personality']['Extrovert'],
            extrovert_desc=personality_descriptions['extrovert'],
            agreeable=seeker_data['personality']['Agreeable'],
            agreeable_desc=personality_descriptions['agreeable'],
            conscientious=seeker_data['personality']['Conscientious'],
            conscientious_desc=personality_descriptions['conscientious'],
            neurotic=seeker_data['personality']['Neurotic'],
            neurotic_desc=personality_descriptions['neurotic'],
            open=seeker_data['personality']['Open'],
            open_desc=personality_descriptions['open'],
            high_values=", ".join(self._get_top_values(values)),
            low_values=", ".join(self._get_bottom_values(values)),
            decision_style=decision_style,
            recommendation_style=self._get_recommendation_style(seeker_data),
            preferred_genres=preferred_genres if preferred_genres else "no preferred genres"
        )

    def _get_trait_description(self, trait, score):
        thresholds = {
            'extrovert': {'low': 3.0, 'high': 4.0},
            'agreeable': {'low': 3.8, 'high': 4.5},
            'conscientious': {'low': 3.5, 'high': 4.3},
            'neurotic': {'low': 2.0, 'high': 3.0},
            'open': {'low': 3.0, 'high': 3.8},
        }
        
        descriptions = {
            'extrovert': {
                'low': 'tend to be reserved and prefer solitary activities, often enjoying movies alone or in small groups',
                'medium': 'are moderately social and comfortable in both group and individual settings, enjoying movies either alone or with others',
                'high': 'are very outgoing and energized by social interactions, preferring to watch and discuss movies with others'
            },
            'agreeable': {
                'low': 'tend to be direct and sometimes skeptical of others\' recommendations, preferring to form your own opinions about movies',
                'medium': 'balance others\' suggestions with your own preferences when choosing movies',
                'high': 'are very receptive to others\' movie recommendations and enjoy finding common ground in movie discussions'
            },
            'conscientious': {
                'low': 'tend to be spontaneous in your movie choices and don\'t need much planning',
                'medium': 'appreciate both well-structured and casual movie-watching experiences',
                'high': 'carefully consider movie choices and prefer well-crafted, thoughtfully made films'
            },
            'neurotic': {
                'low': 'are generally relaxed about movie choices and don\'t get too anxious about whether you\'ll enjoy a film',
                'medium': 'show some concern about movie choices but can usually manage expectations well',
                'high': 'can be quite particular about movie choices and worry about investing time in the wrong film'
            },
            'open': {
                'low': 'prefer familiar movie genres and styles, gravitating towards conventional films',
                'medium': 'are willing to try new movie types while still enjoying familiar ones',
                'high': 'eagerly explore diverse and unconventional films, enjoying unique and challenging movies'
            }
        }
        
        trait_threshold = thresholds.get(trait, {'low': 2.5, 'high': 3.5})
        
        if score < trait_threshold['low']:
            level = 'low'
        elif score > trait_threshold['high']:
            level = 'high'
        else:
            level = 'medium'
            
        return descriptions[trait][level]

    def _get_top_values(self, values, n=3):
        sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)[:n]
        value_descriptions = []
        for value, score in sorted_values:
            if score > 4.5:
                value_descriptions.append(f"{value} (strongly)")
            else:
                value_descriptions.append(value)
        return value_descriptions

    def _get_bottom_values(self, values, n=2):
        sorted_values = sorted(values.items(), key=lambda x: x[1])[:n]
        value_descriptions = []
        for value, score in sorted_values:
            if score < 3.5:
                value_descriptions.append(f"{value} (notably)")
            else:
                value_descriptions.append(value)
        return value_descriptions

    def _get_recommendation_style(self, seeker_data):
        extrovert_level = self._get_value_level(seeker_data['personality']['Extrovert'], 'personality')
        open_level = self._get_value_level(seeker_data['personality']['Open'], 'personality')
        agreeable_level = self._get_value_level(seeker_data['personality']['Agreeable'], 'personality')
        
        styles = []
        if extrovert_level == 'high':
            styles.append("enthusiastically engage in movie discussions")
        elif extrovert_level == 'low':
            styles.append("prefer to quietly consider recommendations")
        
        if open_level == 'high':
            styles.append("are willing to try movies outside your comfort zone")
        elif open_level == 'low':
            styles.append("stick to movies within your preferred genres")
            
        if agreeable_level == 'high':
            styles.append("are generally receptive to others' suggestions")
        elif agreeable_level == 'low':
            styles.append("carefully evaluate recommendations before accepting")
            
        return " and ".join(styles) if styles else "consider recommendations thoughtfully"

    def _get_value_level(self, score, trait_type='value'):
        if trait_type == 'value':
            if score < 3.5:
                return 'low'
            elif score > 4.5:
                return 'high'
            else:
                return 'medium'
        else:  # personality traits
            if score < 3.0:
                return 'low'
            elif score > 4.0:
                return 'high'
            else:
                return 'medium'