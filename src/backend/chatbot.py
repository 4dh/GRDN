import streamlit as st
import pandas as pd
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# set model
model = 'openai'

def parse_and_evaluate_text(text):
    # Find the indices of the opening and closing brackets
    opening_bracket_index = text.find("[")
    closing_bracket_index = text.find("]")

    if opening_bracket_index != -1 and closing_bracket_index != -1:
        # Extract the text within the brackets
        extracted_list = "[" + text[opening_bracket_index + 1: closing_bracket_index] + "]"
        # Return the evaluated text list
        return eval(extracted_list)
        

    else:
        print("Error with parsing plant list")
        return None
    
def chat_response(template, prompt_text, model):
    if model == 'openai':
        chat = ChatOpenAI(temperature=.1)
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template="{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        response = chat(chat_prompt.format_prompt(text= prompt_text).to_messages())
        return response
    # elif model == 'Llama2-7B':
    #     llm = Replicate(
    #         model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    #         temperature=0.1,
    #         #context_window=32,
    #         top_p=0.9,
    #         repetition_penalty=1.0,
    #         max_tokens=2000,
    #         #stop_sequences=["\n\n"],   
    #         )
    #     input_prompt = template + prompt_text
    #     print(input_prompt)
    #     resp = llm.complete(input_prompt)
    #     print(resp)
    #     return resp
    else:
        print("Error with chatbot model")
        return None

# get the plant list from user input
def get_plant_list(input_plant_text):
    template="You are a helpful assistant that knows all about gardening and plants and python data structures."
    text = 'which of the elements of this list can be grown in a garden, [' + input_plant_text + ']? Return JUST a python list object containing the elements that can be grown in a garden. Do not include any other text or explanation.'
    plant_list_text = chat_response(template, text, model)
    plant_list = parse_and_evaluate_text(plant_list_text.content)
    print(plant_list)
    return plant_list

# get plant care tips based on plant list
def get_plant_care_tips(plant_list):
    template="You are a helpful assistant that knows all about gardening, plants, and companion planting."
    text = 'from this list of plants, [' + str(st.session_state.input_plants_raw) + '], generate a list of up to 10 plant care tips or interesting stories of plant compatibility for the plants in the list- maybe 1-2 per plant depending on what you know. Return just the plant care tips in HTML markdown format. Do not include any other text or explanation. It must be in HTML markdown format.'
    plant_care_tips = chat_response(template, text, model)
    print(plant_care_tips.content)
    return plant_care_tips.content

# get compatability matrix for companion planting
def get_compatibility_matrix(plant_list):
    # Convert the compatibility matrix to a string
    with open('data/compatibilities_text.txt', 'r') as file:
        # Read the contents of the file
        compatibility_text = file.read()
    plant_comp_context = compatibility_text
    template="You are a helpful assistant that knows all about gardening, companion planting, and python data structures- specifically compatibility matrices."
    text = 'from this list of plants, [' + str(plant_list) + '], Return JUST a python array (with values separated by commas like this: [[0,1],[1,0]]\n\n ) for companion plant compatibility. Each row and column should represent plants, and the element of the array will contain a -1, 0, or 1 depending on if the relationship between plants is antagonists, neutral, or companions, respectively. You must refer to this knowledge base of information on plant compatibility: \n\n, ' + plant_comp_context + '\n\n A plant\'s compatibility with itself is always 0. Do not include any other text or explanation.'
    compatibility_mat = chat_response(template, text)
    



    # Find the indices of the opening and closing brackets
    opening_bracket_index = compatibility_mat.content.find("[[")
    closing_bracket_index = compatibility_mat.content.find("]]")
    if opening_bracket_index != -1 and closing_bracket_index != -1:
        # Extract the text within the brackets
        extracted_mat = "[" + compatibility_mat.content[opening_bracket_index + 1: closing_bracket_index] + "]]"
        # Return the evaluated mat
        # check to see if compatiblity matrix only contains values of -1, 0, or 1
        if eval(extracted_mat).count('0') + eval(extracted_mat).count('1') == len(eval(extracted_mat)):
            # continue
            pass
        else:
            # try again up to 5 times
            for i in range(5):
                print("Error with parsing plant compatibility matrix. Trying for attempt #" + str(i+1))
                print(extracted_mat)
                extracted_mat = chat_response(template + "remember, it MUST ONLY CONTAIN -1s, 0s, and 1s, like this structure: [[0,1],[1,0]]", text)
                # Extract the text within the brackets
                extracted_mat = "[" + compatibility_mat.content[opening_bracket_index + 1: closing_bracket_index] + "]]"
                print(extracted_mat)
                total_count = 0
                count_0 = extracted_mat.count("0")
                count_1 = extracted_mat.count("1")
                total_count = count_0 + count_1
                print("matrix count of -1, 0, 1: ", total_count)
                # if count euals the number of plants squared, then we have a valid matrix
                print("plant_list_len: ", len(plant_list)**2)
                if total_count == (len(plant_list))**2:
                #if count == eval(extracted_mat):
                    print("success")
                    return eval(extracted_mat)
                    break

        
    else:
        print("Error with parsing plant compatibility matrix")
        # try again up to 5 times
        for i in range(5):
            print("Error with parsing plant compatibility matrix. Trying for attempt #" + str(i+1))
            extracted_mat = chat_response(template + "remember, it MUST ONLY CONTAIN -1s, 0s, and 1s, like this structure: [[0,1],[1,0]]", text)
            # Extract the text within the brackets
            extracted_mat = "[" + compatibility_mat.content[opening_bracket_index + 1: closing_bracket_index] + "]]"
            print(extracted_mat)
            total_count = 0
            count_0 = extracted_mat.count("0")
            count_1 = extracted_mat.count("1")
            total_count = count_0 + count_1
            print("matrix count of -1, 0, 1: ", total_count)
            # if count euals the number of plants squared, then we have a valid matrix
            print("plant_list_len: ", len(plant_list)**2)
            if total_count == (len(plant_list))**2:
            #if count == eval(extracted_mat):
                print("success")
                return eval(extracted_mat)
                break

        return None
    
# get compatability matrix for companion planting via subsetting a hardcoded matrix
# make plant_compatibility.csv into a matrix. it currently has indexes as rows and columns for plant names and then compatibility values as the values
plant_compatibility = pd.read_csv('data/plant_compatibility.csv', index_col=0)

def get_compatibility_matrix_2(plant_list):

    # Subset the matrix to only include the plants in the user's list
    plant_compatibility = st.session_state.raw_plant_compatibility.loc[plant_list, plant_list]

    # full matrix
    full_mat = st.session_state.raw_plant_compatibility.to_numpy()

    # Convert the DataFrame to a NumPy array
    plant_compatibility_matrix = plant_compatibility.to_numpy()

    # Get the list of original indices (from the DataFrame)
    original_indices = plant_compatibility.index.tolist()
    
    # Create a dictionary to map plant names to their original indices
    plant_index_mapping = {plant: index for index, plant in enumerate(original_indices)}

    # Return the matrix and the plant-index mapping
    return plant_compatibility_matrix, full_mat, plant_index_mapping

# get plant groupings from LLM
def get_seed_groupings_from_LLM():
    template="You are a helpful assistant that only outputs python lists of lists of lists of plants."
    # make sure output is strictly and only a list of lists for one grouping
    text ='''I am working on a gardening project and need to optimally group a set of plants based on their compatibility. Below is the compatibility matrix for the plants, where each value represents how well two plants grow together (positive values indicate good compatibility, negative values indicate poor compatibility). I also have specific constraints for planting: there are a certain number of plant beds (n_plant_beds), each bed can have a minimum of min_species species and a maximum of max_species species. Given these constraints, please suggest several groupings of these plants into n_plant_beds beds, optimizing for overall compatibility.

        Number of Plant Beds: ''' + str(st.session_state.n_plant_beds) + '''
        Minimum Species per Bed: ''' + str(st.session_state.min_species) + '''
        Maximum Species per Bed: ''' + str(st.session_state.max_species) + '''
        Plants and Compatibility Matrix:'''+ str(st.session_state.raw_plant_compatibility.loc[st.session_state.input_plants_raw, st.session_state.input_plants_raw]) + '''

        Please provide a grouping that maximize positive interactions within each bed and minimize negative interactions, adhering to the specified bed constraints. Return a list of lists  where each list represents an iteration of plant groupings. Each list within the list represents a bed, and each list within the bed represents the plants in that bed.
        sample output: [['plant1', 'plant2'] #bed1, ['plant3', 'plant4'] #bed2, ['plant1', 'plant3'] #bed3]
        another sample output: [['plant1', 'plant2', 'plant3'] #bed1, ['plant4', 'plant5', 'plant6'] #bed2, ['plant7', 'plant8', 'plant9'] #bed3]
        Note: the number of beds, the number of plants per bed, and the number of plants in the list may vary.
        Note: only output a python list of lists of plants. Do not include any other text or explanation.

        '''


    plant_groupings = chat_response(template, text, model)
    print('response about LLMs choice on groupings', plant_groupings.content)

    # try to eval the string to a list of lists
    try:
        plant_groupings_evaluated = eval(plant_groupings.content)
        # check type of output
        print(type(plant_groupings_evaluated))
        # we expect a list of lists
    except:
        print("Error with parsing plant groupings")
        # try again up to 5 times
        for i in range(5):
            print("Error with parsing plant groupings. Trying for attempt #" + str(i+1))
            plant_groupings = chat_response(template, text)
            print(plant_groupings.content)
            # try to eval the string to a list of lists
            try:
                plant_groupings_evaluated = eval(plant_groupings.content)
                break
            except:
                print("Error with parsing plant groupings")
                continue

    return plant_groupings_evaluated
