import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


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
    
def chat_response(template, prompt_text):
    chat = ChatOpenAI(temperature=.1)
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    response = chat(chat_prompt.format_prompt(text= prompt_text).to_messages())
    return response

# get the plant list from user input
def get_plant_list(input_plant_text):
    chat = ChatOpenAI(temperature=.1)
    template="You are a helpful assistant that knows all about gardening and plants and python data structures."
    text = 'which of the elements of this list can be grown in a garden, [' + input_plant_text + ']? Return JUST a python list object containing the elements that can be grown in a garden. Do not include any other text or explanation.'
    plant_list_text = chat_response(template, text)
    plant_list = parse_and_evaluate_text(plant_list_text.content)
    print(plant_list)
    return plant_list

# get plant care tips based on plant list
def get_plant_care_tips(plant_list):
    chat = ChatOpenAI(temperature=.1)
    template="You are a helpful assistant that knows all about gardening, plants, and companion planting."
    text = 'from this list of plants, [' + str(st.session_state.input_plants_raw) + '], generate a list of up to 10 plant care tips or interesting stories of plant compatibility for the plants in the list- maybe 1-2 per plant depending on what you know. Return just the plant care tips in HTML markdown format. Do not include any other text or explanation. It must be in HTML markdown format.'
    plant_care_tips = chat_response(template, text)
    print(plant_care_tips.content)
    return plant_care_tips.content

# get compatability matrix for companion planting
def get_compatibility_matrix(plant_list):
    chat = ChatOpenAI(temperature=.1)
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
    