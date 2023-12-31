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
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
import subprocess
import time

# set version
# st.session_state.demo_lite = False

# initialize model
# llm = "tbd"

print("BP 4 ")


# initialize model- get 11m depending on st.session_state.demo_lite, and model
def init_llm(model, demo_lite):
    # st.write("BP 4.1: model: ", model)
    if demo_lite == False:
        print("BP 5 : running full demo")
        if model == "Llama2-7b_CPP":
            model_path = "/Users/dheym/Library/CloudStorage/OneDrive-Personal/Documents/side_projects/GRDN/src/models/llama-2-7b-chat.Q4_K_M.gguf"
            print("model path: ", model_path)
            llm = LlamaCPP(
                # You can pass in the URL to a GGML model to download it automatically
                # model_url=model_url,
                # optionally, you can set the path to a pre-downloaded model instead of model_url
                model_path=model_path,
                temperature=0.1,
                max_new_tokens=1000,
                # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
                context_window=3000,
                # kwargs to pass to __call__()
                generate_kwargs={},
                # kwargs to pass to __init__()
                # set to at least 1 to use GPU
                model_kwargs={"n_gpu_layers": 10},
                # transform inputs into Llama2 format
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                verbose=True,
            )
        elif model == "deci-7b_CPP":
            model_path = "/Users/dheym/Library/CloudStorage/OneDrive-Personal/Documents/side_projects/GRDN/src/models/decilm-7b-uniform-gqa-q8_0.gguf"
            print("model path: ", model_path)
            llm = LlamaCPP(
                # You can pass in the URL to a GGML model to download it automatically
                # model_url=model_url,
                # optionally, you can set the path to a pre-downloaded model instead of model_url
                model_path=model_path,
                # model_url = "https://huggingface.co/Deci/DeciLM-7B-instruct-GGUF/resolve/main/decilm-7b-uniform-gqa-q8_0.gguf",
                temperature=0.1,
                max_new_tokens=1000,
                # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
                context_window=3000,
                # kwargs to pass to __call__()
                generate_kwargs={},
                # kwargs to pass to __init__()
                # set to at least 1 to use GPU
                model_kwargs={"n_gpu_layers": 1},
                # transform inputs into Llama2 format
                # messages_to_prompt=messages_to_prompt,
                # completion_to_prompt=completion_to_prompt,
                verbose=True,
            )
        else:
            print("Error with chatbot model")
            return None
        return llm


def parse_and_evaluate_text(text):
    # Find the indices of the opening and closing brackets
    opening_bracket_index = text.find("[")
    closing_bracket_index = text.find("]")

    if opening_bracket_index != -1 and closing_bracket_index != -1:
        # Extract the text within the brackets
        extracted_list = (
            "[" + text[opening_bracket_index + 1 : closing_bracket_index] + "]"
        )
        # Return the evaluated text list
        return eval(extracted_list)

    else:
        print("Error with parsing plant list")
        return None


def chat_response(template, prompt_text, model, demo_lite):
    if model == "openai-gpt35turbo":
        chat = ChatOpenAI(temperature=0.1)
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        response = chat(chat_prompt.format_prompt(text=prompt_text).to_messages())

        return response
        # return response.content
    elif model == "Llama2-7b_CPP" or model == "deci-7b_CPP":
        print("BP 5.1: running full demo, model: ", model)
        if "llm" not in st.session_state:
            st.session_state.llm = init_llm(model, demo_lite)
        response = st.session_state.llm.complete(template + prompt_text)
        return response.text
    else:
        print("Error with chatbot model")
        return None


# # get the plant list from user input
# def get_plant_list(input_plant_text, model):
#     template="You are a helpful assistant that knows all about gardening and plants and python data structures."
#     text = 'which of the elements of this list can be grown in a garden, [' + input_plant_text + ']? Return JUST a python list object containing the elements that can be grown in a garden. Do not include any other text or explanation.'
#     plant_list_text = chat_response(template, text, model)
#     plant_list = parse_and_evaluate_text(plant_list_text.content)
#     print(plant_list)
#     return plant_list


# get plant care tips based on plant list
def get_plant_care_tips(plant_list, model, demo_lite):
    plant_care_tips = ""
    template = "You are a helpful assistant that knows all about gardening, plants, and companion planting."
    text = (
        "from this list of plants, ["
        + str(st.session_state.input_plants_raw)
        + "], generate 1-2 plant care tips for each plant based on what you know. Return just the plant care tips in HTML markdown format. Make sure to use ### for headers. Do not include any other text or explanation before or after the markdown. It must be in HTML markdown format."
    )

    if model == "deci-7b_CPP":
        template = (
            "### System: \n\n You are a helpful assistant that knows all about gardening, plants, and companion planting."
            + "\n\n ### User: Generate gardening tips. Return just the plant care tips in HTML markdown format. Make sure to use ### for headers. Do not include any other text or explanation before or after the markdown. It must be in HTML markdown format. \n\n"
        )
        text = "### Assistant: \n\n"
        print("deci-7b_CPP")
    plant_care_tips = chat_response(template, text, model, demo_lite)
    # check to see if response contains ### or < for headers
    print("BP6", plant_care_tips)
    # st.write(plant_care_tips)
    if (
        "###" not in plant_care_tips
        and "<" not in plant_care_tips
        and model != "deci-7b_CPP"
    ):  # deci-7b_CPP has more general plant care tips
        st.write(plant_care_tips)
        print("Error with parsing plant care tips")
        # try again up to 5 times
        for i in range(5):
            print(
                "Error with parsing plant care tips. Trying for attempt #" + str(i + 1)
            )
            plant_care_tips = chat_response(template, text, model, demo_lite)
            # check to see if response contains ### for headers
            if "###" not in plant_care_tips and "<" not in plant_care_tips:
                continue
            else:
                break
    # remove any text before the first ### or < in the response
    print(plant_care_tips)
    # look for either # or < for headers
    if "###" in plant_care_tips:
        plant_care_tips = "\n\n" + plant_care_tips[plant_care_tips.find("###") :]
    elif "<" in plant_care_tips:
        plant_care_tips = "\n\n" + plant_care_tips[plant_care_tips.find("<") :]
    else:
        print("funky formatting")
        plant_care_tips = plant_care_tips
    print(plant_care_tips)
    return plant_care_tips


# get compatability matrix for companion planting
def get_compatibility_matrix(plant_list, model, demo_lite):
    # Convert the compatibility matrix to a string
    with open("data/compatibilities_text.txt", "r") as file:
        # Read the contents of the file
        compatibility_text = file.read()
    plant_comp_context = compatibility_text
    template = "You are a helpful assistant that knows all about gardening, companion planting, and python data structures- specifically compatibility matrices."
    text = (
        "from this list of plants, ["
        + str(plant_list)
        + "], Return JUST a python array (with values separated by commas like this: [[0,1],[1,0]]\n\n ) for companion plant compatibility. Each row and column should represent plants, and the element of the array will contain a -1, 0, or 1 depending on if the relationship between plants is antagonists, neutral, or companions, respectively. You must refer to this knowledge base of information on plant compatibility: \n\n, "
        + plant_comp_context
        + "\n\n A plant's compatibility with itself is always 0. Do not include any other text or explanation."
    )
    compatibility_mat = chat_response(template, text, model, demo_lite)

    # Find the indices of the opening and closing brackets
    opening_bracket_index = compatibility_mat.content.find("[[")
    closing_bracket_index = compatibility_mat.content.find("]]")
    if opening_bracket_index != -1 and closing_bracket_index != -1:
        # Extract the text within the brackets
        extracted_mat = (
            "["
            + compatibility_mat.content[
                opening_bracket_index + 1 : closing_bracket_index
            ]
            + "]]"
        )
        # Return the evaluated mat
        # check to see if compatiblity matrix only contains values of -1, 0, or 1
        if eval(extracted_mat).count("0") + eval(extracted_mat).count("1") == len(
            eval(extracted_mat)
        ):
            # continue
            pass
        else:
            # try again up to 5 times
            for i in range(5):
                print(
                    "Error with parsing plant compatibility matrix. Trying for attempt #"
                    + str(i + 1)
                )
                print(extracted_mat)
                extracted_mat = chat_response(
                    template
                    + "remember, it MUST ONLY CONTAIN -1s, 0s, and 1s, like this structure: [[0,1],[1,0]]",
                    text,
                    model,
                    demo_lite,
                )
                # Extract the text within the brackets
                extracted_mat = (
                    "["
                    + compatibility_mat.content[
                        opening_bracket_index + 1 : closing_bracket_index
                    ]
                    + "]]"
                )
                print(extracted_mat)
                total_count = 0
                count_0 = extracted_mat.count("0")
                count_1 = extracted_mat.count("1")
                total_count = count_0 + count_1
                print("matrix count of -1, 0, 1: ", total_count)
                # if count euals the number of plants squared, then we have a valid matrix
                print("plant_list_len: ", len(plant_list) ** 2)
                if total_count == (len(plant_list)) ** 2:
                    # if count == eval(extracted_mat):
                    print("success")
                    return eval(extracted_mat)
                    break

    else:
        print("Error with parsing plant compatibility matrix")
        # try again up to 5 times
        for i in range(5):
            print(
                "Error with parsing plant compatibility matrix. Trying for attempt #"
                + str(i + 1)
            )
            extracted_mat = chat_response(
                template
                + "remember, it MUST ONLY CONTAIN -1s, 0s, and 1s, like this structure: [[0,1],[1,0]]",
                text,
                model,
                demo_lite,
            )
            # Extract the text within the brackets
            extracted_mat = (
                "["
                + compatibility_mat.content[
                    opening_bracket_index + 1 : closing_bracket_index
                ]
                + "]]"
            )
            print(extracted_mat)
            total_count = 0
            count_0 = extracted_mat.count("0")
            count_1 = extracted_mat.count("1")
            total_count = count_0 + count_1
            print("matrix count of -1, 0, 1: ", total_count)
            # if count euals the number of plants squared, then we have a valid matrix
            print("plant_list_len: ", len(plant_list) ** 2)
            if total_count == (len(plant_list)) ** 2:
                # if count == eval(extracted_mat):
                print("success")
                return eval(extracted_mat)
                break

        return None


# get compatability matrix for companion planting via subsetting a hardcoded matrix
# make plant_compatibility.csv into a matrix. it currently has indexes as rows and columns for plant names and then compatibility values as the values
plant_compatibility = pd.read_csv("src/data/plant_compatibility.csv", index_col=0)


def get_compatibility_matrix_2(plant_list):
    # Subset the matrix to only include the plants in the user's list
    plant_compatibility = st.session_state.raw_plant_compatibility.loc[
        plant_list, plant_list
    ]

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
def get_seed_groupings_from_LLM(model, demo_lite):
    plant_groupings_evaluated = "no response yet"
    if demo_lite:
        # just return "no response yet" for now
        return plant_groupings_evaluated
    template = "You are a helpful assistant that only outputs python lists of lists of lists of plants."
    # make sure output is strictly and only a list of lists for one grouping
    text = (
        """I am working on a gardening project and need to optimally group a set of plants based on their compatibility. Below is the compatibility matrix for the plants, where each value represents how well two plants grow together (positive values indicate good compatibility, negative values indicate poor compatibility). I also have specific constraints for planting: there are a certain number of plant beds (n_plant_beds), each bed can have a minimum of min_species species and a maximum of max_species species. Given these constraints, please suggest several groupings of these plants into n_plant_beds beds, optimizing for overall compatibility.

        Number of Plant Beds: """
        + str(st.session_state.n_plant_beds)
        + """
        Minimum Species per Bed: """
        + str(st.session_state.min_species)
        + """
        Maximum Species per Bed: """
        + str(st.session_state.max_species)
        + """
        Plants and Compatibility Matrix:"""
        + str(
            st.session_state.raw_plant_compatibility.loc[
                st.session_state.input_plants_raw, st.session_state.input_plants_raw
            ]
        )
        + """

        Please provide a grouping that maximize positive interactions within each bed and minimize negative interactions, adhering to the specified bed constraints. Return a list of lists  where each list represents an iteration of plant groupings. Each list within the list represents a bed, and each list within the bed represents the plants in that bed.
        sample output: [['plant1', 'plant2'] #bed1, ['plant3', 'plant4'] #bed2, ['plant1', 'plant3'] #bed3]
        another sample output: [['plant1', 'plant2', 'plant3'] #bed1, ['plant4', 'plant5', 'plant6'] #bed2, ['plant7', 'plant8', 'plant9'] #bed3]
        Note: the number of beds, the number of plants per bed, and the number of plants in the list may vary.
        Note: only output ONE python list of lists of plants. Do not include any other text or explanation.

        """
    )

    plant_groupings = chat_response(template, text, model, demo_lite)
    # check to see if we've cut off the response due to time limit. if so, return "no response yet" for now
    if plant_groupings == None:
        return "no response yet"
    print("response about LLMs choice on groupings", plant_groupings)

    # try to eval the string to a list of lists
    try:
        plant_groupings_evaluated = eval(plant_groupings)
        # check type of output
        print(type(plant_groupings_evaluated))
        # we expect a list of lists
    except:
        print("Error with parsing plant groupings")
        # try again up to 5 times
        for i in range(5):
            print(
                "Error with parsing plant groupings. Trying for attempt #" + str(i + 1)
            )
            plant_groupings = chat_response(template, text, model, demo_lite)
            print(plant_groupings)
            # try to eval the string to a list of lists
            try:
                # make sure plant1 is not in the output
                if "plant1" in plant_groupings.lower():
                    print("plant1 is in the output")
                    continue
                else:
                    plant_groupings_evaluated = eval(plant_groupings)
                    print("successful eval; output: ", plant_groupings_evaluated)
                    break
            except:
                # try to find the list of lists within the string
                opening_bracket_index = plant_groupings.find("[[")
                closing_bracket_index = plant_groupings.find("]]")
                if opening_bracket_index != -1 and closing_bracket_index != -1:
                    # Extract the text within the brackets
                    extracted_list = (
                        "["
                        + plant_groupings[
                            opening_bracket_index + 1 : closing_bracket_index
                        ]
                        + "]]"
                    )
                    # Return the evaluated text list
                    if "plant1" in extracted_list.lower():
                        print("plant1 is in the output")
                        continue
                    else:
                        plant_groupings_evaluated = eval(extracted_list)
                        print("successful eval; output: ", plant_groupings_evaluated)
                        break
                else:
                    print("Error with parsing plant groupings")
                    continue

    return plant_groupings_evaluated
