import streamlit as st
import pandas as pd
import os
import subprocess
import time

# Lazy imports - only load when actually needed (saves 5-10 seconds on startup)
def _lazy_import_llm_libs():
    """Import heavy LLM libraries only when needed"""
    global ChatOpenAI, ChatPromptTemplate, SystemMessagePromptTemplate
    global AIMessagePromptTemplate, HumanMessagePromptTemplate
    global SimpleDirectoryReader, VectorStoreIndex
    global LlamaCPP, messages_to_prompt, completion_to_prompt
    
    from langchain_community.chat_models import ChatOpenAI
    from langchain_core.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        AIMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    from llama_index.core import (
        SimpleDirectoryReader,
        VectorStoreIndex,
    )
    from llama_index.llms.llama_cpp import LlamaCPP
    
    # Try to import prompt utilities (may not exist in newer versions)
    try:
        from llama_index.llms.llama_cpp.llama_utils import (
            messages_to_prompt,
            completion_to_prompt,
        )
    except ImportError:
        messages_to_prompt = None
        completion_to_prompt = None

# set version
# st.session_state.demo_lite = False

# initialize model
# llm = "tbd"

print("BP 4 ")


# GPU detection and environment configuration
def detect_gpu_and_environment():
    """
    Detect if GPU is available and if running on HuggingFace Spaces
    Returns: dict with gpu_available, is_hf_space, and n_gpu_layers
    """
    config = {
        "gpu_available": False,
        "is_hf_space": False,
        "n_gpu_layers": 0,
        "model_base_path": "/Users/dheym/Library/CloudStorage/OneDrive-Personal/Documents/side_projects/GRDN/src/models"
    }
    
    # Check if running on HuggingFace Spaces
    if os.environ.get("SPACE_ID") or os.environ.get("SPACE_AUTHOR_NAME"):
        config["is_hf_space"] = True
        config["model_base_path"] = "/home/user/app/src/models"  # HF Spaces absolute path
        print("🤗 Running on HuggingFace Spaces")
    
    # Try to detect GPU using torch
    try:
        import torch
        if torch.cuda.is_available():
            config["gpu_available"] = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            config["n_gpu_layers"] = -1  # -1 means offload all layers to GPU
            print(f"🚀 GPU detected: {gpu_name} with {gpu_memory:.2f} GB memory")
            print(f"🚀 Will offload all layers to GPU (n_gpu_layers=-1)")
        else:
            print("⚠️ No GPU detected via torch.cuda")
            config["n_gpu_layers"] = 0
    except ImportError:
        print("⚠️ torch not available, checking alternative methods...")
        # Alternative: check nvidia-smi or environment variables
        if os.path.exists("/usr/bin/nvidia-smi") or os.environ.get("CUDA_VISIBLE_DEVICES"):
            config["gpu_available"] = True
            config["n_gpu_layers"] = -1  # Offload all layers
            print("🚀 GPU likely available (nvidia-smi or CUDA env detected)")
        else:
            config["n_gpu_layers"] = 0
    
    # If on HF Spaces but GPU not detected via torch, still try GPU layers
    if config["is_hf_space"] and not config["gpu_available"]:
        print("🤗 On HF Spaces - attempting GPU acceleration anyway")
        config["gpu_available"] = True
        config["n_gpu_layers"] = -1
    
    return config


# initialize model- get llm depending on st.session_state.demo_lite, and model
def init_llm(model, demo_lite):
    # st.write("BP 4.1: model: ", model)
    if demo_lite == False:
        print("BP 5 : running full demo")
        
        # Load heavy LLM libraries now (lazy import)
        _lazy_import_llm_libs()
        
        # Detect GPU and environment
        env_config = detect_gpu_and_environment()
        n_gpu_layers = env_config["n_gpu_layers"]
        model_base_path = env_config["model_base_path"]
        
        if env_config["gpu_available"]:
            print(f"✅ GPU acceleration ENABLED with {n_gpu_layers} layers")
        else:
            print("⚠️ Running on CPU (no GPU detected)")
        
        # Only Llama 3.2-1B is supported (legacy models removed for simplicity)
        model_path = os.path.join(model_base_path, "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
        print(f"Loading Llama 3.2-1B from: {model_path}")
        
        # Check if model exists
        if not os.path.exists(model_path):
            error_msg = f"⚠️ Model not found at {model_path}"
            if env_config["is_hf_space"]:
                error_msg += ". Please ensure the model file is uploaded to your HuggingFace Space."
            st.error(error_msg)
            print(f"❌ {error_msg}")
            return None
        
        # Initialize Llama 3.2-1B with GPU support
        llm = LlamaCPP(
            model_path=model_path,
            temperature=0.1,
            max_new_tokens=1500,
            context_window=8192,  # Llama 3.2 supports 128K context
            generate_kwargs={},
            model_kwargs={"n_gpu_layers": n_gpu_layers},
            verbose=True,
        )
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
    else:
        # Use Llama 3.2-1B (only supported model)
        print("Using Llama 3.2-1B")
        if "llm" not in st.session_state:
            st.session_state.llm = init_llm(model, demo_lite)
        if st.session_state.llm is None:
            return "Error: Could not initialize LLM. Please check the logs."
        response = st.session_state.llm.complete(template + prompt_text)
        return response.text


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

    plant_care_tips = chat_response(template, text, model, demo_lite)
    # check to see if response contains ### or < for headers
    print("BP6", plant_care_tips)
    # st.write(plant_care_tips)
    
    # Safety check for None response
    if plant_care_tips is None:
        return "Error: Could not generate plant care tips. Please try again or select a different model."
    
    if "###" not in plant_care_tips and "<" not in plant_care_tips:
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
