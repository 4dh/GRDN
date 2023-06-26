# import libraries
import pandas as pd
import numpy as np
import os
import time
import math
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space


# import modules
from backend.chatbot import *
from backend.optimization_algo import *
from frontend.visualizations import *

# setup keys and api info
file_path = '/Users/dheym/Library/CloudStorage/OneDrive-Personal/Documents/side_projects/api_keys/openai_api_keys.txt'
with open(file_path, 'r') as file:
    OPENAI_API_KEY = file.read()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

chat = ChatOpenAI()

from PIL import Image

# UI page config
st.set_page_config(
    page_title="GRDN.AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Function to display chat message with an icon
def chat_message(message, is_user=False):
    if is_user:
        icon = Image.open("assets/cool.png") 
        side = "left"
    else:
        icon = Image.open("assets/bot.png")  
        side = "right"
    
    chat_container = st.container()

    with chat_container:
        col1, col2, col3, col4= st.columns([1,7,7,1])

        with col1:
            if is_user == True:
                st.image(icon, width=50)

        with col2:
            if is_user == True:
                st.markdown(f'<div style="text-align: {side};">{message}</div>', unsafe_allow_html=True)
        with col3:
            if is_user == False:
                st.markdown(f'<div style="text-align: {side};">{message}</div>', unsafe_allow_html=True)
        with col4:
            if is_user == False:
                st.image(icon, width=50)



col1, col2 = st.columns([3,10])
with col1:
     st.image ("assets/flower_tech.png", caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
with col2:
    # add vertical space
    add_vertical_space(3)
    st.title("GRDN.AI")
st.write("AI and optimization powered companion gardening")
colored_header(label='', description='', color_name='green-30')

# Sidebar
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Select a page:", ("Home", "Companion Gardening", "Optimization", "About"))

# add vertical space
with st.sidebar:
    add_vertical_space(2)
# Sidebar
st.sidebar.title("Navigation")

# Define the page options
pages = ["Garden Optimization", "About"]

# Render the selected page content
page = st.sidebar.selectbox("Select a page:", pages)


if page == "Garden Optimization":
    st.sidebar.subheader("Companion Gardening")
    st.write("GRDN is a companion gardening app that helps you plan your garden and maximize your harvest. It uses AI to predict the best plants to grow together and optimization algorithms to optimize how you build your garden.")
    st.write("This app is currently in beta. Please report any bugs.")
    companion_planting_info = """
    Key Benefits 
    - **Pest control:** 
    - **Improved pollination:** 
    - **Maximized space:** 
    - **Nutrient enhancement:** 
    - **Complementary growth:** 

    """

    st.sidebar.markdown(companion_planting_info)
    # Set the initial value of user_name
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ''
    # add in some vertical space
    add_vertical_space(3)
    st.title('GRDN.AI GrowBot: Let\'s get started! 💬')
    
    
    # add in some vertical space
    add_vertical_space(2)

    # Modify the user_name variable based on user input
    if st.session_state['user_name'] == '':
        col1, col2, col3= st.columns([1,2,1])
        with col1:
            st.session_state['user_name_input'] = st.text_input('Enter your name', st.session_state.user_name)
    if 'user_name_input' in st.session_state:
        st.session_state.user_name = st.session_state.user_name_input
        chat_message(st.session_state['user_name'], is_user=True)
            
        # ## past stores User's questions
        # if 'past' not in st.session_state:
        #     st.session_state['past'] = [st.session_state.user_name]



    

    # Layout of input/response containers
    response_container = st.container()
    print("")
    print("____________________")
    print("start of session")
    # User input
    ## Function for taking user provided prompt as input
    def get_text():
        input_text = st.text_input("You: ", "", key="input")
        print(input_text)
        return input_text
    
    ## AI workflow to chat with user, ask them what plants they would like to plant, ask them 
    ## how many plant beds, determine the min and max species per plantbed, and generate a companion plant compatibility matrix
    
    with response_container:
    ## generated stores AI generated responses
        if st.session_state['user_name'] != '':
            # ask the user what they would like to plant
            st.session_state['generated'] = ["Hello " + st.session_state['user_name'] + "! I'm GrowBot, a garden specialist bot!"]
            st.session_state.generated.append("To begin building your garden, let\'s gather some data. Tell me, what plants would you like to plant in your garden?")
            chat_message(st.session_state['generated'][0])
            time.sleep(1)
            chat_message(st.session_state['generated'][1])
            time.sleep(1)
     
            col1, col2, col3= st.columns([1,2,1])
            with col1:
                if 'input_plants_raw' not in st.session_state:
                    input_plants_raw = st.multiselect('plants', ['Apricot',
                                                        'Apple',
                                                        'Asparagus',
                                                        'Basil',
                                                        'Beans',
                                                        'Broad Beans',
                                                        'Bush Beans',
                                                        'Climbing Beans',
                                                        'Beets',
                                                        'Borage',
                                                        'Broccoli',
                                                        'Brussel Sprouts',
                                                        'Cabbages',
                                                        'Chamomile',
                                                        'Carrots',
                                                        'Cauliflower',
                                                        'Celery',
                                                        'Cherry',
                                                        'Chervil',
                                                        'Chives',
                                                        'Coriander',
                                                        'Corn',
                                                        'Cucumber',
                                                        'Dill',
                                                        'Eggplant',
                                                        'Fennel',
                                                        'Marigold',
                                                        'Fruit Trees',
                                                        'Garlic',
                                                        'Gooseberry',
                                                        'Grape Vine',
                                                        'Grass',
                                                        'Horseradish',
                                                        'Lavendar',
                                                        'Leeks',
                                                        'Lemon Balm',
                                                        'Lettuce',
                                                        'Marjoram',
                                                        'Mints',
                                                        'Mulberry',
                                                        'Mustard',
                                                        'Nasturtiums',
                                                        'Onions',
                                                        'Parsley',
                                                        'Parsnip',
                                                        'Peas',
                                                        'Pennyroyal',
                                                        'Potato',
                                                        'Pumpkin',
                                                        'Radish',
                                                        'Raspberry',
                                                        'Rosemary',
                                                        'Roses',
                                                        'Rue',
                                                        'Sage',
                                                        'Savory',
                                                        'Shallots',
                                                        'Silverbeet',
                                                        'Spinach',
                                                        'Squash',
                                                        'Strawberries',
                                                        'Stinging Nettle',
                                                        'Sunflower',
                                                        'Tansy',
                                                        'Thyme',
                                                        'Tomato',
                                                        'Yarrow',
                                                        'Zucchini',
                                                        ])
                    if st.button('continue '):
                        st.session_state['input_plants_raw'] = input_plants_raw
            if 'input_plants_raw' in st.session_state:
                print("BP1")
                # first question is what plants would you like to plant
                plants_response = st.session_state.input_plants_raw
                chat_message(st.session_state.input_plants_raw, is_user=True)
                #st.write(plants_response)
                time.sleep(1)
                chat_message("How many plant beds would you like to have in your garden?")
                col1, col2, col3= st.columns([1,2,1])
                with col1:
                    if 'n_plant_beds' not in st.session_state:
                        # create a streamlit form to get the number of plant beds
                        n_plant_beds = st.number_input('Number of plant beds', min_value=1, max_value=20, value=1, step=1)
                        if st.button('continue'):
                            st.session_state.n_plant_beds = n_plant_beds = n_plant_beds
                    # 1 <= min_species <= max_species <= len(input_plants_raw)
                if 'n_plant_beds' in st.session_state:
                    chat_message("Number of plant beds: " + str(st.session_state.n_plant_beds), is_user=True)
                    col1, col2, col3= st.columns([1,2,1])
                    with col1:
                        if 'min_species' not in st.session_state and 'n_plant_beds' in st.session_state:
                            min_species = st.number_input('Minimum number of species per plant bed', min_value=1, max_value=len(st.session_state.input_plants_raw), value=1, step=1)
                            if st.button('continue  '):
                                st.session_state.min_species = min_species  
                if 'min_species' in st.session_state:
                    chat_message("Minimum number of species per plant bed: " + str(st.session_state.min_species), is_user=True)
                    # max_species >= min_species & max_species >= floor(length(input_plants_raw)-(min_species*n_plant_beds-1) & max_species_per_bed <= len(input_plants_raw)  
                    col1, col2, col3= st.columns([1,2,1])
                    with col1:
                        if 'max_species' not in st.session_state and 'min_species' in st.session_state:
                            # work on the math here
                            #max_species = st.number_input('Maximum number of species per plant bed', min_value=max(st.session_state.min_species, (math.floor(len(st.session_state.input_plants_raw)-(st.session_state.min_species*st.session_state.n_plant_beds-1)))), max_value=len(st.session_state.input_plants_raw), value=2, step=1)
                            max_species = st.number_input('Maximum number of species per plant bed', min_value=st.session_state.min_species, max_value=len(st.session_state.input_plants_raw), value=2, step=1)
                            if st.button(' continue '):
                                st.session_state.max_species = max_species
                if 'max_species' in st.session_state:
                    chat_message("Maximum number of species per plant bed: " + str(st.session_state.max_species), is_user=True)
                if 'max_species' in st.session_state:
                    print("BP2")
                    chat_message("Excellent! A garden of " + str(st.session_state.n_plant_beds) + " plant beds with " + str(st.session_state.min_species) + " min species per plant bed and " + str(st.session_state.max_species) + " max species per plant bed is a great choice. I'm now going to come up with a companion plant compatibility matrix and network visualization for you based on your plant list.")
                    col1, col2, col3= st.columns([1,2,3])
                    with col3:
                        if 'extracted_mat' not in st.session_state:
                            if st.button(' continue'):
                                with st.spinner('generating companion plant compatibility matrix...'):
                                    st.session_state['generating_mat'] = True
                                    # now get compatibility matrix for companion planting
                                    extracted_mat = get_compatibility_matrix(st.session_state.input_plants_raw)
                                    print(extracted_mat)
                                    st.session_state.extracted_mat = extracted_mat
                    if 'extracted_mat' in st.session_state:
                        chat_message("Here is your companion plant compatibility matrix:")
                        col1, col2, col3= st.columns([8,6,1])
                        with col2:
                            with st.expander("Show ugly compatibility matrix of 1's 0's and -1's"):
                                st.write(st.session_state.extracted_mat)
                        chat_message("Here is a network visualization of your companion plant compatibility matrix. It is color coded to show which plants are companions (green), antagonists (violetred), or neutral (grey).")
                        plot_compatibility(st.session_state.input_plants_raw, st.session_state.extracted_mat)
                        chat_message("Now that we have your companion plant compatibility matrix, we can use optimization to maximize your harvest.")
                        chat_message("We will use a genetic algorithm to determine the best way to plant your garden.")
                        chat_message("The genetic algorithm will determine the best way to plant your garden by maximizing the number of companion plants and minimizing the number of antagonists.")
                        chat_message("Set the parameters for the genetic algorithm. Here is more info for your reference:")
                        col1, col2, col3= st.columns([1,2,3])
                        with col3:
                            with st.expander("Show more information about the genetic algorithm parameters"):
                                st.subheader("Plant Optimization Heuristic Performance")
                                st.write("The genetic algorithm parameters impact the performance of the plant optimization heuristic in the following ways:")
                                st.markdown("- **Population Size**: A larger population size allows for a more diverse exploration of the solution space. However, it also increases computational complexity.")
                                st.markdown("- **Number of Generations**: Increasing the number of generations provides more opportunities for the algorithm to converge towards an optimal solution.")
                                st.markdown("- **Tournament Size**: A larger tournament size promotes stronger selection pressure and can lead to faster convergence, but it may also increase the risk of premature convergence.")
                                st.markdown("- **Crossover Rate**: A higher crossover rate increases the exploration capability by creating diverse offspring, potentially improving the algorithm's ability to escape local optima.")
                                st.markdown("- **Mutation Rate**: Mutation introduces random changes in individuals, helping to maintain diversity in the population and preventing premature convergence.")

                        # Run the Genetic Algorithm
                        col1, col2, col3= st.columns([2,1,1])
                        with col1:
                            st.subheader("Genetic Algorithm Parameters")
                            st.write("These parameters control the behavior of the genetic algorithm.")

                            # Genetic Algorithm parameters
                            st.session_state.population_size = st.slider("Population Size", min_value=10, max_value=1000, value=50,
                                                        help="The number of individuals in each generation of the genetic algorithm.")
                            st.session_state.num_generations = st.slider("Number of Generations", min_value=10, max_value=1000, value=100,
                                                        help="The total number of generations to evolve through.")
                            st.session_state.tournament_size = st.slider("Tournament Size", min_value=2, max_value=10, value=3,
                                                        help="The number of individuals competing in each tournament selection round.")
                            st.session_state.crossover_rate = st.slider("Crossover Rate", min_value=0.1, max_value=1.0, step=0.1, value=0.8,
                                                    help="The probability of two individuals undergoing crossover to create offspring.")
                            st.session_state.mutation_rate = st.slider("Mutation Rate", min_value=0.01, max_value=0.1, step=0.01, value=0.05,
                                                    help="The probability of an individual undergoing mutation.")

                            # Run the genetic algorithm
                            if st.button("Run Genetic Algorithm"):
                                grouping = genetic_algorithm_plants()
                                st.session_state.grouping = grouping
                        # visualize the groupings
                        chat_message("Here are the optimized groupings of plants for your garden:")
                        if 'grouping' in st.session_state:
                            visualize_groupings()
                            # show plant care tips
                            chat_message("Here are some plant care tips for your plants. Good luck!")
                            col1, col2, col3= st.columns([1,2,3])
                            with col3:
                                with st.spinner('generating plant care tips...'):
                                    st.session_state.plant_care_tips = get_plant_care_tips(st.session_state.input_plants_raw)
                            styled_text = f'<div style="background-color: #2d5a59; color: white; padding: 10px; border-radius: 5px;">{st.session_state.plant_care_tips}</div>'
                            col1, col2, col3= st.columns([6,10,1])
                            with col2:
                                st.write(styled_text, unsafe_allow_html=True)


if page == "About":
    st.sidebar.subheader("About")
    st.sidebar.write("GRDN is a companion gardening app that helps you plan your garden and maximize your harvest. It uses AI to predict the best plants to grow together and optimization algorithms to optimize how you build your garden.")
    st.sidebar.write("Companion gardening is the practice of planting different plants together to maximize their growth. Companion gardening can help to increase the yield of your garden, improve the health of your plants, and reduce the need for pesticides.")
    st.write("This app is currently in beta. Please report any bugs to the team.")

    col1, col2= st.columns([1,1])
    with col1:  
        st.subheader("Contact Information")
        st.write("Author: Danielle Heymann")
        st.write("Email: dheymann314@gmail.com")
        st.write("LinkedIn: https://www.linkedin.com/in/danielle-heymann/")
    with col2:
        st.subheader("Software, data, and libraries used")
        st.write("Libraries and Software")
        st.markdown("""
        - Python
        - streamlit
        - openai
        - plotly
        - pandas
        - numpy
        - PIL
        - langchain
        - streamlit_chat
        - github copilot
        - chatGPT
        - GPT family of models
        """)
        st.write("Data sources in addition to what GPT was trained on: \n https://waldenlabs.com/the-ultimate-companion-planting-guide-chart/ ")

        st.write("avatars from: https://www.flaticon.com/free-icons/bot")
