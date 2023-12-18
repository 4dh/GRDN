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
from PIL import Image

# import modules
from backend.chatbot import *
from backend.optimization_algo import *
from frontend.visualizations import *


# import compatibilities matrix
# make plant_compatibility.csv into a matrix. it currently has indexes as rows and columns for plant names and then compatibility values as the values
st.session_state.raw_plant_compatibility = pd.read_csv('data/plant_compatibility.csv', index_col=0)
# fill NaN values with 0
st.session_state.raw_plant_compatibility = st.session_state.raw_plant_compatibility.fillna(0)
# get list of plants
st.session_state.plant_list = st.session_state.raw_plant_compatibility.index.tolist()



# setup keys and api info
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN



#chat = ChatOpenAI()


# UI page config
st.set_page_config(
    #page_title="GRDN.AI",
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




st.image ("assets/logo_title_transparent.png", caption=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

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
    # Display the welcome message
    st.title('Let\'s get started! Decide on your garden parameters')
    
    
    # add in some vertical space
    add_vertical_space(2)

    # make a container for this section
    container1 = st.container(border=True)

    with container1:
        # Modify the user_name variable based on user input
        if st.session_state['user_name'] == '':
            col1, col2, col3= st.columns([1,2,1])
            with col1:
                st.session_state['user_name_input'] = st.text_input('Enter your name', st.session_state.user_name)
        if 'user_name_input' in st.session_state:
            st.session_state.user_name = st.session_state.user_name_input
            if st.session_state.user_name != '':
                st.write('Hello ' + st.session_state['user_name'] + '! Let\'s optimize your garden. 🌱')

                # # add in some vertical space
                add_vertical_space(2)

                print("")
                print("____________________")
                print("start of session")

                col1a, col2a= st.columns([1,2])
                enable_max_species = False
                enable_min_species = False

                
                # make a form to get the plant list from the user
                with col1a:
                    with st.form(key = "plant_list_form"):
                        input_plants_raw = st.multiselect('plants', st.session_state.plant_list)
                        submit_button = st.form_submit_button(label='Submit Plant List')
                if submit_button:
                    st.session_state['input_plants_raw'] = input_plants_raw
                    st.session_state.submitted_plant_list = True

                # add in some vertical space
                add_vertical_space(1)

                with col2a:
                    col1, col2, col3= st.columns([1,1,1])
                    if 'input_plants_raw' in st.session_state:
                        print("BP1")
                        # first question is what plants would you like to plant
                        plants_response = st.session_state.input_plants_raw

                        # Initialize session state variables if they don't exist
                        if 'n_plant_beds' not in st.session_state:
                            st.session_state['n_plant_beds'] = 1

                        if 'min_species' not in st.session_state:
                            st.session_state['min_species'] = 1

                        if 'max_species' not in st.session_state:
                            st.session_state['max_species'] = 2

                        # Number of plant beds input
                        with col1:
                            n_plant_beds = st.number_input('Number of plant beds \n', min_value=1, max_value=20, value=st.session_state.n_plant_beds, step=1)
                            st.session_state.n_plant_beds = n_plant_beds
                        with col2:
                            # Minimum species per plant bed input
                            min_species = st.number_input('Minimum number of species per plant bed', 
                                                        min_value=1, 
                                                        max_value=len(st.session_state.input_plants_raw), 
                                                        value=st.session_state.min_species, 
                                                        step=1)
                            st.session_state.min_species = min_species

                            # Maximum species per plant bed input
                            # It will be enabled only if min_species is set
                            enable_max_species = st.session_state.min_species > 0
                        with col3:
                            max_species = st.number_input('Maximum number of species per plant bed', 
                                                        min_value=st.session_state.min_species, 
                                                        max_value=len(st.session_state.input_plants_raw), 
                                                        value=max(st.session_state.min_species, st.session_state.max_species), 
                                                        step=1, 
                                                        disabled=not enable_max_species)
                            if enable_max_species:
                                st.session_state.max_species = max_species

                    # extract the compatibility matrix from the user's input
                    if 'extracted_mat' not in st.session_state:
                        valid = False
                        if 'submitted_plant_list' in st.session_state and st.session_state.submitted_plant_list:
                            # check if the user's input is valid
                            # min species per bed must be less than or equal to max species per bed
                            if (st.session_state.min_species <= st.session_state.max_species
                                ) and (
                                # max species per bed must be less than or equal to the number of plants 
                                st.session_state.max_species <= len(st.session_state.input_plants_raw) 
                                ) and (
                                # max species per bed must be greater than or equal to the min species per bed
                                st.session_state.max_species >= st.session_state.min_species
                                ) and (
                                # min species per bed must be less than or equal to the number of plants
                                st.session_state.min_species <= len(st.session_state.input_plants_raw)
                                ) and (
                                # number of plant beds multiplied by min species per bed must be less than or equal to the number of plants
                                len(st.session_state.input_plants_raw) >= st.session_state.n_plant_beds * st.session_state.min_species
                                ) and (
                                # number of plant beds multiplied by max species per bed must be greater than or equal to the number of plants
                                len(st.session_state.input_plants_raw) <= st.session_state.n_plant_beds * st.session_state.max_species
                                ):
                                valid = True  
                            else:
                                # add a warning message
                                st.warning('Please enter valid parameters. The minimum number of species per plant bed must be less than or equal to the maximum number of species per plant bed. The maximum number of species per plant bed must be less than or equal to the number of plants. The maximum number of species per plant bed must be greater than or equal to the minimum number of species per plant bed. The minimum number of species per plant bed must be less than or equal to the number of plants. The number of plant beds multiplied by the minimum number of species per plant bed must be less than or equal to the number of plants. The number of plant beds multiplied by the maximum number of species per plant bed must be greater than or equal to the number of plants.')

                            if valid:
                                # add in some vertical space
                                add_vertical_space(2)
                                if st.button('Generate Companion Plant Compatibility Matrix'):
                                    with st.spinner('generating companion plant compatibility matrix...'):
                                        st.session_state['generating_mat'] = True
                                        # now get compatibility matrix for companion planting
                                        time.sleep(1)
                                        extracted_mat, full_mat, plant_index_mapping= get_compatibility_matrix_2(st.session_state.input_plants_raw)
                                        print(extracted_mat)
                                        st.session_state.extracted_mat = extracted_mat
                                        st.session_state.full_mat = full_mat
                                        st.session_state.plant_index_mapping = plant_index_mapping
    # add in some vertical space
    add_vertical_space(4)

    # display the companion plant compatibility matrix
    if 'extracted_mat' in st.session_state:
        # add a title for the next section- companion plant compatibility matrix based on user input
        st.title('Your companion plant compatibility matrix')
        # make a container for this section
        container2 = st.container(border=True)
        with container2:
            col1, col2 = st.columns([8,4])
            # display the companion plant compatibility matrix
            with col2:
                st.write("Here is your companion plant compatibility matrix:")
                with st.expander("Show ugly compatibility matrix of 1's 0's and -1's"):
                    st.write(st.session_state.extracted_mat)
            with col1:
                st.write("Here is a network visualization of your companion plant compatibility matrix. It is color coded to show which plants are companions (green), antagonists (violetred), or neutral (grey).")
                plot_compatibility_with_agraph(st.session_state.input_plants_raw, st.session_state.full_mat)
                st.session_state['got_mat'] = True
        
        if 'got_mat' in st.session_state:
            # add in some vertical space
            add_vertical_space(4) 
            # make a container for this section
            container3 = st.container(border=True)
            with container3:   
                st.title('Optimizing companion planting with the genetic algorithm and AI')
                st.write("Now that we have your companion plant compatibility matrix, we can use optimization to maximize your harvest. We will use a genetic algorithm to determine the best way to plant your garden. The genetic algorithm will determine the best way to plant your garden by maximizing the number of companion plants and minimizing the number of antagonists.")
                st.write("Set the parameters for the genetic algorithm. Here is more info for your reference:")
                with st.form(key = "genetic_algorithm_form"):
                    col1, col2= st.columns([1,1])
                    with col2:
                        with st.expander("Show more information about the genetic algorithm parameters"):
                            st.subheader("Plant Optimization Heuristic Performance")
                            st.write("The genetic algorithm parameters impact the performance of the plant optimization heuristic in the following ways:")
                            st.markdown("- **Population Size**: A larger population size allows for a more diverse exploration of the solution space. However, it also increases computational complexity.")
                            st.markdown("- **Number of Generations**: Increasing the number of generations provides more opportunities for the algorithm to converge towards an optimal solution.")
                            st.markdown("- **Tournament Size**: A larger tournament size promotes stronger selection pressure and can lead to faster convergence, but it may also increase the risk of premature convergence.")
                            st.markdown("- **Crossover Rate**: A higher crossover rate increases the exploration capability by creating diverse offspring, potentially improving the algorithm's ability to escape local optima.")
                            st.markdown("- **Mutation Rate**: Mutation introduces random changes in individuals, helping to maintain diversity in the population and preventing premature convergence.")
                            # seed population rate
                            st.markdown("- **Seed Population Rate**: The seed population rate is the percentage of the population that is generated based on the LLM's interpretation of compatibility. The remaining percentage of the population is generated randomly. A higher seed population rate increases the likelihood that the genetic algorithm will converge towards a solution that is compatible.")
                    # Run the Genetic Algorithm
                    with col1:
                        st.subheader("Genetic Algorithm Parameters")
                        st.write("These parameters control the behavior of the genetic algorithm.")

                        # Genetic Algorithm parameters
                        st.session_state.population_size = st.slider("Population Size", min_value=500, max_value=3000, value=550,
                                                    help="The number of individuals in each generation of the genetic algorithm.")
                        st.session_state.num_generations = st.slider("Number of Generations", min_value=100, max_value=1500, value=500,
                                                    help="The total number of generations to evolve through.")
                        st.session_state.tournament_size = st.slider("Tournament Size", min_value=5, max_value=20, value=10,
                                                    help="The number of individuals competing in each tournament selection round.")
                        st.session_state.crossover_rate = st.slider("Crossover Rate", min_value=0.1, max_value=1.0, step=0.1, value=0.8,
                                                help="The probability of two individuals undergoing crossover to create offspring.")
                        st.session_state.mutation_rate = st.slider("Mutation Rate", min_value=0.01, max_value=0.9, step=0.01, value=0.3,
                                                help="The probability of an individual undergoing mutation.")
                        st.session_state.seed_population_rate = st.slider("Seed Population Rate", min_value=0.0, max_value=.02, step=0.001, value=0.08, 
                                                                          help="The percentage of the population that is generated based on the LLM's interpretation of compatibility. The remaining percentage of the population is generated randomly.")
                        
                        # 
                        # Run the genetic algorithm
                        if st.form_submit_button(label='Run Genetic Algorithm'):
                            grouping = genetic_algorithm_plants()
                            st.session_state.grouping = grouping

                # visualize the groupings
                # add in some vertical space
                add_vertical_space(4)
                # make a container for this section
                st.title(st.session_state.user_name + "'s optimized garden")
                st.header("Here are the optimized groupings of plants for your garden")
                container4 = st.container(border=True)
                with container4:
                    if 'grouping' in st.session_state:
                        visualize_groupings()
                        if 'best_fitness' in st.session_state:
                            # embed score.png
                            col1b, col2b = st.columns([2,11])
                            with col1b:
                                st.image("assets/score.png", caption=None, width = 160, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
                            with col2b:
                                #st.write("\n")
                                st.header("|  " + str(st.session_state.best_fitness))
                            st.write("The genetic algorithm converged towards a solution with a fitness score of " + str(st.session_state.best_fitness) + ".")
                        # Add vertical space
                        add_vertical_space(4)
                        # show plant care tips
                        st.header("Plant care tips")
                        with st.spinner('generating plant care tips...'):
                            st.write("Here are some plant care tips for your plants. Good luck!")
                            if 'plant_care_tips' not in st.session_state:
                                st.session_state.plant_care_tips = get_plant_care_tips(st.session_state.input_plants_raw)
                        styled_text = f'<div style="background-color: #2d5a59; color: white; padding: 10px; border-radius: 5px;">{st.session_state.plant_care_tips}</div>'
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
