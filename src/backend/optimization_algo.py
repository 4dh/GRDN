import random
import numpy as np
import streamlit as st

# import all functions from src.backend.chatbot
from backend.chatbot import *

def genetic_algorithm_plants():
    # Define the compatibility matrix
    compatibility_matrix = st.session_state.full_mat
    # Define the list of plants
    plant_list = st.session_state.plant_list

    # Define the user-selected plants, number of plant beds, and constraints
    user_plants = st.session_state.input_plants_raw
    num_plant_beds = st.session_state.n_plant_beds
    # 1 <= min_species_per_bed <= max_species_per_bed <= len(user_plants)
    min_species_per_bed = st.session_state.min_species
    # max_species_per_bed >= floor(length(user_plants)-(min_species_per_bed*num_plant_beds-1) & max_species_per_bed <= len(user_plants)  
    max_species_per_bed = st.session_state.max_species


    # Genetic Algorithm parameters
    population_size = st.session_state.population_size
    num_generations = st.session_state.num_generations
    tournament_size = st.session_state.tournament_size
    crossover_rate = st.session_state.crossover_rate
    mutation_rate = st.session_state.mutation_rate
    seed_population_rate = st.session_state.seed_population_rate 


    def generate_initial_population():
        population = []

        # Add seed groupings to the population, validated and replaced as necessary
        num_seeds = int(population_size * st.session_state.seed_population_rate)  # 10% of the population as seeds
        # we generate just one seed grouping for this beta language model suggestion feature
        seed_grouping = get_language_model_suggestions()
        valid_seed_grouping = validate_and_replace(seed_grouping)
        population.append(valid_seed_grouping)


        # Fill the rest of the population with random groupings, also validated and replaced
        while len(population) < population_size:
            random_grouping = generate_random_grouping()
            valid_random_grouping = validate_and_replace(random_grouping)
            population.append(valid_random_grouping)

        return population


    def generate_random_grouping():
        random.shuffle(user_plants)
        remaining_plants = user_plants.copy()
        grouping = []

        total_plants = len(user_plants)
        plants_per_bed = total_plants // num_plant_beds
        extra_plants = total_plants % num_plant_beds

        for bed_index in range(num_plant_beds):
            if bed_index < extra_plants:
                # Distribute extra plants among the first few beds
                num_species_in_bed = plants_per_bed + 1
            else:
                num_species_in_bed = plants_per_bed

            # Ensure the bed size is within the min and max constraints
            num_species_in_bed = max(min_species_per_bed, min(num_species_in_bed, max_species_per_bed))

            bed = remaining_plants[:num_species_in_bed]
            remaining_plants = remaining_plants[num_species_in_bed:]
            grouping.append(bed)

        return grouping



    # Perform crossover between two parents, preserving at least one occurrence of each plant
    def crossover(parent1, parent2):
        if random.random() < crossover_rate:
            crossover_point = random.randint(1, num_plant_beds - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]

            # Ensure each plant appears at least once in the offspring
            for plant in user_plants:
                if all(plant not in bed for bed in child1):
                    # Find a bed with fewer species and add the missing plant
                    min_bed_index = min(range(len(child1)), key=lambda i: len(child1[i]))
                    child1[min_bed_index].append(plant)
                if all(plant not in bed for bed in child2):
                    # Find a bed with fewer species and add the missing plant
                    min_bed_index = min(range(len(child2)), key=lambda i: len(child2[i]))
                    child2[min_bed_index].append(plant)

            return child1, child2
        else:
            return parent1, parent2
        
    # Perform mutation on an individual, ensuring no bed exceeds the maximum species constraint
    def mutate(individual):
        if random.random() < mutation_rate:
            mutated_bed = random.randint(0, num_plant_beds - 1)
            species_in_bed = individual[mutated_bed]

            # Remove excess species if there are more than the maximum constraint
            if len(species_in_bed) > max_species_per_bed:
                species_in_bed = random.sample(species_in_bed, max_species_per_bed)

            # Add missing plants by performing swaps between current species and missing plants
            missing_plants = [plant for plant in user_plants if plant not in species_in_bed]
            num_missing_plants = min(len(missing_plants), max_species_per_bed - len(species_in_bed))
            for _ in range(num_missing_plants):
                swap_species = random.choice(missing_plants)
                missing_plants.remove(swap_species)
                species_in_bed.append(swap_species)
                species_in_bed.remove(random.choice(species_in_bed))

            individual[mutated_bed] = species_in_bed

        return individual

    # Calculate the fitness score of the grouping
    def calculate_fitness(grouping):
        positive_reward_factor = 1000  # Adjust this to increase the reward for compatible species
        negative_penalty_factor = 2000  # Adjust this to increase the penalty for incompatible species

        # Define penalties for not meeting constraints
        penalty_for_exceeding_max = 500  # Adjust as needed
        penalty_for_not_meeting_min = 500  # Adjust as needed
        penalty_for_not_having_all_plants = 1000  # Adjust as needed

        score = 0
        # Iterate over each plant bed
        for bed in grouping:
            for i in range(len(bed)):
                for j in range(i + 1, len(bed)):
                    # get the plant name
                    species1_name = bed[i]
                    species2_name = bed[j]
                    species1_index = plant_list.index(species1_name)
                    species2_index = plant_list.index(species2_name)

                    # Compatibility score between two species in the same bed
                    compatibility_score = compatibility_matrix[species1_index][species2_index]
                    
                    if compatibility_score > 0:
                        # Positive reward for compatible species
                        score += compatibility_score*positive_reward_factor
                    elif compatibility_score < 0:
                        # Negative penalty for incompatible species
                        score += compatibility_score*negative_penalty_factor


        # Apply penalties for not meeting constraints
        if len(bed) > max_species_per_bed:
            score -= penalty_for_exceeding_max
        if len(bed) < min_species_per_bed:
            score -= penalty_for_not_meeting_min
        if len(set(plant for bed in grouping for plant in bed)) < len(user_plants):
            score -= penalty_for_not_having_all_plants

        return score


    # Perform tournament selection
    def tournament_selection(population):
        selected = []
        for _ in range(population_size):
            participants = random.sample(population, tournament_size)
            winner = max(participants, key=calculate_fitness)
            selected.append(winner)
        return selected

    # Perform replacement of the population with the offspring, ensuring maximum species constraint is met
    def replacement(population, offspring):
        sorted_population = sorted(population, key=calculate_fitness, reverse=True)
        sorted_offspring = sorted(offspring, key=calculate_fitness, reverse=True)

        # Adjust the offspring to meet the maximum species constraint
        adjusted_offspring = []
        for individual in sorted_offspring:
            for bed_idx in range(num_plant_beds):
                species_in_bed = individual[bed_idx]
                if len(species_in_bed) > max_species_per_bed:
                    species_in_bed = random.sample(species_in_bed, max_species_per_bed)
                individual[bed_idx] = species_in_bed
            adjusted_offspring.append(individual)

        return sorted_population[:population_size - len(adjusted_offspring)] + adjusted_offspring

    # Genetic Algorithm main function
    def genetic_algorithm():
        population = generate_initial_population()

        for generation in range(num_generations):
            print(f"Generation {generation + 1}")

            selected_population = tournament_selection(population)
            offspring = []

            for _ in range(population_size // 2):
                parent1 = random.choice(selected_population)
                parent2 = random.choice(selected_population)
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1)
                child2 = mutate(child2)
                offspring.extend([child1, child2])

            population = replacement(population, offspring)
            # Validate and replace any missing plants in the new population
            population = [validate_and_replace(grouping) for grouping in population]


        best_grouping = max(population, key=calculate_fitness)
        best_grouping = validate_and_replace(best_grouping)
        best_fitness = calculate_fitness(best_grouping)
        print(f"Best Grouping: {best_grouping}")
        print(f"Fitness Score: {best_fitness}")
        st.session_state.best_grouping = best_grouping
        st.session_state.best_fitness = best_fitness
        # st.write(f"Best Grouping: {best_grouping}")
        # st.write(f"Fitness Score: {best_fitness}")
        return best_grouping
    
    # def validate_and_replace(grouping):
    #     print("Grouping structure before validation:", grouping)
    #     all_plants = set(user_plants)
    #     for bed in grouping:
    #         all_plants -= set(bed)

    #     # Replace missing plants
    #     for missing_plant in all_plants:
    #         replaced = False
    #         for bed in grouping:
    #             if len(set(bed)) != len(bed):  # Check for duplicates
    #                 for i, plant in enumerate(bed):
    #                     if bed.count(plant) > 1:  # Found a duplicate
    #                         bed[i] = missing_plant
    #                         replaced = True
    #                         break
    #             if replaced:
    #                 break

    #         # If no duplicates were found, replace a random plant
    #         if not replaced:
    #             random_bed = random.choice(grouping)
    #             random_bed[random.randint(0, len(random_bed) - 1)] = missing_plant

    #     return grouping
    
    ############
    ############ experimental

    def adjust_grouping(grouping):
    # Determine the plants that are missing in the grouping
        plants_in_grouping = set(plant for bed in grouping for plant in bed)
        missing_plants = set(user_plants) - plants_in_grouping

        for missing_plant in missing_plants:
            # Find a bed that can accommodate the missing plant without exceeding max_species_per_bed
            suitable_bed = next((bed for bed in grouping if len(bed) < max_species_per_bed), None)
            if suitable_bed is not None:
                suitable_bed.append(missing_plant)
            else:
                # If no suitable bed is found, replace a random plant in a random bed
                random_bed = random.choice(grouping)
                random_bed[random.randint(0, len(random_bed) - 1)] = missing_plant

        # Ensure min_species_per_bed and max_species_per_bed constraints
        for bed in grouping:
            while len(bed) < min_species_per_bed:
                additional_plant = random.choice([plant for plant in user_plants if plant not in bed])
                bed.append(additional_plant)
            while len(bed) > max_species_per_bed:
                bed.remove(random.choice(bed))

        return grouping
    
    def validate_and_replace(grouping):
        best_grouping = None
        best_fitness = float('-inf')

        for _ in range(5):  # Generate 5 different configurations
            temp_grouping = [bed.copy() for bed in grouping]
            temp_grouping = adjust_grouping(temp_grouping)
            current_fitness = calculate_fitness(temp_grouping)

            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_grouping = temp_grouping

        return best_grouping



    ############
    def get_language_model_suggestions():
        # Placeholder for your implementation
        # This should return a list of seed groupings based on the compatibility matrix
        st.session_state.seed_groupings = get_seed_groupings_from_LLM()
        return st.session_state.seed_groupings


    
    best_grouping = genetic_algorithm()
    return best_grouping