import random
import numpy as np
import streamlit as st

def genetic_algorithm_plants():
    # Define the compatibility matrix
    compatibility_matrix = st.session_state.extracted_mat

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


    def generate_initial_population():
        population = []
        species_pool = user_plants.copy()  # Create a pool of species
        random.shuffle(species_pool)

        for _ in range(population_size):
            grouping = []
            species_copy = species_pool.copy()

            # Distribute the remaining species evenly among the plant beds
            for _ in range(num_plant_beds):
                if not species_copy:
                    break

                num_species = min_species_per_bed
                if len(species_copy) < min_species_per_bed:
                    num_species = len(species_copy)

                species = species_copy[:num_species]
                grouping.append(species)
                species_copy = species_copy[num_species:]

            # Fill any remaining empty plant beds with randomly selected species
            for _ in range(num_plant_beds - len(grouping)):
                if not species_copy:
                    break

                num_species = 1
                species = species_copy[:num_species]
                grouping.append(species)
                species_copy = species_copy[num_species:]

            population.append(grouping)

        return population

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

    # Calculate the fitness score of a grouping
    def calculate_fitness(grouping):
        score = 0
        for bed1 in range(num_plant_beds):
            for bed2 in range(bed1 + 1, num_plant_beds):
                for species1 in grouping[bed1]:
                    for species2 in grouping[bed2]:
                        species1_index = user_plants.index(species1)
                        species2_index = user_plants.index(species2)
                        score += compatibility_matrix[species1_index][species2_index]
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

        best_grouping = max(population, key=calculate_fitness)
        best_fitness = calculate_fitness(best_grouping)
        print(f"Best Grouping: {best_grouping}")
        print(f"Fitness Score: {best_fitness}")
        # st.write(f"Best Grouping: {best_grouping}")
        # st.write(f"Fitness Score: {best_fitness}")
        return best_grouping
    
    best_grouping = genetic_algorithm()
    return best_grouping