import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from streamlit_agraph import agraph, Node, Edge, Config


def plot_compatibility(plants, compatibility_matrix, is_mini=False):
    # Create the graph
    G = nx.Graph()
    G.add_nodes_from(plants)
    for i in range(len(plants)):
        for j in range(i + 1, len(plants)):
            if compatibility_matrix[i][j] == 0:
                G.add_edge(plants[i], plants[j], color="dimgrey")
            else:
                G.add_edge(
                    plants[i],
                    plants[j],
                    color="green"
                    if compatibility_matrix[i][j] == 1
                    else "mediumvioletred",
                )

    # Generate positions for the nodes
    pos = nx.spring_layout(G)

    # Create node trace
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        text=list(G.nodes()),
        mode="markers+text",
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            size=40,
            color="lightblue",
            line_width=2,
        ),
    )

    # Create edge trace
    edge_trace = go.Scatter(
        x=[], y=[], line=dict(width=1, color="dimgrey"), hoverinfo="none", mode="lines"
    )

    # Add coordinates to edge trace
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace["x"] += tuple([x0, x1, None])
        edge_trace["y"] += tuple([y0, y1, None])

    # Create edge traces for colored edges
    edge_traces = []
    edge_legend = set()  # Set to store unique edge colors
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        color = edge[2]["color"]
        trace = go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(width=2, color=color),
            hoverinfo="none",
        )
        edge_traces.append(trace)
        edge_legend.add(color)  # Add edge color to the set

    # Create layout
    layout = go.Layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, *edge_traces, node_trace], layout=layout)

    # Create custom legend for edge colors
    custom_legend = []
    legend_names = ["Neutral", "Negative", "Positive"]
    legend_colors = ["dimgrey", "mediumvioletred", "green"]

    for name, color in zip(legend_names, legend_colors):
        custom_legend.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=color),
                name=f"{name}",
                showlegend=True,
                hoverinfo="none",
            )
        )
    if is_mini == False:
        # Create layout for custom legend figure
        legend_layout = go.Layout(
            title="Plant Compatibility Network Graph",
            showlegend=True,
            margin=dict(b=1, t=100),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=120,
            legend=dict(
                title="Edge Colors",
                orientation="h",
                x=-1,
                y=1.1,
                bgcolor="rgba(0,0,0,0)",
            ),
        )
    else:
        fig.update_layout(
            autosize=False,
            width=300,
            height=300,
        )

    if is_mini == False:
        # Create figure for custom legend
        legend_fig = go.Figure(data=custom_legend, layout=legend_layout)
        # Render the custom legend using Plotly in Streamlit
        st.plotly_chart(legend_fig, use_container_width=True)

    # Render the graph using Plotly in Streamlit
    st.plotly_chart(fig)


# this is not used as it needs to be refactored and is not working as intended
def show_plant_tips():
    tips_string = st.session_state.plant_care_tips

    tips_list = tips_string.split("\n")
    num_tips = len(tips_list)
    st.markdown(
        "## Plant Care Tips for your plants: "
        + str(st.session_state.input_plants_raw)
        + "\n\n"
        + st.session_state.plant_care_tips
    )


def visualize_groupings_sankey():
    groupings = st.session_state.grouping
    compatibility_matrix = st.session_state.extracted_mat
    plant_list = st.session_state.input_plants_raw

    for i, bed_species in enumerate(groupings):
        st.subheader(f"Plant Bed {i + 1}")

        # Create the nodes
        nodes = []
        for species in bed_species:
            nodes.append(species)

        # Create the links
        links = []
        for j, species1 in enumerate(bed_species):
            for k, species2 in enumerate(bed_species):
                if j < k:
                    species1_index = plant_list.index(species1)
                    species2_index = plant_list.index(species2)
                    compatibility = compatibility_matrix[species1_index][species2_index]

                    if compatibility == 1:
                        color = "green"
                    elif compatibility == -1:
                        color = "pink"
                    else:
                        color = "grey"

                    links.append(
                        dict(source=j, target=k, value=compatibility, color=color)
                    )

        # Create the Sankey diagram
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(label=nodes, color="lightblue"),
                    link=dict(
                        source=[link["source"] for link in links],
                        target=[link["target"] for link in links],
                        value=[link["value"] for link in links],
                        color=[link["color"] for link in links],
                    ),
                )
            ]
        )

        # Set the layout properties
        layout = go.Layout(
            plot_bgcolor="black", paper_bgcolor="black", title_font=dict(color="white")
        )

        # Set the figure layout
        fig.update_layout(layout)

        # Render the Sankey diagram in Streamlit
        st.plotly_chart(fig)


def visualize_groupings():
    groupings = st.session_state.grouping
    compatibility_matrix = st.session_state.extracted_mat
    plant_list = st.session_state.input_plants_raw

    def generate_grouping_matrices(groupings, compatibility_matrix, plant_list):
        grouping_matrices = []
        for grouping in groupings:
            indices = [plant_list.index(plant) for plant in grouping]
            submatrix = [[compatibility_matrix[i][j] for j in indices] for i in indices]
            grouping_matrices.append(submatrix)
        return grouping_matrices

    grouping_matrices = generate_grouping_matrices(
        groupings, compatibility_matrix, plant_list
    )
    for i, submatrix in enumerate(grouping_matrices):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(f"Plant Bed {i + 1}")
            st.write("Plant List")
            st.write(groupings[i])
        with col2:
            plot_compatibility_with_agraph(
                groupings[i], st.session_state.full_mat, is_mini=True
            )


def plot_compatibility_with_agraph(plants, compatibility_matrix, is_mini=False):
    # Create nodes and edges for the graph
    nodes = []
    edges = []

    # Function to get the image URL for a plant
    def get_image_url(plant_name):
        index = st.session_state.plant_list.index(plant_name)
        image_path = f"https://github.com/4dh/GRDN/blob/dev/src/assets/plant_images/plant_{index}.png?raw=true"
        print(image_path)
        return image_path

    size_n = 32 if not is_mini else 24
    # Create nodes with images
    for plant in plants:
        nodes.append(
            Node(
                id=plant,
                label=plant,
                # make text bigger
                font={"size": 20},
                # spread nodes out
                scaling={"label": {"enabled": True}},
                size=size_n,
                shape="circularImage",
                image=get_image_url(plant),
            )
        )

    # Create edges based on compatibility
    # for i in range(len(st.session_state.plant_list)):
    # loop through all plants in raw long list and find the index of the plant in the plant list to get relevant metadata. skip if we are looking at the same plant
    for i, i_p in enumerate(st.session_state.plant_list):
        for j, j_p in enumerate(st.session_state.plant_list):
            if i != j:
                # check if plants[i] and plants[j]  are in input_plants_raw
                # print(st.session_state.input_plants_raw)
                if is_mini == False:
                    length_e = 300
                else:
                    length_e = 150

                if (
                    i_p in st.session_state.input_plants_raw
                    and j_p in st.session_state.input_plants_raw
                ):
                    # use the compatibility matrix and the plant to index mapping to determine the color of the edge
                    if compatibility_matrix[i][j] == 1:
                        color = "green"
                        edges.append(
                            Edge(
                                source=i_p,
                                target=j_p,
                                width=3.5,
                                type="CURVE_SMOOTH",
                                color=color,
                                length=length_e,
                            )
                        )
                        print(i, j, i_p, j_p, color)
                    elif compatibility_matrix[i][j] == -1:
                        color = "mediumvioletred"
                        edges.append(
                            Edge(
                                source=i_p,
                                target=j_p,
                                width=3.5,
                                type="CURVE_SMOOTH",
                                color=color,
                                length=length_e,
                            )
                        )
                        print(i, j, i_p, j_p, color)

                    else:
                        color = "dimgrey"
                        edges.append(
                            Edge(
                                source=i_p,
                                target=j_p,
                                width=0.2,
                                type="CURVE_SMOOTH",
                                color=color,
                                length=length_e,
                            )
                        )
                        print(i, j, i_p, j_p, color)

    # Configuration for the graph
    config = Config(
        width=650 if not is_mini else 400,
        height=400 if not is_mini else 400,
        directed=False,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=True,
        maxZoom=5,
        minZoom=0.2,
        initialZoom=4,
    )

    # Handling for non-mini version
    if not is_mini:
        # Create custom legend for edge colors at the top of the page
        custom_legend = []
        legend_names = ["Neutral", "Negative", "Positive"]
        legend_colors = ["dimgrey", "mediumvioletred", "green"]

        for name, color in zip(legend_names, legend_colors):
            custom_legend.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(color=color),
                    name=name,
                    showlegend=True,
                    hoverinfo="none",
                )
            )

        # Create layout for custom legend figure
        legend_layout = go.Layout(
            title="Plant Compatibility Network Graph",
            showlegend=True,
            margin=dict(b=1, t=100),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=120,
            legend=dict(
                title="Edge Colors",
                orientation="h",
                # make it appear above the graph
                x=-1,
                y=1.1,
                bgcolor="rgba(0,0,0,0)",
            ),
        )

        # Create figure for custom legend
        legend_fig = go.Figure(data=custom_legend, layout=legend_layout)

        # Render the custom legend using Plotly in Streamlit
        st.plotly_chart(legend_fig, use_container_width=True)

    # Render the graph using streamlit-agraph
    return_value = agraph(nodes=nodes, edges=edges, config=config)
