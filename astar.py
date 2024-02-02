import heapq


def astar(start, end, grid_size, cost_func):
    # Define the possible movements (up, down, left, right)
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Create a priority queue (heap) to store nodes to be explored
    open_list = [(0, start)]

    # Create a dictionary to keep track of the cost to reach each node
    g_scores = {start: 0}

    # Create a dictionary to store the parent node of each explored node
    parents = {}

    # Create a function to calculate the Manhattan distance heuristic
    def heuristic(node):
        return abs(node[0] - end[0]) + abs(node[1] - end[1])

    while open_list:
        # Get the node with the lowest f_score (g_score + heuristic)
        current_cost, current_node = heapq.heappop(open_list)

        if current_node == end:
            # Reconstruct the path from the end to the start
            path = []
            while current_node in parents:
                path.append(current_node)
                current_node = parents[current_node]
            path.append(start)
            path.reverse()
            return path

        for move in moves:
            neighbor = (current_node[0] + move[0], current_node[1] + move[1])

            if 0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1]:
                # Calculate the tentative g_score for the neighbor
                tentative_g_score = g_scores[current_node] + cost_func(
                    current_node, neighbor
                )

                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    # Update the g_score and f_score for the neighbor
                    g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor)

                    # Add the neighbor to the open list with the new f_score
                    heapq.heappush(open_list, (f_score, neighbor))

                    # Update the parent for the neighbor
                    parents[neighbor] = current_node

    # If no path is found, return an empty list
    return []


# for testing
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean_distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# cost function that does not like right edges
def cost_func(a, b):
    if b[1] == 4:
        return 100
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


if __name__ == "__main__":
    # Example usage:
    # Define a cost function, for example, the Manhattan distance between two points
    start = (0, 0)
    end = (4, 4)
    grid_size = (5, 5)

    path = astar(start, end, grid_size, manhattan_distance)
    print("Manhattan Path: ", path)

    path = astar(start, end, grid_size, euclidean_distance)
    print("Euclidean Path: ", path)

    path = astar(start, end, grid_size, cost_func)
    print("Cost Function Path: ", path)
