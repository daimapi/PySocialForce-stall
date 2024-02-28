import heapq
import numpy as np

def dijkstra(graph, start, finish):
    # Initialize distances dictionary with infinity for all nodes except start node
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    # Priority queue to store nodes with their distances
    priority_queue = [(0, start)]

    # Dictionary to store the previous node for each node on the shortest path
    previous = {}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # If current node is the finish point, reconstruct the shortest path
        if current_node == finish:
            path = []
            while current_node in previous:
                path.insert(0, current_node)
                current_node = previous[current_node]
            path.insert(0, start)
            return distances[finish], path

        # If current distance is greater than the distance already found for this node, skip
        if current_distance > distances[current_node]:
            continue

        # Iterate through neighbors of current node
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            # If the new distance is smaller, update the distance and previous node
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    # If no path is found, return None
    return None, None

# Example graph dictionary
graph = {
    #'s': {'a': 8, 'b': 4},
    'a': {'b': 2, 'd': 1},
    'b': {'a': 2, 'c': 3},
    'c': {'b': 3, 'd': 4},
    'd': {'a': 1, 'c': 4}
}

# Example start and finish points
start_point = 'c'
finish_point = 'a'

# Find the shortest path using Dijkstra's Algorithm
shortest_distance, shortest_path = dijkstra(graph, start_point, finish_point)

if shortest_path is not None:
    print("Shortest distance from", start_point, "to", finish_point, "is", shortest_distance)
    print("next is", shortest_path[1])
else:
    print("No path found from", start_point, "to", finish_point)

#
#a = np.array([[1,2,3,4],
#              [5,6,7,8]])
#c = a == np.array([1,2,3,4])
#ans = []
#for _ in c:
#    ans.append(_.sum() == 4)
#print(ans)
print(len(['[36.337552 35.14259 ]', '[28.95373 35.14259]']))
#0