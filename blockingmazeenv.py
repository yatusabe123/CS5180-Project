class BlockingMaze(object):
    """ Implementation of the Blocking Maze """
    def __init__(self):
        # We define the grid for the maze on the left in Example 8.2
        self.left_grid = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]])

        # We define the grid for the maze on the right in Example 8.2
        self.right_grid = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0]])

        # We initialize the grid with using the left maze first
        self.grid = self.left_grid.copy()
        # Save the size of the maze
        self.grid_height, self.grid_width = self.grid.shape

        # We define the observation space using all empty cells.
        # We consider all cells. Because the environment switch will results in change of the state space
        self.observation_space = np.argwhere(np.zeros((self.grid_height, self.grid_width)) == 0.0).tolist()

        # We define the action space
        self.action_space = {
            "up": np.array([-1, 0]),
            "down": np.array([1, 0]),
            "left": np.array([0, -1]),
            "right": np.array([0, 1])
        }
        self.action_names = ["up", "down", "left", "right"]

        # We define the start state
        self.start_location = [5, 3]

        # We define the goal state
        self.goal_location = [0, 8]

        # We find all wall locations in the grid
        self.walls = np.argwhere(self.grid == 1.0).tolist()

        # We define other useful variables
        self.agent_location = None
        self.action = None

    def reset(self, shift_maze=None):
        """
        Args:
            shift_maze (string): if None, reset function simply set the agent back to the start location
                                 if left, reset function will switch to the maze on the left
                                 if right, reset function will switch to the maze on the right
        """
        if shift_maze is not None:
            if shift_maze == "left":
                self.grid = self.left_grid.copy()
            elif shift_maze == "right":
                self.grid = self.right_grid.copy()
            else:
                raise Exception("Invalid shift operation")

            # reset the shape
            self.grid_height, self.grid_width = self.grid.shape

            # reset the walls
            self.walls = np.argwhere(self.grid == 1.0).tolist()

        # We reset the agent location to the start state
        self.agent_location = self.start_location

        # We set the information
        info = {}
        return self.agent_location, info

    def step(self, action):
        """
        Args:
            action (string): name of the action (e.g., "up"/"down"/"left"/"right")
        """
        # Convert the agent's location to array
        loc_arr = np.array(self.agent_location)

        # Convert the abstract action to movement array
        act_arr = self.action_space[action]

        # Compute the next location
        next_agent_location = np.clip(loc_arr + act_arr,
                                      a_min=np.array([0, 0]),
                                      a_max=np.array([self.grid_height - 1, self.grid_width - 1])).tolist()

        # Check if it crashes into a wall
        if next_agent_location in self.walls:
            # If True, the agent keeps in the original state
            next_agent_location = self.agent_location

        # Compute the reward
        reward = 1 if next_agent_location == self.goal_location else 0

        # Check the termination
        terminated = True if reward == 1 else False

        # Update the action location
        self.agent_location = next_agent_location
        self.action = action

        return next_agent_location, reward, terminated, False, {}

    def render(self):
        # plot the agent and the goal
        # agent = 1
        # goal = 2
        plot_arr = self.grid.copy()
        plot_arr[self.agent_location[0], self.agent_location[1]] = 2
        plot_arr[self.goal_location[0], self.goal_location[1]] = 3
        plt.clf()
        plt.title(f"state={self.agent_location}, act={self.action}")
        plt.imshow(plot_arr)
        plt.show(block=False)
        plt.pause(0.01)
