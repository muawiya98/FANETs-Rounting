import numpy as np
import math
import config as cf


class Mobility:

    def __init__(self, movement_func):
        # only random walk.

        self.movement_func = movement_func

    def __call__(self, *args):
        if self.movement_func == 'spiral':
            raise ValueError('Not working yet.')
            # return self.move_spiral(*args)
        elif self.movement_func == 'random':
            return self.move_random(*args[1:])
        elif self.movement_func == 'random_sub_grid':
            return self.move_random_in_subgrid(*args[1:])

    @staticmethod
    def move_random(org_x, org_y, plane_name):
        x = cf.seed_simulator.uniform(- cf.WALKING_SPEED[plane_name], cf.WALKING_SPEED[plane_name])
        y = cf.seed_simulator.uniform(- cf.WALKING_SPEED[plane_name], cf.WALKING_SPEED[plane_name])
        return max(cf.FORBIDDEN_EDGES, min(x + org_x, cf.AREA_WIDTH-cf.FORBIDDEN_EDGES)),\
               max(cf.FORBIDDEN_EDGES, min(y + org_y, cf.AREA_LENGTH-cf.FORBIDDEN_EDGES))

    @staticmethod
    def move_random_in_subgrid(org_x, org_y, plane_name):
        """
        Moves the node randomly within the boundaries of its current subgrid.

        org_x, org_y    : original coordinates of the node
        plane_name      : the plane type (used to look up WALKING_SPEED)
        """
        # Determine the subgrid indices (row and column) for the current position
        subgrid_col = int(org_x // cf.SUB_GRID_SIZE)
        subgrid_row = int(org_y // cf.SUB_GRID_SIZE)

        # Calculate the bottom-left corner of the subgrid
        subgrid_x = subgrid_col * cf.SUB_GRID_SIZE
        subgrid_y = subgrid_row * cf.SUB_GRID_SIZE

        # Generate random movement for x and y within the plane's walking speed
        x_move = cf.seed_simulator.uniform(-cf.WALKING_SPEED[plane_name], cf.WALKING_SPEED[plane_name])
        y_move = cf.seed_simulator.uniform(-cf.WALKING_SPEED[plane_name], cf.WALKING_SPEED[plane_name])

        # Calculate new positions after the random movement
        new_x = org_x + x_move
        new_y = org_y + y_move

        # Ensure the new position stays within the subgrid's boundaries
        new_x = max(subgrid_x + cf.FORBIDDEN_EDGES, min(new_x, subgrid_x + cf.SUB_GRID_SIZE - cf.FORBIDDEN_EDGES))
        new_y = max(subgrid_y + cf.FORBIDDEN_EDGES, min(new_y, subgrid_y + cf.SUB_GRID_SIZE - cf.FORBIDDEN_EDGES))

        return new_x, new_y
