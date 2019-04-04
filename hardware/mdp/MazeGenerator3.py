import sys
from random import shuffle
import numpy as np

def shuffled(x):
    y = list(x)
    shuffle(y)
    return y

DIRECTIONS = (
    (0, -1),
    (0, 1),
    (1, 0),
    (-1, 0),
)

def make_maze(width, height, cellsize):
    cellsize1 = cellsize+1 # cellsize including one wall
    field_width = width*cellsize1+1
    field_height = height*cellsize1+1
    field = [0]*(field_width*field_height)
    stack = [(0, 0, shuffled(DIRECTIONS))]
    while stack:
        x, y, directions = stack[-1]
        dx, dy = directions.pop()
        # no other ways to go from here
        if not directions:
            stack.pop()
        # new cell
        nx = x+dx
        ny = y+dy
        # out of bounds
        if not (0 <= nx < width and 0 <= ny < height):
            continue
        # index of new cell in field
        fx = 1+nx*cellsize1
        fy = 1+ny*cellsize1
        fi = fx+fy*field_width
        # already visited
        if field[fi]:
            continue
        # tear down walls
        if dx > 0:
            a = -1
            b = field_width
        elif dx < 0:
            a = cellsize
            b = field_width
        elif dy > 0:
            a = -field_width
            b = 1
        else:
            a = cellsize*field_width
            b = 1
        for offset in range(cellsize):
            field[fi+a+b*offset] = 1
        # clear cell
        for y in range(0, cellsize):
            for x in range(0, cellsize):
                field[fi+x+y*field_width] = 1
        # visit cell
        stack.append([nx, ny, shuffled(DIRECTIONS)])
    
    return field

def GenerateMaze(height, width, withoutWall=False):
    
    cellsize = 1
    fields = make_maze(width, height, cellsize)
    w = (cellsize+1)*width+1
    h = (cellsize+1)*height+1

    maze = np.zeros((h, w), dtype=np.int)
    possibleGoals = []

    for y in range(h):
        maze[y, :] = np.array([x and "0" or "1" for x in fields[y*w:y*w+w]], dtype=np.int)
        possibleGoals.extend(np.flatnonzero(maze[y, :] == 0) + y * w)

    goalState = possibleGoals[np.random.randint(0, len(possibleGoals))]
    goalState = (int(goalState / w), int(goalState % w))

    if withoutWall:
        sh = maze.shape
        maze = maze[1:sh[0]-1, 1:sh[1]-1]
        goalState = (goalState[0] - 1, goalState[1] - 1)
    
    return maze, goalState

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: %s width height cellsize" % sys.argv[0])
    width, height, cellsize = map(int, sys.argv[1:])
#    fields = make_maze(width, height, cellsize)
#    w = (cellsize+1)*width+1
#    h = (cellsize+1)*height+1

#    for y in xrange(h):
#        print "".join(map(lambda x: x and "0" or "1", fields[y*w:y*w+w]))

    print(GenerateMaze(height, width))
