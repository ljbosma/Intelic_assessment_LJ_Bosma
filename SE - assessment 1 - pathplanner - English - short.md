# Assessment 1

Avalor AI is working on an advanced drone swarm that can independently monitor an area. To be able to search an area effectively, we use a world model. An example of this can be a grid, divided into equal planes, where each plane in the grid is provided with a numerical value. This value then might represent the information contained in that plane. When a drone flies a path, it adds up the values of all the planes it passes, the result is the total value of the path. The longer you have not been somewhere, the greater the chance that the situation has changed. You can represent this by slowly increasing the value of the plane to the original value after you have visited a plane. Adjust your model and algorithm to deal with this.

### Assignment

Write a path planning algorithm that returns the most valuable possible path for a given N, t, T, (x, y).

Given are:

- N: Size of the grid to be searched: N x N.
- t: Total number of discrete time steps
- T: Maximum duration of the algorithm in milliseconds.
- (x, y): Index of drone starting position

All values are discrete numbers.

This applies:

- It costs a drone 1 time step to move to an adjacent space. This can be done horizontally, vertically and diagonally.
- The presence of a drone in the area results in achieving the observation score for the area in question. The drone can achieve the score of a plane multiple times per run of the algorithm, as it increases over time.
- The algorithm must always end in the given time limit T.
- The initial score of each plane is given in the data, but the speed at which the score increases can be chosen yourself.

### Conditions/tips:

- You are free to choose any programming language.
- Write the algorithm for a single drone. You don't have to consider functionality for a drone swarm for now.
- Your algorithm must return the path plus the total score achieved within time T.
- Write a solution that also works for larger values of N and t within the available time T.
- The appendix contains three example levels of different sizes that you can use.
- We do not look at the final score, but at your methodology, your explanation and quality of the solution.

Note: If you are applying for Software Engineer, or Fullstack Developer we are less interested in the algorithm. More in the structure of the project. 

### Extra challenge:

If you are looking for an extra challenge, you can take the following challenge:

- Swarm algorithm: In reality, of course, we fly several drones at the same time. There is no added value if drones search the same section. Adjust your model and algorithm to deal with this.
   - The drones start at random locations in the grid.
   - The algorithm must be able to handle X numbers of drones. As input you get X different start_positions and as a result it returns X number of paths and 1 total score.
   - Use your own imagination to make the necessary assumptions.

### Delivery and assessment

We would like to receive a link to your project code no later than one hour before the start of the assessment. During the assessment you will talk us through your code and we will ask some questions. Duration: +/- 40 min.

During the assessment, your code quality, your explanation of the algorithm, and your answers to follow-up questions will be concidered.

If you have any questions, please feel free to call or email us.
